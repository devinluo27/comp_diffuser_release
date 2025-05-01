from collections import namedtuple
import numpy as np
import torch, pdb, random, math

from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.buffer import ReplayBuffer
import diffuser.utils as utils

## 3 elem 'obs_trajs act_trajs conditions'
from diffuser.datasets.comp.comp_dataset_v1 import Batch_v1
from diffuser.datasets.ogb_dset.ogb_utils import ogb_load_env, ogb_sequence_dataset
from diffuser.datasets.ogb_dset.ogb_buffer_v2 import OgB_ReplayBuffer_V2

class OgB_SeqDataset_V2(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True,
        dset_h5path=None,
        dataset_config={},
        ):
        """
        Compared to v1, we add two traj padding methods in this v2 dataset because
        ** The last part of trajs of the OGBench stitch datasets (especially Antmaze)
        stays at the same cell, we need to make the start chunk has similar behavior to enable 
        more effective stitching.

        some assertation for the path len below is only for sanity check, might be ok to remove them

        """
        ## maze2d_set_terminals
        self.env = env = ogb_load_env(env)
        env.len_seg = dataset_config.get('len_seg', None)
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        # self.env = env = ogb_load_env(env) ## no need
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.dataset_config = dataset_config
        
        ## -----------------------
        # pdb.set_trace() ## check env is correct
        assert preprocess_fns == [], 'no need'

        ## call get_dataset inside
        env.dset_h5path = dset_h5path
        self.obs_select_dim = dataset_config['obs_select_dim'] 

        self.dset_type = dataset_config['dset_type'] # 'ours')

        ## a list of dict, e.g., len is 5000==1M/200
        data_d_list = ogb_sequence_dataset(env, self.preprocess_fn, dataset_config)


        def_pathlen = 400 if 'human' in env.name.lower() else 200
        def_pathlen = 500 if 'explore' in env.name else def_pathlen
        def_pathlen = 500 if 'antsoccer-medium' in env.name else def_pathlen

        ## ------------ Dec 29 New Padding Setup -----------
        ## a general padding option, while pad_type is just for 'buf'
        self.pad_option_2 = dataset_config.get('pad_option_2', None)
        assert self.pad_option_2 in [None, 'buf', 'get_item']

        if self.use_padding:
            assert self.pad_option_2 in ['buf', 'get_item']
        
        
        self.max_ori_path_len = dataset_config.get('max_ori_path_len', None)

        # pdb.set_trace() ##

        ## ------- Setup for Padding buf --------
        # if use_padding:
        if self.pad_option_2 == 'buf':
            ## padding happening at buffer level
            assert use_padding
            ## assert False, 'bug free, but not used for now'
            ## no bug ['last', 'first_last'], but only need to pad 'first'
            assert dataset_config['pad_type'] in ['first']
            ## *** Now ExtraPad is the len pad to each path, because every path is in len 200/400
            replBuf_config=dict(use_padding=True, tgt_hzn=def_pathlen, **dataset_config)
            assert 'ogb' in self.dset_type
            def_pathlen = def_pathlen + dataset_config['extra_pad']

            fields = OgB_ReplayBuffer_V2(max_n_episodes, max_path_length, termination_penalty, replBuf_config=replBuf_config)
        elif self.pad_option_2 == 'get_item':
            ## ------- Params for Padding get_item --------
            ## Padding when sampling from the dataset, e.g., dset[0]
            assert 'extra_pad' not in dataset_config and use_padding
            ## e.g., [0, 32], randomly sample from 0 to 32 to pad to the start
            self.getit_pad_len_range = dataset_config['getit_pad_len_range']
            self.getit_pad_prob = dataset_config['getit_pad_prob']
            
            ## important: for get_item pad
            self.Avail_Pad_Methods = ['first', 'last', 'first_last']
            self.Avail_Rm_Sides = ['front', 'tail', 'front_tail']

            ## a list
            self.getit_rm_side_prob = dataset_config['getit_rm_side_prob']
            assert np.isclose(sum(self.getit_rm_side_prob), 1.0)
            self.getit_pad_method_prob = dataset_config['getit_pad_method_prob']
            assert np.isclose(sum(self.getit_pad_method_prob), 1.0)
            
            # pdb.set_trace() ## Dec 29, check back after Dinner

            ## old impl
            # self.getit_rm_side = dataset_config['getit_rm_side']
            # self.pad_method = dataset_config['pad_method']

            replBuf_config=dict(use_padding=False)
            fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty, replBuf_config=replBuf_config)
            
            ## --------------------------------------------
        else:
            assert dataset_config.get('pad_type', None) == None
            replBuf_config=dict(use_padding=False)
            

            fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty, replBuf_config=replBuf_config)
        
        ## --------------------------------------

        


        for i, episode in enumerate(data_d_list):
            fields.add_path(episode)
        fields.finalize()

        ## fields.observations.shape ## (n_paths, max_len, dim)
        # pdb.set_trace() ## obs is not normalized now

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])


        ## ------- Trim the End Part of the Stitch Dataset -------
        self.use_path_trim = dataset_config.get('use_path_trim', False)
        self.dset_size_scale = dataset_config.get('dset_size_scale', 1)
        if self.use_path_trim:
            # assert not use_padding, 'exclusive'
            assert self.pad_option_2 != 'buf'
            ## always just keep the front of a traj
            self.path_keep_len = dataset_config['path_keep_len'] ## e.g., 120
            ## if not using pre-defined normalization values,
            ## the normalizer value will change if we directly cut here.
            ## maybe we play tricks somewhere else!
            ## checked normalizer not affected
            tmp_p_len = np.full_like(fields['path_lengths'], fill_value=self.path_keep_len)
            fields['path_lengths'] = tmp_p_len

            fields = self.trimmed_pad_update_fields(fields)

            def_pathlen = fields['path_lengths'][0]
            utils.print_color(f'[Dataset] {def_pathlen=}', c='c')

            assert self.max_ori_path_len is None, 'check'
            # pdb.set_trace()

        
        ## -------------------------------------------------------



        self.indices = self.make_indices(fields.path_lengths, horizon)
        self.len_indices = len(self.indices)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        norm_const_dict = dataset_config.get('norm_const_dict', None)


        if 'navigate' not in env.name:
            # pdb.set_trace() ## check normalization effect
            assert (np.array(self.path_lengths) == def_pathlen).all(), 'ogbench default length'


        ## check if the norm_const_dict is correct
        if norm_const_dict and 'luotest' not in str(dset_h5path):
            for k_name in ['actions', 'observations']:
                utils.print_color(f'{k_name=}')
                print(self.normalizer.normalizers[k_name].mins)
                print(self.normalizer.normalizers[k_name].maxs)
                # pdb.set_trace()
                assert np.isclose(norm_const_dict[k_name][0], 
                                  self.normalizer.normalizers[k_name].mins, atol=1e-5).all()
                assert np.isclose(norm_const_dict[k_name][1], 
                                  self.normalizer.normalizers[k_name].maxs, atol=1e-5).all()
        
        print(fields)
        utils.print_color(f'Dataset Len: {len(self.indices)}', c='y')
        utils.print_color(f'Dataset Len After Scale: {self.__len__()}', c='y')
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')
        # pdb.set_trace() ## see normalizer

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            # pdb.set_trace()
            if self.max_ori_path_len is not None:
                assert path_length > 500, 'sanity check, can be removed'
                assert 'antsoccer-medium' in self.env.name, 'sanity check, can be removed'
                ## temporary solution
                path_length = min(path_length, self.max_ori_path_len + self.dataset_config['extra_pad'])
            # pdb.set_trace()


            max_start = min(path_length - 1, self.max_path_length - horizon)
            ## NOTE: this will automatically pad 0, which is bad
            # if not self.use_padding:
                # max_start = min(max_start, path_length - horizon)
            
            ## e.g., min(199, 200-160=40), 40 is included
            max_start = min(max_start, path_length - horizon)
            # for start in range(max_start):
            for start in range(max_start+1):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        ## important: by default: goal-conditioned
        return {
                0: observations[0],
                self.horizon - 1: observations[-1],
            }

    def __len__(self):
        # return len(self.indices) ## ori
        return len(self.indices) * self.dset_size_scale ## default 1

    def __getitem__(self, idx, eps=1e-4):
        
        if True:
            idx = idx % self.len_indices

        path_ind, start, end = self.indices[idx]

        ## e.g., (384, 2), normed to [-1,1]?
        obs_trajs = self.fields.normed_observations[path_ind, start:end]
        act_trajs = self.fields.normed_actions[path_ind, start:end]

        
        if self.pad_option_2 == 'get_item':
            # pdb.set_trace() ## obs_trajs[-10:]
            tmp_v = random.uniform(0,1)
            if tmp_v < self.getit_pad_prob:
                ## both sides in the range are included
                pad_len = random.randint(self.getit_pad_len_range[0], self.getit_pad_len_range[1])
                
                # Rm ['front', 'tail', 'front_tail']
                # Pad ['first', 'last', 'first_last']

                tmp_rm_side = random.choices(self.Avail_Rm_Sides, 
                            weights=self.getit_rm_side_prob, k=1)[0]
                
                ## First, shorten the traj
                obs_trajs = self.getit_rm_before_pad(obs_trajs, pad_len, tmp_rm_side)
                act_trajs = self.getit_rm_before_pad(act_trajs, pad_len, tmp_rm_side)
                

                tmp_pad_meth = random.choices(self.Avail_Pad_Methods, 
                            weights=self.getit_pad_method_prob, k=1)[0]
                
                # pdb.set_trace()
                # print(f'{tmp_rm_side=}, {tmp_pad_meth=}')
                ## Second, do padding
                obs_trajs = self.getit_do_pad(obs_trajs, pad_len, 'near', tmp_pad_meth)
                act_trajs = self.getit_do_pad(act_trajs, pad_len, 'zero', tmp_pad_meth)

        


        conditions = self.get_conditions(obs_trajs)
        # order: 1.action, 2.obs
        # trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch_v1(obs_trajs, act_trajs, conditions)
        return batch

    def getit_rm_before_pad(self, traj, rm_len: int, rm_side):
        """
        remove from two (one) ends before doing padding
        Args:
            traj: can be obs_traj, or act_traj
        Returns:
            traj: the shortened traj
        """
        assert self.pad_option_2 == 'get_item'

        ## traj: H,D
        ## Implement the other Padding method, which is likely better!
        full_len = len(traj)
        if rm_side == 'tail': ## prev: self.getit_rm_side 
            traj_sh = traj[: full_len - rm_len]
        elif rm_side == 'front':
            traj_sh = traj[rm_len:]
        elif rm_side == 'front_tail': ## maybe use this?
            tmp_len_1 = rm_len // 2 ## 2
            tmp_len_2 = rm_len - tmp_len_1 ## 10
            ## e.g., if full_len=100, then [2:90] -> len=88
            # pdb.set_trace()
            traj_sh = traj[ tmp_len_1 : (full_len - tmp_len_2)  ]
        else:
            raise NotImplementedError

        assert len(traj_sh) == full_len - rm_len

        # pdb.set_trace() ## check len

        return traj_sh


    def getit_do_pad(self, traj, pad_len, pad_v: str, pad_type: str):
        """
        pad_type is a very similar concept as pad_method
        """
        

        if pad_v == 'zero':
            elem_last = np.zeros_like(traj[-1:])
            elem_first = np.zeros_like(traj[0:1])
        elif pad_v == 'near':
            elem_last = traj[-1:]
            elem_first = traj[0:1]
        

        if pad_type == 'last':
            traj_pd = np.concatenate( [traj,] + [elem_last]*pad_len , axis=0 )
        elif pad_type == 'first_last':
            tmp_len_1 = math.ceil(pad_len / 2)
            tmp_len_2 = pad_len - tmp_len_1
            traj_pd = np.concatenate( [elem_first]*tmp_len_1 + [traj,] + [elem_last]*tmp_len_2 , axis=0 )
        elif pad_type == 'first':
            traj_pd = np.concatenate( [elem_first]*pad_len + [traj,] , axis=0 )
        else:
            assert False
        
        # pdb.set_trace()


        return traj_pd






    def trimmed_pad_update_fields(self, fields):
        '''
        Dec 29, maybe just not used
        '''
        if not self.dataset_config.get('do_trimmed_pad', False):
            return fields
        ## ---- Temporary ----
        obs_all = fields.observations[:, :self.path_keep_len, :]
        
        plen_fr = self.dataset_config['trim_pad_first']
        plen_end = self.dataset_config['trim_pad_end']
        ## pad obs
        obs_all = np.concatenate([obs_all[:, 0:1, :],] * plen_fr +
                                 [obs_all[:, :, :],] + 
                                 [obs_all[:, -1:, :],] * plen_end, axis=1)
        # pdb.set_trace() ## check shape
        
        act_all = fields.actions[:, :self.path_keep_len, :]
        ## pad act
        act_all = np.concatenate([np.zeros_like(act_all[:, 0:1, :]),] * plen_fr +
                                 [act_all[:, :, :],] + 
                                 [np.zeros_like(act_all[:, -1:, :]),] * plen_end, axis=1)
        
        # pdb.set_trace() ## check shape


        fields.observations = obs_all
        fields.actions = act_all

        ## FIXME: check path_len is sync to other func in class
        cur_len = fields.observations.shape[1]
        assert cur_len == act_all.shape[1] and cur_len < self.max_path_length
        fields.path_lengths[:] = cur_len
        assert (fields.path_lengths == self.path_keep_len + plen_fr + plen_end).all()
        
        # pdb.set_trace()

        ## ---- Temporary ----

        return fields