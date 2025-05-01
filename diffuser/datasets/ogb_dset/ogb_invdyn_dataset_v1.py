from collections import namedtuple
import numpy as np
import torch, pdb

from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.buffer import ReplayBuffer
import diffuser.utils as utils
from diffuser.datasets.ogb_dset.ogb_utils import ogb_load_env, ogb_sequence_dataset

Batch_og_inv_v1 = namedtuple('Batch_og_inv_v1', 'obs_trajs act_trajs conditions val_lens')

## Add Padding and a valid length variable
class OgB_InvDyn_SeqDataset_V1(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True,
        dset_h5path=None,
        dataset_config={},
        ):
        ## maze2d_set_terminals
        self.env = env = ogb_load_env(env)
        env.len_seg = dataset_config.get('len_seg', None)
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        # self.env = env = ogb_load_env(env) ## no need
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        ## -----------------------
        # pdb.set_trace() ## check env is correct
        assert preprocess_fns == [], 'no need'

        ## call get_dataset inside
        env.dset_h5path = dset_h5path
        self.obs_select_dim = dataset_config['obs_select_dim'] 

        self.dset_type = dataset_config['dset_type'] # 'ours')

        ## a list of dict, e.g., len is 5000==1M/200
        data_d_list = ogb_sequence_dataset(env, self.preprocess_fn, dataset_config)



        if True:
            ## just let it pad zero, because we do not use padding for training
            assert dataset_config.get('pad_type', None) == None
            replBuf_config=dict(use_padding=False)

        self.dataset_config = dataset_config
        self.max_ori_path_len = dataset_config.get('max_ori_path_len', None)
            

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty, replBuf_config=replBuf_config)
        for i, episode in enumerate(data_d_list):
            fields.add_path(episode)
        fields.finalize()

        # pdb.set_trace() ## Oct 14, obs is not normalized now

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        norm_const_dict = dataset_config.get('norm_const_dict', None)
        
        # pdb.set_trace() ## check normalization effect
        def_pathlen = 400 if 'human' in env.name.lower() else 200
        def_pathlen = 500 if 'explore' in env.name else def_pathlen
        def_pathlen = 500 if 'antsoccer-medium' in env.name else def_pathlen

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
        utils.print_color(f'[OgB_InvDyn_SeqDataset_V1]' 
                          f'Dataset Len: {len(self.indices)}', c='y')
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

            if self.max_ori_path_len is not None:
                assert path_length >= 500, 'sanity check, can be removed'
                assert 'antsoccer-medium' in self.env.name, 'sanity check, can be removed'
                ## temporary solution
                path_length = min(path_length, self.max_ori_path_len + self.dataset_config['extra_pad'])
                assert self.dataset_config['extra_pad'] == 0



            ## e.g., path_length=200, max_start=199
            max_start = min(path_length - 1, self.max_path_length - horizon)
            assert max_start == path_length - 1, 'make sure all transition is included'

            ## NOTE: this will automatically pad 0, which is bad
            if not self.use_padding:
                assert False, 'seems like we do not use it, because we want all transition to be included, and we have a val_len flag'
                ## e.g., min(199, 200-160=40)
                max_start = min(max_start, path_length - horizon)
            else:
                ## e.g., 198
                max_start = max_start - 1


            ### ----- Debug -------

            
            ## e.g., [0,1,2,...,198], inclusive
            # pdb.set_trace()

            for start in range(max_start+1):
                ## start=198, end=202 -> [198,199,200,201]
                end = start + horizon
                ## the first n is valid (not padding)
                val_len = min(path_length - start, horizon)
                indices.append( (i, start, end, val_len) )
        
        # pdb.set_trace()
            
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
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end, val_len = self.indices[idx]

        ## e.g., (4->10, 2), normed to [-1,1]
        obs_trajs = self.fields.normed_observations[path_ind, start:end]
        act_trajs = self.fields.normed_actions[path_ind, start:end]

        ## important: goal cond might be the padding, so close for now
        ## conditions = self.get_conditions(obs_trajs) # ori
        conditions = np.zeros(shape=[1,], dtype=np.int32) # None
        
        # order: 1.action, 2.obs
        # pdb.set_trace()

        batch = Batch_og_inv_v1(obs_trajs, act_trajs, conditions, val_len)

        return batch

