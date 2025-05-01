from collections import namedtuple
import numpy as np
import torch, pdb, random

from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.datasets.d4rl import load_environment
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.comp.ben_pad_buffer import Ben_Pad_ReplayBuffer
from diffuser.datasets.comp.comp_dataset_v1 import Batch_v1
from .comp_data_utils import comp_sequence_dataset
import diffuser.utils as utils


class Ben_Pad_SeqDataset_V1(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True,
        dset_h5path=None,
        dataset_config={},
        ):
        """
        use_padding: buffer level padding
        """
        ## maze2d_set_terminals
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.dataset_config = dataset_config
        ## -----------------------
        ## call get_dataset inside
        env.dset_h5path = dset_h5path
        self.obs_select_dim = dataset_config['obs_select_dim'] 

        itr = comp_sequence_dataset(env, self.preprocess_fn, dataset_config)


        if use_padding:
            assert False
            assert dataset_config['pad_type'] in ['last', 'first_last']
            replBuf_config=dict(use_padding=True, tgt_hzn=horizon, **dataset_config)
        else:
            assert dataset_config['pad_type'] in [None, 'pad_sample_v1']
            replBuf_config=dict(use_padding=False)
        

        self.pad_type = dataset_config['pad_type']
        self.padsam_first_prob = dataset_config['padsam_first_prob']
        self.padsam_drift_max = dataset_config['padsam_drift_max']
        self.padsam_drift_lowlimit = dataset_config['padsam_drift_lowlimit']
            

        fields = Ben_Pad_ReplayBuffer(max_n_episodes, max_path_length, termination_penalty, replBuf_config=replBuf_config)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()
        self.fields = fields
        # pdb.set_trace() ## obs probably is not normalized now

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        
        self.indices = self.make_indices_padS_v1(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        norm_const_dict = dataset_config.get('norm_const_dict', None)
        # pdb.set_trace()
        self.dset_type = dataset_config.get('dset_type', 'ours')

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
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices_padS_v1(self, path_lengths, horizon):
        '''
            just the same as diffuser, but sample all
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        self.cnt_pad_item = 0
        self.cnt_no_pad_item = 0
        indices = []
        for i, path_length in enumerate(path_lengths):
            ## e.g., path_length=100, horizon=128
            if path_length < horizon:
                sm_start = 0
                sm_end = path_length
                for _ in range(self.dataset_config['n_smhzn_repeat']): # 10
                    indices.append((i, sm_start, sm_end))
                    self.cnt_pad_item += 1
            else:
                max_start = min(path_length - 1, self.max_path_length - horizon)
                ## NOTE: this will automatically pad 0, which is bad
                # if not self.use_padding:
                    # max_start = min(max_start, path_length - horizon)
                
                max_start = min(max_start, path_length - horizon)
                for start in range(max_start+1): ## check
                    end = start + horizon
                    indices.append((i, start, end))
                    self.cnt_no_pad_item += 1
            # pdb.set_trace()
        indices = np.array(indices)
        utils.print_color(f'{self.cnt_pad_item=}; {self.cnt_no_pad_item=}')
        # pdb.set_trace()
        return indices
    
    def make_indices_v2(self):
        pass

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
        path_ind, start, end = self.indices[idx]

        ## e.g., (384, 2), normed to [-1,1]?
        obs_trajs = self.fields.normed_observations[path_ind, start:end]
        act_trajs = self.fields.normed_actions[path_ind, start:end]

        
        traj_len = len(obs_trajs)
        pad_len = self.horizon - traj_len
        if self.pad_type == 'pad_sample_v1' and pad_len > 0:
            assert obs_trajs.shape[-1] == 2
            tmp_drift = random.randint(0, self.padsam_drift_max)
            if traj_len - tmp_drift > self.padsam_drift_lowlimit: ## 15
                pass
            else:
                tmp_drift = 0
            pad_len = pad_len + tmp_drift

            if random.random() < self.padsam_first_prob: ## 0.5
                obs_trajs = obs_trajs[:  traj_len-tmp_drift ]
                act_trajs = act_trajs[:  traj_len-tmp_drift ]
                ## first
                obs_trajs = np.concatenate( [ obs_trajs[0:1], ] * pad_len + [obs_trajs,] , axis=0 )
                act_trajs = np.concatenate( [ np.zeros_like(act_trajs[0:1]), ] * pad_len + [act_trajs,] , axis=0)
            else:
                obs_trajs = obs_trajs[ tmp_drift: ]
                act_trajs = act_trajs[ tmp_drift: ]

                obs_trajs = np.concatenate( [obs_trajs,] + [ obs_trajs[-1:], ] * pad_len , axis=0 )
                act_trajs = np.concatenate( [act_trajs,] + [ np.zeros_like(act_trajs[-1:]), ] * pad_len , axis=0 )


        # pdb.set_trace()


        conditions = self.get_conditions(obs_trajs)
        # order: 1.action, 2.obs
        # trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch_v1(obs_trajs, act_trajs, conditions)
        return batch

