import numpy as np
import pdb

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty,
                 replBuf_config={}):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int32),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty
        self.use_padding = replBuf_config['use_padding']
        self.replBuf_config = replBuf_config
        self.pad_type = self.replBuf_config.get('pad_type', None)

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def get_pad_len(self, path: dict):
        hzn = self.replBuf_config['tgt_hzn']
        len_path = len(path['observations'])
        assert path['observations'].shape[1] == 2, 'only work for maze2d xy'
        pad_len = hzn - len_path + self.replBuf_config['extra_pad'] # 1
        assert self.replBuf_config['extra_pad'] >= 1
        
        return pad_len
    
    def pad_path_with_last(self, path: dict):
        ## TODO: check back
        pad_len = self.get_pad_len(path) ## total pad_len
        if pad_len <= 0:
            return path ## no need to pad

        # pdb.set_trace()
        new_path = {}
        for key in path.keys():
            elem = path[key]
            ## FIXME: temporary solution, placeholder
            if key in ['actions', 'infos/qvel']:
                elem_last = np.zeros_like(elem[-1:])
            else:
                elem_last = elem[-1:]
            
            if key in ['actions', 'infos/qvel']:
                elem_first = np.zeros_like(elem[0:1])
            else:
                elem_first = elem[0:1]
            

            # pdb.set_trace()
            if self.pad_type == 'last':
                new_path[key] = np.concatenate( [elem,] + [elem_last]*pad_len , axis=0 )
            elif self.pad_type == 'first_last':
                # tmp_len = (pad_len // 2) + 1
                tmp_len = round(pad_len / 2)
                new_path[key] = np.concatenate( [elem_first]*tmp_len + [elem,] + [elem_last]*tmp_len , axis=0 )
            else:
                assert False

        # pdb.set_trace()
        return new_path


    def add_path(self, path):
        if self.pad_type in ['last', 'first_last'] and self.use_padding:
            path = self.pad_path_with_last(path)
            # pdb.set_trace()
        else:
            assert self.pad_type is None
        path_length = len(path['observations'])

        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## in ogbench, just ['observations', 'actions']
        # pdb.set_trace() ## Dec 22 path.keys()?, can switch if order below
        ## penalize early termination
        # if path['terminals'].any() and self.termination_penalty is not None: ## ori
        if (self.termination_penalty is not None) and path['terminals'].any():
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')
