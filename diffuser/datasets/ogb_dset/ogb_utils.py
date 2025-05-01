import diffuser.utils as utils
import numpy as np
import collections, pdb

## ---------- Dec 21, For OGBench ---------------
def ogb_load_env(e_name_og,): ## env_only
    import ogbench
    utils.print_color(f'[ogb_load_env] {e_name_og=}')
    if type(e_name_og) != str:
        ## indicate that it is already an env
        assert 'ogbench.locomaze.' in str(type(e_name_og))
        return e_name_og
    ## compact_dataset=False,
    wrapped_env = ogbench.make_env_and_datasets(e_name_og, env_only=True)

    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = e_name_og
    return env

## ---------- Jan 20, For paper vis, change resolution ------------
def ogb_load_env_kwargs(e_name_og, **kwargs): ## env_only
    import ogbench
    utils.print_color(f'[ogb_load_env] {e_name_og=}')
    if type(e_name_og) != str:
        ## indicate that it is already an env
        assert 'ogbench.locomaze.' in str(type(e_name_og))
        return e_name_og
    ## compact_dataset=False,
    wrapped_env = ogbench.make_env_and_datasets(e_name_og, env_only=True, **kwargs)

    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = e_name_og
    return env


def ogb_get_dataset(env,):
    '''env: an ogbench env'''
    from ogbench.utils import ogb_load_dataset
    ## just return the trainset
    return ogb_load_dataset(env)




def ogb_seg_dset_trajs(traj_cat: np.ndarray, terminals: np.ndarray, verbose=False):
    """
    For OGBench Dataset Preprocessing,
    Segment the very long 1M transitions dataset into pieces according to terminals
    traj_cat: [h,dim], a very long traj that is discontinuous
    terminals: [h,], float of 0 or 1, included
    """
    assert len(traj_cat) == len(terminals)
    assert traj_cat.ndim == 2 and traj_cat.ndim == terminals.ndim + 1
    
    ## [199 399] + 1 --> [200, 400, 600,...]
    term_idxs = np.where(terminals==1,)[0] + 1
    utils.print_color(f'{term_idxs[:5]=}, {term_idxs[-5:]=}')
    sub_trajs = np.split(traj_cat, term_idxs, axis=0)
    ## remove the last empty one, there is more efficient way.
    out_trajs = []
    for i_t, s_tj in enumerate(sub_trajs):
        ## the last one might be in shape [0, 29]
        if len(s_tj) > 0:
            out_trajs.append(s_tj)
            ## sanity check in real production, the len should be the same
            if len(sub_trajs) > 100:
                assert len(s_tj) == len(sub_trajs[0])
        
        if verbose:
            print(f'{i_t} {s_tj.shape=}')
        


    out_trajs = np.array(out_trajs)

    print(f'[ogb_seg_dset_trajs] {len(out_trajs)=}, {out_trajs.shape=}')

    return out_trajs # sub_trajs



def ogb_xy_to_ij(env, xy_trajs):
    """
    For OGBench, translate the OGBench coordinate into our 2D rendering coordinate,
    so we can render without using mujoco renderer. 
    xy_trajs: [B,H,2] or [H,2]
    """
    # _offset_x = 4
    # _offset_y  = 4
    assert xy_trajs.ndim in [2, 3] and xy_trajs.shape[-1] == 2
    maze_unit = env.get_maze_unit() # 4 # self._maze_unit
    i = (xy_trajs[..., 1:2] + env.get_offset_y() + 0.5 * maze_unit ) / maze_unit
    j = (xy_trajs[..., 0:1] + env.get_offset_x() + 0.5 * maze_unit ) / maze_unit
    
    out = np.concatenate([i, j], axis=-1) - 0.5
    
    return out



## Dec 21 From Here
def ogb_sequence_dataset(env, preprocess_fn, dataset_config):
    """
    Copy init from sequence_dataset
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    from diffuser.datasets.ogb_dset.ogb_utils import ogb_get_dataset, ogb_seg_dset_trajs

    obs_select_dim = list(dataset_config['obs_select_dim']) # must be list
    ## ogb a dict: ['observations', 'actions', 'terminals']
    dataset = ogb_get_dataset(env)

    # pdb.set_trace() ## Dec 21, check dataset and obs_select_dim

    ## TODO: maybe just do nothing
    dataset = preprocess_fn(dataset)

    N = dataset['observations'].shape[0]
    utils.print_color(f'dataset size {N=}')

    obs_all = dataset['observations']
    act_all = dataset['actions']
    term_all = dataset['terminals']
    obs_seg_trajs = ogb_seg_dset_trajs(traj_cat=obs_all, terminals=term_all)
    act_seg_trajs = ogb_seg_dset_trajs(act_all, term_all)
    ## sanity check because in ogbench all traj_len==200
    assert obs_seg_trajs[-1].shape == obs_seg_trajs[0].shape, f'{obs_seg_trajs[-1].shape=}' 
    
    n_trajs = len(obs_seg_trajs)
    out_list = [] ## a list of dict
    for i_tj in range(n_trajs):
        assert obs_seg_trajs[i_tj].ndim == 2, 'H, Dim'
        data_ = collections.defaultdict(list)
        ## already arrays
        data_['observations'] = obs_seg_trajs[i_tj][:, obs_select_dim]
        data_['actions'] = act_seg_trajs[i_tj]
        out_list.append(data_)
    
    # pdb.set_trace() ## check data_

    return out_list
    



    

    



## -------------------------------------------


from diffuser.datasets.ogb_dset.ogb_utils import ogb_xy_to_ij

def ogb_get_rowcol_obs_trajs_from_xy(env, obs_trajs):
    """
    convert the original ogb xy2d scale (very large due to the ant/humanoid model)
    to the scale in our offline renderer.
    Args:
        env: should be a real ogb env instance
    Returns:
        obs_trajs: np (b,h,2)
    """
    ## to support ogbench
    # from diffuser.datasets.ogb_dset.ogb_utils import ogb_xy_to_ij
    if type(obs_trajs) == list:
        obs_trajs = np.array(obs_trajs)
    assert obs_trajs.ndim in [2, 3]
    # pdb.set_trace()
    obs_trajs = obs_trajs[..., :2] ## should be B,H,D or H,D
    obs_trajs = ogb_xy_to_ij(env, xy_trajs=obs_trajs)
    return obs_trajs


def ogb_get_rowcol_obs_trajs_from_xy_list(env, obs_trajs: list):
    """
    The only difference is that the input is a list of trajs, 
    so this func supports converting traj of different length in a list
    Args:
        env: should be a real ogb env instance
    Returns:
        obs_trajs: np (b,h,2)
    """
    assert type(obs_trajs) == list

    if obs_trajs[0].ndim == 2:
        ## a list of (H,D), but H is different
        out_list = []
        for tmp_tj in obs_trajs:
            tmp_tj = tmp_tj[:, :2]
            tmp_tj = ogb_xy_to_ij(env, xy_trajs=tmp_tj)
            out_list.append(tmp_tj)
        return out_list
    
    else:
        raise NotImplementedError




