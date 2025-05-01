import gym
import numpy as np
import einops
from scipy.spatial.transform import Rotation as R
import pdb
import diffuser.utils as utils
from .d4rl import load_environment

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn

def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)

def get_policy_preprocess_fn(fn_names):
    fns = [eval(name) for name in fn_names]
    return compose(*fns)

#-----------------------------------------------------------------------------#
#-------------------------- preprocessing functions --------------------------#
#-----------------------------------------------------------------------------#

#------------------------ @TODO: remove some of these ------------------------#

def arctanh_actions(*args, **kwargs):
    epsilon = 1e-4

    def _fn(dataset):
        actions = dataset['actions']
        assert actions.min() >= -1 and actions.max() <= 1, \
            f'applying arctanh to actions in range [{actions.min()}, {actions.max()}]'
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)
        dataset['actions'] = np.arctanh(actions)
        return dataset

    return _fn

def add_deltas(env):

    def _fn(dataset):
        deltas = dataset['next_observations'] - dataset['observations']
        dataset['deltas'] = deltas
        return dataset

    return _fn


def maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        """looks like just set a terminal signal if at goal
        seems like a stupid way to cut dataset.
        """
        ## dict_keys(['actions', 'infos/goal', 'infos/qpos', 'infos/qvel', 'observations', 'rewards', 'terminals', 'timeouts'])
        ## what is the meaning of timeouts here??
        ## Large: 16726,0
        utils.print_color(f"{dataset['timeouts'].sum()=}, {dataset['terminals'].sum()=}")
        # pdb.set_trace()
        ## check smoothiness: yes, all trajs are closely connected, max: 0.127
        # obs_all = dataset['observations'][:,:2]
        # dist_all = np.linalg.norm(obs_all[1:] - obs_all[:-1], axis=1)

        xy = dataset['observations'][:,:2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        ## checked correct, cut when first arrive goal: [7,9]
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        return dataset

    return _fn




def ben_maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env

    def _fn(dataset):
        """different from the our default settings, basically just copy a timeouts key 
        and do nothing here.
        """
        ## dict_keys(['actions', 'infos/goal', 'infos/qpos', 'infos/qvel', 'observations', 'rewards', 'terminals', 'timeouts'])
        ## what is the meaning of timeouts here??
        dataset['timeouts'] = dataset['terminals'].copy()
        ## 16726,0
        print(f"{dataset['timeouts'].sum()=}, {dataset['terminals'].sum()=}")

        
        ## idx of where an episode teminates
        timeout_steps = np.where(dataset['timeouts'])[0]
        ## for the first ep, both end inclusive;
        ## for others, (st,end], only end idx is inclusive
        ## So, the timeout_steps idx is included in the prev episode
        path_lengths = timeout_steps - np.concatenate([[-1], timeout_steps[:-1]], axis=0)
        ## miss the first one
        # path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        # pdb.set_trace()
        # assert path_lengths.sum() == 

        utils.print_color(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        # dataset['timeouts'] = timeouts
        return dataset

    return _fn



## Oct 17, We segment the consecutive trajs into pieces
def maze2d_set_terminals_seg(env):
    env = load_environment(env) if type(env) == str else env
    len_seg = env.len_seg

    def _fn(dataset):
        """looks like just set a terminal signal if at goal
        seems like a stupid way to cut dataset.
        """
        ## dict_keys(['actions', 'infos/goal', 'infos/qpos', 'infos/qvel', 'observations', 'rewards', 'terminals', 'timeouts'])
        ## Large Maze: 16726,0
        utils.print_color(f"1: {dataset['timeouts'].sum()=}, {dataset['terminals'].sum()=}")

        len_dset = len(dataset['timeouts'])
        ## last one is discard
        stop_idx = np.arange(len_seg-1, len_dset, step=len_seg)
        timeouts = np.zeros_like(dataset['timeouts'], dtype=bool)
        timeouts[stop_idx] = True
        # pdb.set_trace()

        timeout_steps = np.where(timeouts)[0]
        
        assert (timeout_steps == stop_idx).all()

        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        assert (path_lengths == path_lengths[0]).all()

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts

        utils.print_color(f"2: {dataset['timeouts'].sum()=}, {dataset['terminals'].sum()=}")
        # pdb.set_trace()

        return dataset

    return _fn



#-------------------------- block-stacking --------------------------#

def blocks_quat_to_euler(observations):
    '''
        input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
            xyz: 3
            quat: 4
            contact: 1

        returns : [ N x robot_dim + n_blocks * 10] = [ N x 47 ]
            xyz: 3
            sin: 3
            cos: 3
            contact: 1
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        quat = block_info[:, 3:-1]
        contact = block_info[:, -1:]

        euler = R.from_quat(quat).as_euler('xyz')
        sin = np.sin(euler)
        cos = np.cos(euler)

        X = np.concatenate([
            X,
            xpos,
            sin,
            cos,
            contact,
        ], axis=-1)

    return X

def blocks_euler_to_quat_2d(observations):
    robot_dim = 7
    block_dim = 10
    n_blocks = 4

    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        sin = block_info[:, 3:6]
        cos = block_info[:, 6:9]
        contact = block_info[:, 9:]

        euler = np.arctan2(sin, cos)
        quat = R.from_euler('xyz', euler, degrees=False).as_quat()

        X = np.concatenate([
            X,
            xpos,
            quat,
            contact,
        ], axis=-1)

    return X

def blocks_euler_to_quat(paths):
    return np.stack([
        blocks_euler_to_quat_2d(path)
        for path in paths
    ], axis=0)

def blocks_process_cubes(env):

    def _fn(dataset):
        for key in ['observations', 'next_observations']:
            dataset[key] = blocks_quat_to_euler(dataset[key])
        return dataset

    return _fn

def blocks_remove_kuka(env):

    def _fn(dataset):
        for key in ['observations', 'next_observations']:
            dataset[key] = dataset[key][:, 7:]
        return dataset

    return _fn

def blocks_add_kuka(observations):
    '''
        observations : [ batch_size x horizon x 32 ]
    '''
    robot_dim = 7
    batch_size, horizon, _ = observations.shape
    observations = np.concatenate([
        np.zeros((batch_size, horizon, 7)),
        observations,
    ], axis=-1)
    return observations

def blocks_cumsum_quat(deltas):
    '''
        deltas : [ batch_size x horizon x transition_dim ]
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert deltas.shape[-1] == robot_dim + n_blocks * block_dim

    batch_size, horizon, _ = deltas.shape

    cumsum = deltas.cumsum(axis=1)
    for i in range(n_blocks):
        start = robot_dim + i * block_dim + 3
        end = start + 4

        quat = deltas[:, :, start:end].copy()

        quat = einops.rearrange(quat, 'b h q -> (b h) q')
        euler = R.from_quat(quat).as_euler('xyz')
        euler = einops.rearrange(euler, '(b h) e -> b h e', b=batch_size)
        cumsum_euler = euler.cumsum(axis=1)

        cumsum_euler = einops.rearrange(cumsum_euler, 'b h e -> (b h) e')
        cumsum_quat = R.from_euler('xyz', cumsum_euler).as_quat()
        cumsum_quat = einops.rearrange(cumsum_quat, '(b h) q -> b h q', b=batch_size)

        cumsum[:, :, start:end] = cumsum_quat.copy()

    return cumsum

def blocks_delta_quat_helper(observations, next_observations):
    '''
        input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
            xyz: 3
            quat: 4
            contact: 1
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert observations.shape[-1] == next_observations.shape[-1] == robot_dim + n_blocks * block_dim

    deltas = (next_observations - observations)[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]
        next_block_info = next_observations[:, start:end]

        xpos = block_info[:, :3]
        next_xpos = next_block_info[:, :3]

        quat = block_info[:, 3:-1]
        next_quat = next_block_info[:, 3:-1]

        contact = block_info[:, -1:]
        next_contact = next_block_info[:, -1:]

        delta_xpos = next_xpos - xpos
        delta_contact = next_contact - contact

        rot = R.from_quat(quat)
        next_rot = R.from_quat(next_quat)

        delta_quat = (next_rot * rot.inv()).as_quat()
        w = delta_quat[:, -1:]

        ## make w positive to avoid [0, 0, 0, -1]
        delta_quat = delta_quat * np.sign(w)

        ## apply rot then delta to ensure we end at next_rot
        ## delta * rot = next_rot * rot' * rot = next_rot
        next_euler = next_rot.as_euler('xyz')
        next_euler_check = (R.from_quat(delta_quat) * rot).as_euler('xyz')
        assert np.allclose(next_euler, next_euler_check)

        deltas = np.concatenate([
            deltas,
            delta_xpos,
            delta_quat,
            delta_contact,
        ], axis=-1)

    return deltas

def blocks_add_deltas(env):

    def _fn(dataset):
        deltas = blocks_delta_quat_helper(dataset['observations'], dataset['next_observations'])
        # deltas = dataset['next_observations'] - dataset['observations']
        dataset['deltas'] = deltas
        return dataset

    return _fn
