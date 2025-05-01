import collections, pdb
import numpy as np
from diffuser.datasets.d4rl import get_dataset

def comp_sequence_dataset(env, preprocess_fn, dataset_config):
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
    obs_select_dim = list(dataset_config['obs_select_dim']) # must be list
    ## a dict: (['actions', 'infos/goal', 'infos/qpos', 
    ## 'infos/qvel', 'observations', 'rewards', 'terminals', 'timeouts'])
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    # pdb.set_trace() ## True

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        
        # pdb.set_trace() ## check key name correct
        ## important: add each field in dataset to data_
        for k in dataset:
            if 'metadata' in k: continue
            if k == 'observations':
                ## obs_select_dim must be list
                data_[k].append( dataset[k][i][ obs_select_dim ] )
            else:
                data_[k].append(dataset[k][i])

        ## if True, cut and create on episode here
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            
            ## in ori, but feels no need
            # if 'maze2d' in env.name:
                # episode_data = process_maze2d_episode(episode_data)
            
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
