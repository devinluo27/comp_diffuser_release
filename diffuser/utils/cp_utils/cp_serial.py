import os, h5py, pdb
from diffuser.utils.serialization import load_config, \
    RandomNumberDataset, get_latest_epoch, DiffusionExperiment
import numpy as np
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer
from diffuser.datasets import *
import diffuser.utils as utils


def load_stgl_sml_diffusion(*loadpath, epoch='latest', device='cuda:0', ld_config={}):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    
    dfu_model_loadpath = ld_config.get("dfu_model_loadpath", None)
    if dfu_model_loadpath:
        ### use another model, e.g., train on small trajs
        model_config = load_config(dfu_model_loadpath, 'model_config.pkl')
        dfu_model_config = load_config(dfu_model_loadpath, 'dfu_model.pkl')
    else:
        model_config = load_config(*loadpath, 'model_config.pkl')
        dfu_model_config = load_config(*loadpath, 'dfu_model.pkl')

    # comp_diffusion_config = load_config(*loadpath, 'comp_diffusion.pkl')
    sml_trainer_config = load_config(*loadpath, 'small_trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    sml_trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    if ld_config.get('use_rand_dset', True):
        dataset = RandomNumberDataset(size=1)
    else:
        dataset = dataset_config()
   
    renderer = render_config()
    model = model_config()
    dfu_model = dfu_model_config(model=model)

    # comp_diffusion = comp_diffusion_config(dfu_model=dfu_model)

    # from diffuser.models.comp_dfu.comp_training_v1 import Comp_Trainer_v1
    # trainer: Comp_Trainer_v1
    trainer = sml_trainer_config(diffusion_model=dfu_model, 
                              dataset=dataset, 
                              renderer=renderer,
                              )


    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    if dfu_model_loadpath:
        # pdb.set_trace()
        epoch = get_latest_epoch(  dfu_model_loadpath.split('/')  )
        trainer.load_from_small(epoch, sm_logdir=dfu_model_loadpath)
        from diffuser.utils import print_color
        print_color(f'\n\n[Load from Another] {dfu_model_loadpath}\n\n',)

    else:
        trainer.load(epoch)
    
    ## dataset renderer model dfu_model comp_diffusion ema trainer epoch'
    utils.freeze_model(model)
    utils.freeze_model(dfu_model)
    utils.freeze_model(trainer.ema_model)


    return DiffusionExperiment(dataset, renderer, model, dfu_model, trainer.ema_model, trainer, epoch)


def get_stgl_lh_ev_probs_fname(env_name):
    """
    Our Maze: load the evaluation problems out
    """
    hdf5_path = None
    # pdb.set_trace()
    if env_name == 'maze2d-large-v1':
        hdf5_path = "data/m2d/ev_probs/maze2d_lg_ev_prob_bt2way_nppp3_rprange02.hdf5"
    elif env_name == 'PointMaze_Large-v3': ## Ben
        ## Oct 29
        hdf5_path = "data/m2d/ev_probs/ben/ben_maze2d_lg_ev_prob_numEp10_eSdSt0.hdf5"
    elif env_name == 'PointMaze_Medium-v3':
        hdf5_path = "data/m2d/ev_probs/ben/ben_maze2d_Me_ev_prob_numEp10_eSdSt0.hdf5"
    elif env_name == 'PointMaze_UMaze-v3':
        hdf5_path = "data/m2d/ev_probs/ben/ben_maze2d_Umz_ev_prob_numEp10_eSdSt0.hdf5"
    else:
        raise NotImplementedError

    return hdf5_path


def load_stgl_lh_ev_probs_hdf5(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in dataset_file.keys():
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
        from diffuser.utils import print_color
        print_color(f'[load ev probs] {data_dict.keys()=}')
    
    return data_dict



def load_stgl_sml_datasetNormalizer(args_train,):
    '''
    Directly create a normalizer with given value, so no need to load the dataset, which is slow.
    Returns:
        normalizer: a class
    '''
    assert type(args_train.normalizer) == str
    # is_kuka = 'kuka' in train_env_list.name # hasattr(train_env_list, 'robot_env')
    normalizer = eval(args_train.normalizer)
    
    # ------------------- load from abc, can be extracted -------------------
    data_dict = {}
    dset_type = args_train.dataset_config.get('dset_type', 'ours')


    if args_train.dataset == "maze2d-large-v1":
        if dset_type == 'ours':
            data_dict['actions'] = np.array([MAZE_Large_Act_Min, MAZE_Large_Act_Max], dtype=np.float32)
            data_dict['observations'] = np.array([MAZE_Large_Obs_Min, MAZE_Large_Obs_Max], dtype=np.float32)
        elif dset_type == 'bens_pm_large':
            data_dict['actions'] = np.array([Ben_maze_large_Act_Min, Ben_maze_large_Act_Max], dtype=np.float32)
            data_dict['observations'] = np.array([Ben_maze_large_Obs_Min, Ben_maze_large_Obs_Max], dtype=np.float32)


        else:
            raise NotImplementedError
    elif args_train.dataset == "maze2d-medium-v1":

        if dset_type == 'bens_pm_medium':
            data_dict['actions'] = np.array([Ben_maze_large_Act_Min, 
                                             Ben_maze_large_Act_Max], dtype=np.float32)
            data_dict['observations'] = np.array([Ben_maze_Medium_Obs_Min, 
                                                  Ben_maze_Medium_Obs_Max], dtype=np.float32)
        else:
            raise NotImplementedError


    elif args_train.dataset == "maze2d-umaze-v1":
        if dset_type == 'bens_pm_umaze':
            data_dict['actions'] = np.array([Ben_maze_large_Act_Min, 
                                             Ben_maze_large_Act_Max], dtype=np.float32)
            data_dict['observations'] = np.array([Ben_maze_UMaze_Obs_Min, 
                                                  Ben_maze_UMaze_Obs_Max], dtype=np.float32)
        else:
            raise NotImplementedError



    ## TODO: Add normalizer for other mazes
    else:
        assert False, 'to be implemented'

    norm_const_dict = args_train.dataset_config.get('norm_const_dict', False)
    if norm_const_dict:
        assert np.isclose(data_dict['actions'], np.array(norm_const_dict['actions'], dtype=np.float32)).all()
        data_dict['observations'] = np.array(norm_const_dict['observations'], dtype=np.float32)
        pdb.set_trace() ## should be good, just stop for checking
    # -------------------
    else:
        print('args_train: no norm_const_dict.')

    d_norm = DatasetNormalizer(data_dict, normalizer, eval_solo=True, path_lengths=None)

    # pdb.set_trace()

    return d_norm

    
        


##### ---------- cd sml version ------------ 

def load_cd_sml_diffusion(*loadpath, epoch='latest', device='cuda:0', ld_config={}):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    
    dfu_model_loadpath = ld_config.get("dfu_model_loadpath", None)
    if dfu_model_loadpath:
        ### use another model, e.g., train on small trajs
        model_config = load_config(dfu_model_loadpath, 'model_config.pkl')
        dfu_model_config = load_config(dfu_model_loadpath, 'dfu_model.pkl')
    else:
        model_config = load_config(*loadpath, 'model_config.pkl')
        dfu_model_config = load_config(*loadpath, 'dfu_model.pkl')

    # comp_diffusion_config = load_config(*loadpath, 'comp_diffusion.pkl')
    sml_trainer_config = load_config(*loadpath, 'small_trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    sml_trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    # dataset = RandomNumberDataset(size=1)
    renderer = render_config()
    model = model_config()
    dfu_model = dfu_model_config(model=model)

    # comp_diffusion = comp_diffusion_config(dfu_model=dfu_model)

    # from diffuser.models.comp_dfu.comp_training_v1 import Comp_Trainer_v1
    # trainer: Comp_Trainer_v1
    trainer = sml_trainer_config(diffusion_model=dfu_model, 
                              dataset=dataset, 
                              renderer=renderer,
                              )


    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    if dfu_model_loadpath:
        # pdb.set_trace()
        epoch = get_latest_epoch(  dfu_model_loadpath.split('/')  )
        trainer.load_from_small(epoch, sm_logdir=dfu_model_loadpath)
        from diffuser.utils import print_color
        print_color(f'\n\n[Load from Another] {dfu_model_loadpath}\n\n',)

    else:
        trainer.load(epoch)
    
    ## dataset renderer model dfu_model comp_diffusion ema trainer epoch'

    return DiffusionExperiment(dataset, renderer, model, dfu_model, trainer.ema_model, trainer, epoch)

