import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')
Comp_DfuExp = namedtuple('Comp_DfuExp', 'dataset renderer model dfu_model comp_diffusion ema trainer epoch')


def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_diffusion(*loadpath, epoch='latest', device='cuda:0', ld_config={}):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    if ld_config.get('use_rand_dset', False):
        dataset = RandomNumberDataset(size=1)
    else:
        dataset = dataset_config()
    
    if ld_config.get('use_rd_v2', False):
        from diffuser.guides.render_m2d import Maze2dRenderer_V2
        render_config._class = Maze2dRenderer_V2
    # pdb.set_trace()

    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)

def load_comp_diffusion(*loadpath, epoch='latest', device='cuda:0', ld_config={}):
    # dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    
    dfu_model_loadpath = ld_config.get("dfu_model_loadpath", None)
    if dfu_model_loadpath:
        from diffuser.utils import print_color
        print_color(f'[Load from Outside] {dfu_model_loadpath}', c='y')
        ### use another model, e.g., train on small trajs
        model_config = load_config(dfu_model_loadpath, 'model_config.pkl')
        dfu_model_config = load_config(dfu_model_loadpath, 'dfu_model.pkl')
    else:
        model_config = load_config(*loadpath, 'model_config.pkl')
        dfu_model_config = load_config(*loadpath, 'dfu_model.pkl')

    comp_diffusion_config = load_config(*loadpath, 'comp_diffusion.pkl')
    comp_trainer_config = load_config(*loadpath, 'comp_trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    comp_trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    # dataset = dataset_config()
    dataset = RandomNumberDataset(size=1)
    renderer = render_config()
    model = model_config()
    dfu_model = dfu_model_config(model=model)

    comp_diffusion = comp_diffusion_config(dfu_model=dfu_model)

    from diffuser.models.comp_dfu.comp_training_v1 import Comp_Trainer_v1
    trainer: Comp_Trainer_v1
    trainer = comp_trainer_config(diffusion_model=comp_diffusion, 
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

    return Comp_DfuExp(dataset, renderer, model, dfu_model, comp_diffusion, trainer.ema_model, trainer, epoch)


import numpy as np
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer
def load_comp_datasetNormalizer(args_train, ):
    ''' h5path: path to train dataset
    normalizer: a class
    '''
    assert type(args_train.normalizer) == str
    # is_kuka = 'kuka' in train_env_list.name # hasattr(train_env_list, 'robot_env')
    normalizer = eval(args_train.normalizer)
    
    # ------------------- load from abc, can be extracted -------------------
    norm_const_dict = args_train.dataset_config.get('norm_const_dict', False)
    if norm_const_dict:
        data_dict = {}
        # data_dict['actions'] = np.array([[-1,-1], [1,1]], dtype=np.float32)
        data_dict['actions'] = np.array(norm_const_dict['actions'], dtype=np.float32)
        data_dict['observations'] = np.array(norm_const_dict['observations'], dtype=np.float32)
        # pdb.set_trace()
        d_norm = DatasetNormalizer(data_dict, normalizer, eval_solo=True, path_lengths=None)
        return d_norm
    # -------------------
    else:
        assert False

class RandomNumberDataset(torch.utils.data.Dataset):
    '''
    a placeholder dataset
    '''
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]