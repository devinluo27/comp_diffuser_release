import numpy as np
import torch, pdb, sys, os
import torch.nn.functional as F
import diffuser.utils as utils
from colorama import Fore
import einops, imageio, json
from datetime import datetime
import os.path as osp
from contextlib import contextmanager



def print_color(s, *args, c='r'):
    if c == 'r':
        # print(Fore.RED + s + Fore.RESET)
        print(Fore.RED, end='')
        print(s, *args, Fore.RESET)
    elif c == 'b':
        # print(Fore.BLUE + s + Fore.RESET)
        print(Fore.BLUE, end='')
        print(s, *args, Fore.RESET)
    elif c == 'y':
        # print(Fore.YELLOW + s + Fore.RESET)
        print(Fore.YELLOW, end='')
        print(s, *args, Fore.RESET)
    else:
        # print(Fore.CYAN + s + Fore.RESET)
        print(Fore.CYAN, end='')
        print(s, *args, Fore.RESET)
    
    

def get_time():
    return datetime.now().strftime("%y%m%d-%H%M%S")

import os ## TODO: from here
def get_sample_savedir(logdir, i_s, div_freq):
    '''get a subdir under logdir, e.g., {logdir}/0'''
    div_freq = 100000
    subdir = str( (i_s // div_freq) * div_freq )
    sample_savedir = os.path.join(logdir, subdir)
    if not os.path.isdir(sample_savedir):
        os.makedirs(sample_savedir)
    return sample_savedir

def save_img(save_path: str, img: np.ndarray):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    imageio.imsave(save_path, img)
    print(f'[save_img] {save_path}')



def save_json(j_data: dict, full_path):
    with open(full_path, "w") as f:
        json.dump(j_data, f, indent=2)
    print(f'[save_json] {full_path}')

def rename_fn(src_name, new_name):
    os.rename(src=src_name, dst=new_name)
    utils.print_color(f'[rename fn to] {new_name}')

#### =========================
#### ========== Ben ==========


def ben_get_m2d_spec(dset_type):
    if dset_type == 'bens_pm_large':
        maze_x_map_center = 6.0
        maze_y_map_center = 4.5
    elif dset_type == 'bens_pm_medium':
        maze_x_map_center = 4.0
        maze_y_map_center = 4.0
    elif dset_type == 'bens_pm_umaze':
        maze_x_map_center = 2.5
        maze_y_map_center = 2.5
    else:
        assert False
    maze_size_scaling = 1.0

    return maze_x_map_center, maze_y_map_center, maze_size_scaling


def ben_xy_to_luo_rowcol(dset_type: str, trajs: np.ndarray):
    """
    xy_pos: is the position in mujoco, np 1d or 2d (h,2) or 3d (B,h,2)
    """

    maze_x_map_center, maze_y_map_center, maze_size_scaling = \
                        ben_get_m2d_spec(dset_type)
    
    # if trajs.ndim == 1:
        # trajs = trajs[None,None,]
    # elif trajs.ndim == 2:
        # trajs = trajs[None,]
    # pdb.set_trace()
    
    assert trajs.shape[-1] == 2

    rowcol_pos_1 = (trajs[..., 0:1] + maze_x_map_center) / maze_size_scaling - 0.5
    rowcol_pos_0 = (maze_y_map_center - trajs[..., 1:2]) / maze_size_scaling - 0.5


    return np.concatenate([rowcol_pos_0, rowcol_pos_1, ], axis=-1)


def ben_luo_rowcol_to_xy(dset_type: str, trajs: np.ndarray):
    """
    Oct 30
    xy_pos: is the position in mujoco, np 1d or 2d (h,2) or 3d (B,h,2)
    """

    maze_x_map_center, maze_y_map_center, maze_size_scaling = \
                        ben_get_m2d_spec(dset_type)

    assert trajs.shape[-1] == 2
    ## dim?
    # pdb.set_trace()
    # x = (rowcol_pos[1] + 0.5) * self.maze_size_scaling - self.x_map_center
    tjs_x = (trajs[..., 1:2] + 0.5) * maze_size_scaling - maze_x_map_center
    # y = self.y_map_center - (rowcol_pos[0] + 0.5) * self.maze_size_scaling
    tjs_y = maze_y_map_center - (trajs[..., 0:1] + 0.5) * maze_size_scaling

    # pdb.set_trace() ## Check

    return np.concatenate([tjs_x, tjs_y, ], axis=-1)



def seed_env_list(env_list, seed_list):
    ## seed the given list of env
    for i_e in range( len(env_list) ):
        env_list[i_e].seed( seed_list[i_e] )
    return env_list


#### =========================


@contextmanager
def suppress_stdout():
    # Save the current stdout
    old_stdout = sys.stdout
    # Redirect stdout to devnull using 'with open'
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            # Restore stdout to its original state
            sys.stdout = old_stdout

import warnings
@contextmanager
def suppress_warnings(category=Warning):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category)
        yield


def freeze_model(model):
    model.eval()
    # Set requires_grad to False for all parameters
    for param in model.parameters():
        param.requires_grad = False