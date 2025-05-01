import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings, pdb

from diffuser.datasets.d4rl import load_environment
from diffuser.datasets.d4rl import Is_Gym_Robot_Env, load_env_gym_robo, Is_OgB_Robot_Env
from .render_utils import make_traj_images
import diffuser.utils as utils

class Maze2dRenderer_V2:
    '''Luo Implement Version, July 22'''

    def __init__(self, env, observation_dim=None):
        ## FIXME: multiple env creation...
        ## Oct 29
        if Is_Gym_Robot_Env:
            self.env = load_env_gym_robo(env)
        elif Is_OgB_Robot_Env:
            # pdb.set_trace() ## TODO:
            from diffuser.datasets.ogb_dset.ogb_utils import ogb_load_env
            self.env = ogb_load_env(env) ## can be a str or already an env
            # assert type(env) != str
            # self.env = env

        else:
            self.env = load_environment(env)
        self.env_name = self.env.name
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        # self.goal = None
        # self._background = self.env.maze_arr == 10
        # self._remove_margins = False
        # self._extent = (0, 1, 1, 0)

    def renders(self, obs_traj, conditions=None, **kwargs):
        '''
        renders one traj: H,dim(2)
        '''
        assert obs_traj.ndim == 2
        ## A list of np (h,w,4)
        imgs = make_traj_images(self.env, obs_traj[None,], **kwargs )
        img = imgs[0]
        return img

        

    def composite(self, savepath, obs_trajs, ncol=5, pad_len=0, 
                return_rows=False, **kwargs):
        '''
            savepath : str
            obs_trajs : [ n_paths x horizon x 2 ]
            pad_len: pad around each sub img
            return_rows: if True, return imgs splited by row, shape: (n_rows, H, W*ncol, C)
        '''
        n_res = len(obs_trajs) % ncol
        if n_res != 0: # and len(obs_trajs) > ncol:
            ncol = len(obs_trajs)
            # tmp_pad_trajs = np.zeros_like(obs_trajs[:ncol-n_res])
            # obs_trajs = np.concatenate([obs_trajs, tmp_pad_trajs], axis=0)
        
        ## --------- New Jan 4 ------------
        if type(obs_trajs) == list:
            ## when the input is a list with diff hzn
            if len(obs_trajs) >= 2: ## and obs_trajs[0].shape[0] != obs_trajs[1].shape[0]:
                ## padding to the max len and convert to np
                obs_trajs = np.array(utils.pad_traj2d_list(obs_trajs))
                # pdb.set_trace()
            else:
                obs_trajs = np.array(obs_trajs)
        ## -------------------------------

        
        assert len(obs_trajs) % ncol == 0, 'Number of paths must be divisible by number of columns'
        
        ## list of H W 4
        images = make_traj_images(self.env, obs_trajs, **kwargs )
        ## B H W 4
        images = np.stack(images, axis=0)
        
        if 'large' in self.env_name and len(obs_trajs) > 1:
            pad_len = 30
        if pad_len > 0: ## pad 0
            images = np.pad(images, ((0,0), (pad_len,)*2, (pad_len,)*2, (0,0)), constant_values=255)
            

        nrow = len(images) // ncol
        img_whole = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        
        if savepath is not None:
            if savepath[-3:] == 'jpg': ## seems to be even larger than png
                ## e.g. (3360, 3000, 4)
                img_whole = img_whole[:, :, :3]
            imageio.imsave(savepath, img_whole)
            print(f'Saved {len(obs_trajs)} samples to: {savepath}')

        if return_rows:
            img_rows = einops.rearrange(images,
            '(nrow ncol) H W C -> nrow H (ncol W) C', nrow=nrow, ncol=ncol)
            return img_whole, img_rows 


        return img_whole


