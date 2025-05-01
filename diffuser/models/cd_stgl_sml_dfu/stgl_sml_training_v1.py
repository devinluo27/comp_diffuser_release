import os
import copy
import numpy as np
import torch
import einops, wandb
import pdb

from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.models.helpers import apply_conditioning
from diffuser.utils.timer import Timer
from diffuser.utils.train_utils import get_lr
from diffuser.utils.training import cycle, EMA
from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
import diffuser.utils as utils

class Stgl_Sml_Trainer_v1(object):
    def __init__(
        self,
        diffusion_model: Stgl_Sml_GauDiffusion_InvDyn_V1,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,

        results_folder='./results',
        n_reference=8,
        n_samples=2,
        device='cuda',
        trainer_dict={},
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq


        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=6, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder

        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0
        self.device = device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for i_tr in range(n_train_steps):
            for i_ac in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                ## important: check what is inside the batch
                # pdb.set_trace()

                # batch = batch_to_device(batch)
                obs_trajs, act_trajs, cond_st_gl = utils.to_device_tp(*batch, device=self.device)
                
                if self.model.tr_cond_type == 'no':
                    cond_st_gl = {}
                # loss, infos = self.model.loss(*batch)
                loss, infos = self.model.loss(x_clean=obs_trajs, cond_st_gl=cond_st_gl)

                # pdb.set_trace()

                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

                # pdb.set_trace()
                ## save to online
                metrics = {k:v.detach().item() for k, v in infos.items()}
                
                metrics['train/it'] = self.step
                metrics['train/loss'] = loss.detach().item()
                metrics['train/lr'] = get_lr(self.optimizer)
                wandb.log(metrics, step=self.step)


            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.ema_model.eval()
                self.render_samples(n_samples=self.n_samples, do_cond=False)
                self.render_samples(n_samples=self.n_samples, do_cond='both_ovlp')
                self.render_samples(n_samples=self.n_samples, do_cond='both_stgl')
                self.render_samples(n_samples=self.n_samples, do_cond='st_endovlp')
                self.render_samples(n_samples=self.n_samples, do_cond='stovlp_gl')

                self.ema_model.train()
                if self.step > 5e5: ## less sampling, faster training
                    self.sample_freq = 30000

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        utils.print_color(f'[ utils/training ] Saved model to {savepath}', c='y')
        

    def load4resume(self, loadpath):
        ## Dec 26
        data = torch.load(loadpath)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.step = data['step']
    

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        obs_trajs = to_np(batch.obs_trajs)

        ## [ batch_size x horizon x observation_dim (4 pos+vel) ]
        normed_observations = obs_trajs # [:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        
        observations = self.get_rowcol_obs_trajs(observations)

        savepath = os.path.join(self.logdir, f'_sample-reference.png')

        is_non_keypt = None # self.model.get_is_non_keypt(batch_size, None).numpy()
        
        self.renderer.composite(savepath, observations, is_non_keypt=is_non_keypt)

    def render_samples(self, batch_size=1, n_samples=2, do_cond=None):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            stgl_cond = to_device(batch.conditions, 'cuda:0')

            ## B,2
            ## repeat each item in conditions `n_samples` times
            stgl_cond = apply_dict(
                einops.repeat,
                stgl_cond,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            # pdb.set_trace()


            ## old: [ n_samples x horizon x (action_dim + observation_dim) ]
            if do_cond == 'both_ovlp':
                # trajs = batch.trajectories

                traj_full = einops.repeat(batch.obs_trajs, 'b h d -> (repeat b) h d', 
                                          repeat=n_samples).to('cuda:0')
                g_cond = dict(do_cond='both_ovlp', traj_full=traj_full, t_type='rand',)


                samples = self.ema_model.conditional_sample(g_cond=g_cond)
            elif do_cond == 'both_stgl':

                traj_full = einops.repeat(batch.obs_trajs, 'b h d -> (repeat b) h d', 
                                          repeat=n_samples).to('cuda:0')
                
                g_cond = dict(do_cond='both_stgl', traj_full=traj_full, t_type='rand', stgl_cond=stgl_cond)
                samples = self.ema_model.conditional_sample(g_cond=g_cond)
                # pdb.set_trace() ## check apply
                samples = apply_conditioning(samples, stgl_cond, 0)
                
            elif do_cond == 'st_endovlp':

                traj_full = einops.repeat(batch.obs_trajs, 'b h d -> (repeat b) h d', 
                                          repeat=n_samples).to('cuda:0')
                g_cond = dict(do_cond='st_endovlp', traj_full=traj_full, t_type='rand', stgl_cond=stgl_cond)
                samples = self.ema_model.conditional_sample(g_cond=g_cond)
                
            elif do_cond == 'stovlp_gl':

                traj_full = einops.repeat(batch.obs_trajs, 'b h d -> (repeat b) h d', 
                                          repeat=n_samples).to('cuda:0')
                g_cond = dict(do_cond='stovlp_gl', traj_full=traj_full, t_type='rand', stgl_cond=stgl_cond)
                samples = self.ema_model.conditional_sample(g_cond=g_cond)

            elif do_cond in [None, False]:
                samples = self.ema_model.sample_unCond(batch_size=len(stgl_cond[0]))
            else:
                raise NotImplementedError
            
            ## (10, 380, 2)
            samples = to_np(samples)

            ##
            act_dim = 0 if self.model.is_inv_dyn_dfu else self.dataset.action_dim
            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, act_dim:]
            # pdb.set_trace()

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]
            

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            observations = self.get_rowcol_obs_trajs(observations)
            ##########

            sample_savedir = self.get_sample_savedir(self.step)
            self.debug_mode = False
            
            if self.debug_mode:
                sample_savedir = os.path.join(self.logdir, f'debug-vis')
                if not os.path.isdir(sample_savedir):
                    os.makedirs(sample_savedir)

            savepath = os.path.join(sample_savedir, f'sample-{self.step}-{i}-{do_cond}.png')

            # is_non_keypt = self.model.get_is_non_keypt(n_samples, None).numpy()
            is_non_keypt = None

            # pdb.set_trace()
            
            is_cond = do_cond not in [None, False]
            if is_cond:
                traj_full_unnorm = self.dataset.normalizer.unnormalize(to_np(traj_full), 'observations')
                ## convert from Ben's format
                traj_full_unnorm = self.get_rowcol_obs_trajs(traj_full_unnorm)
                # pdb.set_trace()
                st_traj_un, end_traj_un = self.ema_model.extract_ovlp_from_full(traj_full_unnorm)
                # sp_xy_1 = torch.cat([st_traj, end_traj], dim=1)
                # pdb.set_trace()
                self.renderer.composite(savepath, observations, is_non_keypt=is_non_keypt, 
                                        sp_xy_1=st_traj_un, sp_xy_2=end_traj_un,)
            else:
                self.renderer.composite(savepath, observations, is_non_keypt=is_non_keypt)


    def get_sample_savedir(self, i):
        div_freq = 100000
        subdir = str( (i // div_freq) * div_freq )
        sample_savedir = os.path.join(self.logdir, subdir)
        if not os.path.isdir(sample_savedir):
            os.makedirs(sample_savedir)
        return sample_savedir


    def get_rowcol_obs_trajs(self, obs_trajs):
        ## special handling for xy in ben's dataset
        dset_type = self.dataset.dset_type 
        if 'ogb' in dset_type:
            ## to support ogbench
            from diffuser.datasets.ogb_dset.ogb_utils import ogb_xy_to_ij
            assert obs_trajs.ndim in [2,3]
            # pdb.set_trace()
            obs_trajs = obs_trajs[..., :2] ## should be B,H,D or H,D

            obs_trajs = ogb_xy_to_ij(self.dataset.env, xy_trajs=obs_trajs)
            
        elif dset_type != 'ours':
            assert 'ben' in dset_type.lower()
            obs_trajs = utils.ben_xy_to_luo_rowcol(dset_type, obs_trajs)
        
        return obs_trajs
