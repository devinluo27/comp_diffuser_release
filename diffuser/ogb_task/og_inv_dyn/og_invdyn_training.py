import os, copy
import numpy as np
import torch, einops, wandb, pdb

from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.models.helpers import apply_conditioning
from diffuser.utils.timer import Timer
from diffuser.utils.train_utils import get_lr
from diffuser.utils.training import cycle, EMA
import diffuser.utils as utils
from diffuser.guides.render_m2d import Maze2dRenderer_V2


class OgB_InvDyn_Trainer_v1(object):
    def __init__(
        self,
        inv_model: torch.nn.Module,
        dataset,
        renderer,
        goal_sel_idxs, ## which dims in obs/stateSpace will be used as goal
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
        self.model = inv_model
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
        self.goal_sel_idxs = goal_sel_idxs
        
        # pdb.set_trace()
        
        self.renderer: Maze2dRenderer_V2 = renderer

        self.optimizer = torch.optim.Adam(inv_model.parameters(), lr=train_lr)

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
        self.model.train()

        for i_tr in range(n_train_steps):
            for i_ac in range(self.gradient_accumulate_every):
                ## obs_trajs conditions returns act_trajs is_pads
                batch = next(self.dataloader)
                ## important: check what is inside the batch
                
                # obs_trajs, cond_st, returns, act_trajs, is_pads = batch

                ## val_lens in range of [2,3,4,...,hzn=10]
                obs_trajs, act_trajs, conditions, val_lens = batch


                ## implement the look-ahead for the inv dyn model
                # pdb.set_trace() ## check shape
                ## obs_trajs: torch.Size([128, 4, 11])
                hzn_tmp = obs_trajs.shape[1] ## e.g., 4
                b_s = obs_trajs.shape[0]
                b_idxs = np.arange( b_s )
                

                if True:
                    ## if hzn=4
                    ## low always 0
                    st_idxs = np.zeros(shape=(b_s,), dtype=np.int32)
                    ## low=1, hzn=4 then possible high=1,2,3, shape: (B,)
                    goal_idxs = np.random.randint(low=st_idxs+1, high=hzn_tmp)

                    # pdb.set_trace() ## check shape

                    ## remove invalid padding states in the trajs
                    ## will be int32 or int64
                    goal_idxs = np.clip(goal_idxs, a_min=0, a_max=val_lens-1)

                    # x_t = obs_trajs[b_idxs, st_idxs, :]
                    x_t = obs_trajs[b_idxs, st_idxs, :]
                    x_t_1 = obs_trajs[b_idxs , goal_idxs, :]
                    
                    ## for goal, just pick the idxs of interest
                    # pdb.set_trace()
                    x_t_1 = x_t_1[:, self.goal_sel_idxs]

                    # pdb.set_trace()
                    ## always the first action
                    a_t = act_trajs[b_idxs, st_idxs, :]

                

                assert x_t.shape[0] == b_s

                # obs_trajs, cond_st, returns, act_trajs, is_pads = utils.to_device_tp(*batch, device=self.device)
                x_t, x_t_1, a_t = utils.to_device_tp(x_t, x_t_1, a_t, device=self.device)


                
                loss, infos = self.model.loss(x_t, x_t_1, a_t)

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

                self.ema_model.train()

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
        

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])




    def load4resume(self, loadpath):
        data = torch.load(loadpath)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.step = data['step']



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

        ## obs_trajs: (20, 96, ant:29)
        ## get trajectories and condition at t=0 from batch
        obs_trajs = to_np(batch.obs_trajs)

        ## [ batch_size x horizon x observation_dim (4 pos+vel) ]
        normed_observations = obs_trajs # [:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        tmp_path_1 = os.path.join(self.logdir, f'_sample-reference.png')

        ## (20, h=6, 2)
        obs_trajs_2d = self.get_rowcol_obs_trajs(observations[:, :, :2])
        # pdb.set_trace()
        
        self.renderer.composite(tmp_path_1, obs_trajs_2d)
        


    def get_rowcol_obs_trajs(self, obs_trajs):
        ## special handling for xy in ben's dataset
        dset_type = self.dataset.dset_type 
        if 'ogb' in dset_type:
            ## to support ogbench
            from diffuser.datasets.ogb_dset.ogb_utils import ogb_xy_to_ij
            assert obs_trajs.ndim == 3
            # pdb.set_trace()
            obs_trajs = obs_trajs[:, :, :2] ## should be B,H,D
            obs_trajs = ogb_xy_to_ij(self.dataset.env, xy_trajs=obs_trajs)
            
        elif dset_type != 'ours':
            assert 'ben' in dset_type.lower()
            obs_trajs = utils.ben_xy_to_luo_rowcol(dset_type, obs_trajs)
        
        return obs_trajs