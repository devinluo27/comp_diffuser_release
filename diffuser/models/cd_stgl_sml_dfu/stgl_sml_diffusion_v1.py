import numpy as np
import torch
from torch import nn
import pdb, einops
import torch.nn.functional as F

import diffuser.utils as utils
from diffuser.models.helpers import (
    cosine_beta_schedule, extract_2d, apply_conditioning, tensor_randint, Losses,
)
from diffuser.models.hi_helpers import MLP_InvDyn
from diffuser.models.cd_stgl_sml_dfu import Unet1D_TjTi_Stgl_Cond_V1
from diffuser.models.comp_dfu.comp_diffusion_v1 import ModelPrediction

class Stgl_Sml_GauDiffusion_InvDyn_V1(nn.Module):
    def __init__(self, model: Unet1D_TjTi_Stgl_Cond_V1, 
                 horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
        diff_config={},
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.obs_manual_loss_weights = diff_config['obs_manual_loss_weights']

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights,)
        # pdb.set_trace() ## check loss_weights shape: [80,2], self.loss_fn.weights
        assert self.obs_manual_loss_weights == {}
        

        ##
        self.diff_config = diff_config
        ## no need for now
        # self.t_rand_prob = diff_config['t_rand_prob']
        ## str, noise injection type besides purely random
        # self.t_fix_type = diff_config['t_fix_type']
        ## same: same t at horizon level, identical to normal diffusion 
        # assert self.t_fix_type in ['same',] or 'hi_' in self.t_fix_type
        # self.setup_hi()
        # self.setup_invdyn()
        self.is_inv_dyn_dfu = True
        self.len_ovlp_cd = self.diff_config['len_ovlp_cd']
        self.is_direct_train = self.diff_config.get('is_direct_train', False)
        if self.is_direct_train:
            self.setup_sep16()
    
    def setup_sep16(self):
        self.infer_deno_type = self.diff_config['infer_deno_type']
        self.w_loss_type = self.diff_config['w_loss_type']
        self.is_train_inv = False
        self.is_train_diff = True
        self.tr_cond_type = 'stgl' ## same as diffuser
        ## changed to 2.0 on Dec 26 00:27 am
        self.condition_guidance_w = self.diff_config.get('condition_guidance_w', 2.0) # 1.0
        # self.eval_n_mcmc = self.diff_config['eval_n_mcmc']
        self.var_temp = 1.0
        self.tr_inpat_prob = self.diff_config['tr_inpat_prob']
        self.tr_ovlp_prob = self.diff_config['tr_ovlp_prob']
        utils.print_color(f"{self.diff_config['tr_1side_drop_prob']=}")
        self.tr_no_ovlp_none = self.diff_config.get('tr_no_ovlp_none', False)
        utils.print_color(f"{self.tr_no_ovlp_none=}")
        assert self.tr_inpat_prob + self.tr_ovlp_prob == 1.0

        ## ----- Dec 3, for DDIM -----
        ddim_set_alpha_to_one = self.diff_config.get('ddim_set_alpha_to_one', True)
        self.final_alpha_cumprod = torch.tensor([1.0,], ) \
            if ddim_set_alpha_to_one else torch.clone(self.alphas_cumprod[0:1]) # tensor of size (1,)
        self.num_train_timesteps = self.n_timesteps
        self.ddim_num_inference_steps = self.diff_config.get('ddim_steps', 50)
        self.ddim_eta = 1.0 # before Dec 31: 0.0
        self.use_ddim = True # False ## NOTE: change from False to True on Dec 9
        self.use_eta_noise = False
        ## ----------------------------


    def setup_invdyn(self):
        """
        setup for inverse dynamic models
        """
        assert False, 'no bug, but not used'
        self.diff_name = 'invdyn'
        self.inv_model_type = self.diff_config['inv_model_type'] # mlp
        if self.inv_model_type == 'mlp':
            self.inv_model = MLP_InvDyn(**self.diff_config['invModel_config'])
        else:
            self.inv_model = None ## no need
            # raise NotImplementedError()
        self.is_train_inv = self.diff_config['is_train_inv']
        self.is_train_diff = self.diff_config['is_train_diff']
        
        
        


    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        # self.action_weight = action_weight
        assert discount == 1 and weights_dict is None

        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[ ind ] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        ## shape: H,dim, e.g., 384,6
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        assert (loss_weights == 1).all()

        ## manually set a0 weight, in default diffuser, all w is 1
        # loss_weights[0, :self.action_dim] = action_weight
        
        ## important: directly set impaint state to loss 0, July 20
        ## actually, this job might be already done in apply_condition in Janner impl
        if len(self.obs_manual_loss_weights) > 0:
            ## idx k is at hzn level
            for k, v in self.obs_manual_loss_weights.items():
                loss_weights[k, :] = v
                print(f'[set manual loss weight] {k} {v}')
        
        # pdb.set_trace()

        return loss_weights

    



    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t_2d, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        assert t_2d.ndim == 2
        # pdb.set_trace()
        ## x_t: B, H, dim
        if self.predict_epsilon:
            ## directly switch to 2d version
            return (
                ## B,H,1 * B,H,dim
                extract_2d(self.sqrt_recip_alphas_cumprod, t_2d, x_t.shape) * x_t -
                extract_2d(self.sqrt_recipm1_alphas_cumprod, t_2d, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        '''from x_0 and x_t to x_{t-1}
        see equeation 6 and 7
        '''
        # pdb.set_trace() ## check buffer dim
        ## directly, e.g., 10,384,6
        posterior_mean = (
            extract_2d(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_2d(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        ## now 2D, not 1D in vanilla diffusion
        ## both two e.g., [B=10, H=384, 1]
        posterior_variance = extract_2d(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_2d(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t_2d, tj_cond, return_modelout=False):
        ## timesteps is 2D tensor: (B, H) ; this is x0 
        # x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t))

        ### batch_repeat_tensor_in_dict Oct 6 1:20am
        # out_model = self.small_model_pred(x, t_2d, tj_cond)
        if tj_cond['do_cond']:
            
            x_2, t_2d_2, tj_cond_2 = utils.batch_repeat_tensor_in_dict(x, t_2d, tj_cond, n_rp=2)

            # pdb.set_trace() ## inspect: the design of half_fd, we only drop ovlp not inpat
            ##
            assert (t_2d_2[0] == t_2d_2[0,0]).all(), 'sanity check'
            t_1d_2 = t_2d_2[:, 0]
            out = self.model(x_2, t_1d_2, tj_cond_2, force_dropout=True, half_fd=True)
            out_cd = out[:len(x), :, :]
            out_uncd = out[len(x):, :, :]
            out_model = out_uncd + self.condition_guidance_w * (out_cd - out_uncd)

        else:
            ## unconditional
            out_model = self.small_model_pred(x, t_2d, tj_cond)






        x_recon = self.predict_start_from_noise(x, t_2d=t_2d, noise=out_model)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t_2d)
        # pdb.set_trace()

        if return_modelout:
            pred_epsilon = self.predict_noise_from_start(x_t=x, t_2d=t_2d, x0=x_recon)
            return model_mean, posterior_variance, posterior_log_variance, x_recon, pred_epsilon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, tj_cond, timesteps, mask_same_t=None):
        """
        mask_same_t: bool tensor, (B,H); if True, the point remains same noise level
        """
        ## timesteps is 2D tensor: (B, H)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t_2d=timesteps, tj_cond=tj_cond,)
        noise = self.var_temp * torch.randn_like(x)
        
        ## TODO: check in hierachy mode
        # no noise when t == 0, shape: B,H,1
        nonzero_mask = (1 - (timesteps == 0).float()).reshape( b, self.horizon, *((1,) * (len(x.shape) - 2)) )
        ## check model_log_variance
        # pdb.set_trace() # check shape
        x_t_minus_1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
        if mask_same_t is not None:
            assert False, 'no bug, not used'
            ## B,H
            assert mask_same_t.shape == timesteps.shape and mask_same_t.dtype == torch.bool
            ## TODO: temporally
            assert (timesteps[mask_same_t] == self.n_timesteps - 1).all() or \
                     (timesteps[mask_same_t] == 0).all() 
            # b_indices = torch.arange( b ).unsqueeze(1).expand_as()
            ## Caution: B,H,dim[(B,H)] -> (n_select,dim)
            # pdb.set_trace()
            ## just copy is enough
            x_t_minus_1[mask_same_t] = x[mask_same_t]
            
            return x_t_minus_1
        else:
            ## normal diffusion
            return x_t_minus_1

    
    def get_tj_cond(self, x, g_cond, timesteps):
        """
        TODO: Directly copy from p_sample_loop, probably we can later use this func in the func
        generate the input for denoising
        - g_cond: a dict
        - timesteps: (B,H)
        """
        if g_cond['do_cond'] == 'both_ovlp': ## full_clean
            st_traj, end_traj = self.extract_ovlp_from_full(g_cond['traj_full'])
            x, tj_cond = self.create_eval_tj_cond(
                x_et=x,
                st_traj=st_traj,
                end_traj=end_traj,
                t_1d_st=timesteps[:,0],
                t_1d_end=timesteps[:,0], 
                # is_rand=True,
                t_type=g_cond['t_type'],
                is_noisy=False,
                stgl_cond={},
                )
            # pdb.set_trace()
            tj_cond['do_cond'] = True
            # pdb.set_trace()
        elif g_cond['do_cond'] == 'both_stgl':
            x, tj_cond = self.create_eval_tj_cond(
                x_et=x,
                st_traj=None,
                end_traj=None,
                t_1d_st=timesteps[:,0],
                t_1d_end=timesteps[:,0], 
                ##
                t_type=g_cond['t_type'],
                is_noisy=False,
                stgl_cond=g_cond['stgl_cond'],
                )
            tj_cond['do_cond'] = True

        elif g_cond['do_cond'] == 'st_endovlp':
            _, end_traj = self.extract_ovlp_from_full(g_cond['traj_full'])
            x, tj_cond = self.create_eval_tj_cond(
                x_et=x,
                st_traj=None,
                end_traj=end_traj,
                t_1d_st=timesteps[:,0],
                t_1d_end=timesteps[:,0], 
                ##
                t_type=g_cond['t_type'],
                is_noisy=False,
                stgl_cond={0: g_cond['stgl_cond'][0]},
                )
            tj_cond['do_cond'] = True
        
        elif g_cond['do_cond'] == 'stovlp_gl':
            st_traj, _ = self.extract_ovlp_from_full(g_cond['traj_full'])
            x, tj_cond = self.create_eval_tj_cond(
                x_et=x,
                st_traj=st_traj,
                end_traj=None,
                t_1d_st=timesteps[:,0],
                t_1d_end=timesteps[:,0], 
                ##
                t_type=g_cond['t_type'],
                is_noisy=False,
                stgl_cond={self.horizon-1: g_cond['stgl_cond'][self.horizon-1] },
                )
            tj_cond['do_cond'] = True


        elif g_cond['do_cond'] == False:
            ## drop everything
            tj_cond = dict(st_ovlp_is_drop=None, end_ovlp_is_drop=None, 
                            is_st_inpat=torch.zeros_like(x[:,0,0]).to(torch.bool),
                            is_end_inpat=torch.zeros_like(x[:,0,0]).to(torch.bool),
                            )
            tj_cond['do_cond'] = False
        else: 
            raise NotImplementedError

        ## x is also modified!
        return x, tj_cond




    

    @torch.no_grad()
    def p_sample_loop(self, shape, g_cond, verbose=True, return_diffusion=False):
        '''
        Temporal, assume when inference, in one step, all t are the same
        '''
        device = self.betas.device

        batch_size = shape[0]
        x = self.var_temp * torch.randn(shape, device=device)
        # x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        from tqdm import tqdm
        for i in tqdm(reversed(range(0, self.n_timesteps))):
            
            ## timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long) # old
            ## e.g., (B=10,H=384)
            timesteps = torch.full((batch_size, self.horizon), i, device=device, dtype=torch.long)
            # pdb.set_trace()

            x, tj_cond = self.get_tj_cond(x, g_cond, timesteps)
            

            x = self.p_sample(x, tj_cond, timesteps)
            # x = apply_conditioning(x, cond, 0)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, g_cond, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        # assert False, 'not finished'
        device = self.betas.device
        batch_size = len(g_cond['traj_full']) ## TODO: check

        # if tj_cond['st_ovlp_is_drop'] is not None:
        #     batch_size = len(tj_cond['st_ovlp_traj'])
        # elif  tj_cond['end_ovlp_is_drop'] is not None:
        #     batch_size = len(tj_cond['end_ovlp_traj'])
        # else:
        #     assert False
        # pdb.set_trace()

        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        if self.infer_deno_type == 'hi_fix_v1':
            raise NotImplementedError
            return self.p_sample_loop_hi_fix_v1(shape, cond, *args, **kwargs)
        elif self.infer_deno_type == 'same':
            if self.use_ddim:
                return self.ddim_p_sample_loop(shape, g_cond, *args, **kwargs)
            else:
                return self.p_sample_loop(shape, g_cond, *args, **kwargs)
        else:
            raise NotImplementedError
            return self.p_sample_loop(shape, cond, *args, **kwargs)
        
    @torch.no_grad()
    def sample_unCond(self, batch_size, *args, horizon=None, **kwargs):
        '''
            batch_size : int
        '''
        device = self.betas.device
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)
        g_cond = dict(do_cond=False)
        ## placeholder

        if self.infer_deno_type == 'hi_fix_v1':
            raise NotImplementedError
            # return self.p_sample_loop_hi_fix_v1(shape, cond, *args, **kwargs)
        elif self.infer_deno_type == 'same':
            return self.p_sample_loop(shape, g_cond, *args, **kwargs)
        else:
            raise NotImplementedError



    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t_2d, noise=None, mask_no_noise=None):
        '''add noise to x_t from x_0
        mask_no_noise: bool (B,H), if True, then do not add any noise to x_start
        '''
        assert t_2d.ndim == 2 # B, horizon

        if noise is None:
            noise = torch.randn_like(x_start)
        
        ## vanilla diffusion: (B, 1, 1) * x_start: (B, H, dim) e.g., [32, 128, 6]
        # sample = (
        #     extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        #     extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        # )

        ## Ours: t (B, H, 1) * x_start (B, H, dim)
        q_coef1 = extract_2d(self.sqrt_alphas_cumprod, t_2d, x_start.shape)
        q_coef2 = extract_2d(self.sqrt_one_minus_alphas_cumprod, t_2d, x_start.shape)

        ## when t=0, 0.9999 * x_start + 0.0137 * noise 
        sample = q_coef1 * x_start + q_coef2 * noise

        # pdb.set_trace()
        if mask_no_noise is not None:
            assert False
            ## (B H C)[(B,H)]
            assert mask_no_noise.shape == x_start.shape[:2] and mask_no_noise.dtype == torch.bool
            ## keypt should not have any noise
            sample[mask_no_noise] = x_start[mask_no_noise]


        return sample

    def p_losses(self, x_start, x_noisy, noise, t_2d, tj_cond, mask_no_noise=None, batch_loss_w=None,):
        '''batch_loss_w: tensor (B,H) float, the loss weight of this specific batch'''
        assert self.is_direct_train
        ## torch.Size([B, H=384, dim=6]) state+action
        # noise = torch.randn_like(x_start)

        # x_noisy = self.q_sample(x_start=x_start, t_2d=t_2d, noise=noise, mask_no_noise=mask_no_noise)


        # x_noisy = apply_conditioning(x_noisy, cond, 0)

        ## Make sure tj_cond is totaly ready here
        ## check loss weight for inpaint part!

        if self.w_loss_type == 'no_unused':
            assert False
            batch_loss_w = batch_loss_w[:, :, None] ## B,H,1
        elif self.w_loss_type == 'all':
            # batch_loss_w = 1.
            ## B,H,1
            batch_loss_w = torch.ones_like(x_start[:, :, :1])
            ## (B,), no need loss for the inpainting state
            batch_loss_w[tj_cond['is_st_inpat'], 0] = 0.
            batch_loss_w[tj_cond['is_end_inpat'], self.horizon-1] = 0.
            # pdb.set_trace()
            assert x_start.shape[1] == self.horizon

        else:
            raise NotImplementedError()
        
        # pdb.set_trace()

        # x_recon = self.model(x_noisy, t)
        x_recon = self.small_model_pred(x_noisy, t_2d, tj_cond)
        ## NOTE:
        ## looks like this can auto ignore loss at t=0 and -1? if x_recon is pred_x_0
        # x_recon = apply_conditioning(x_recon, cond, 0)

        assert x_noisy.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, ext_loss_w=batch_loss_w)
        else:
            loss, info = self.loss_fn(x_recon, x_start, ext_loss_w=batch_loss_w)

        return loss, info
    
    def apply_cond_mask(self, x_recon, tj_cond):
        pass
    
    def extract_ovlp_from_full(self, x: torch.Tensor):
        """x: either np or tensor"""
        st_traj = x[:, :self.len_ovlp_cd, :]
        end_traj = x[:, -self.len_ovlp_cd:, :]
        if torch.is_tensor(st_traj):
            assert torch.is_tensor(end_traj)
            st_traj = st_traj.detach().clone()
            end_traj = end_traj.detach().clone()
        else:
            assert type(st_traj) == np.ndarray
            assert type(end_traj) == np.ndarray

        return st_traj, end_traj



    def create_train_tj_cond(self, x_clean: torch.Tensor, 
                             x_noisy: torch.Tensor, ## for model input, change inpainting part
                             t_1d_st: torch.Tensor, 
                             t_1d_end: torch.Tensor, 
                             cond_st_gl: dict,
                             is_rand):
        """
        t_1d: (B,)
        """
        batch_size = x_clean.shape[0]
        # pdb.set_trace() ## use larger bs
        all_drop_prob = self.diff_config['tr_1side_drop_prob'] ## 0.15
        device = x_clean.device

        # st_is_all_drop = np.random.uniform(low=0, high=1, size=(batch_size,)) < all_drop_prob
        # end_is_all_drop = np.random.uniform(low=0, high=1, size=(batch_size,)) < all_drop_prob

        st_is_all_drop = torch.rand( size=(batch_size,), device=device ) < all_drop_prob
        end_is_all_drop = torch.rand( size=(batch_size,), device=device ) < all_drop_prob


        ### ------------ Useless
        # ovlp_drop_prob = self.diff_config['tr_ovlp_drop_prob'] ## 0.10
        # inpat_drop_prob = self.diff_config['tr_inpat_drop_prob'] ## 0.10
        # st_is_ovlp_drop = torch.rand( size=(batch_size,), device=x.device ) < ovlp_drop_prob
        # st_is_inpat_drop = torch.rand( size=(batch_size,), device=x.device ) < inpat_drop_prob
        ### ------------
        # pdb.set_trace()


        ## True if we want to do condition
        ## (B,); bool, True if this sample will use overlap as condition
        st_cd_use_ovlp = torch.rand( size=(batch_size,), device=device ) < self.tr_ovlp_prob
        end_cd_use_ovlp = torch.rand( size=(batch_size,), device=device ) < self.tr_ovlp_prob

        st_cd_use_inpat = ~ st_cd_use_ovlp
        end_cd_use_inpat = ~ end_cd_use_ovlp

        ## set those to be dropout to False, so no condition at all for 0.15 * bs
        st_cd_use_ovlp[st_is_all_drop] = False
        st_cd_use_inpat[st_is_all_drop] = False

        end_cd_use_ovlp[end_is_all_drop] = False
        end_cd_use_inpat[end_is_all_drop] = False

        ####### ----- NEW, Oct 11, we can ignore useless conditions.
        if self.tr_no_ovlp_none: ## default False
            tmp_st_nn = torch.logical_and(st_cd_use_ovlp, end_is_all_drop)
            tmp_stnn_vv = torch.rand_like(tmp_st_nn.to(torch.float32)) < self.diff_config['non_repla_inpat_prob'] # 0.5
            tmp_stnn_inpat = torch.logical_and(tmp_st_nn, tmp_stnn_vv)
            # tmp_stnn_none = torch.logical_and(tmp_st_nn, ~tmp_stnn_vv)
            # if tmp_no_need.any():
                # n_repl = tmp_no_need.sum().item()
            # st_cd_use_ovlp[tmp_stnn_] = True
            st_cd_use_ovlp[tmp_st_nn] = False 
            st_cd_use_inpat[tmp_stnn_inpat] = True

            tmp_end_nn = torch.logical_and(st_is_all_drop, end_cd_use_ovlp)
            tmp_endnn_vv = torch.rand_like(tmp_end_nn.to(torch.float32)) < self.diff_config['non_repla_inpat_prob']
            tmp_endnn_inpat = torch.logical_and(tmp_end_nn, tmp_endnn_vv)
            ###
            end_cd_use_ovlp[tmp_end_nn] = False
            end_cd_use_inpat[tmp_endnn_inpat] = True

        ####### -------------
        ####### -------------


        ###### TODO:
        ###### We need to modify the corresponding samples for inpainting
        # pdb.set_trace() ## check cond_st_gl --> 0: (B,2), horizon-1:(B,2)
        cond_st = { 0: cond_st_gl[0][st_cd_use_inpat] }
        x_noisy[ st_cd_use_inpat ] = apply_conditioning( x_noisy[ st_cd_use_inpat ], conditions=cond_st, action_dim=0 )
        ####
        cond_end = {self.horizon-1: cond_st_gl[self.horizon-1][end_cd_use_inpat]}
        x_noisy[ end_cd_use_inpat ] = apply_conditioning( x_noisy[ end_cd_use_inpat ], cond_end, 0 )

        # pdb.set_trace() #### check if replace properly



        ## t range is [0, 255(self.n_timesteps-1)], the only available range is -0 or -1
        if is_rand:
            t_1d_st = t_1d_st - torch.randint_like(t_1d_st, low=0, high=2) # [0,2)
            t_1d_end = t_1d_end - torch.randint_like(t_1d_end, low=0, high=2)
        else:
            assert False
            # t_1d_st = t_1d.clone()
            # t_1d_end = t_1d.clone()
        ## Oct 6 newly Added
        t_1d_st = torch.clamp(t_1d_st, min=0, max=self.n_timesteps-1)
        t_1d_end = torch.clamp(t_1d_end, min=0, max=self.n_timesteps-1)


        ## TODO: slow, can be improved
        t_2d_st = torch.repeat_interleave(t_1d_st[:, None], repeats=self.len_ovlp_cd, dim=1)


        t_2d_end = torch.repeat_interleave(t_1d_end[:, None], repeats=self.len_ovlp_cd, dim=1)
        # pdb.set_trace()

        ## add noise
        st_traj = x_clean[:, :self.len_ovlp_cd, :].detach().clone()
        st_traj = self.q_sample(x_start=st_traj, t_2d=t_2d_st, noise=None, mask_no_noise=None)

        end_traj = x_clean[:, -self.len_ovlp_cd:, :].detach().clone()
        end_traj = self.q_sample(x_start=end_traj, t_2d=t_2d_end, noise=None, mask_no_noise=None)

        # pdb.set_trace()

        tj_cond = {
            'st_ovlp_is_drop': ~st_cd_use_ovlp, # st_is_drop,
            'end_ovlp_is_drop': ~end_cd_use_ovlp, #end_is_drop,
            ##
            'st_ovlp_traj': st_traj,
            'end_ovlp_traj': end_traj,
            ##
            'st_ovlp_t': t_1d_st,
            'end_ovlp_t': t_1d_end,
            ## NEW
            'is_st_inpat': st_cd_use_inpat,
            'is_end_inpat': end_cd_use_inpat,

        }
        # pdb.set_trace() ## TODO: check

        return x_noisy, tj_cond



    def loss(self, x_clean, cond_st_gl):
        assert self.is_direct_train
        batch_size = len(x_clean)
        # pdb.set_trace() ## check x dim
        if True:
            t_1d = torch.randint(0, self.n_timesteps, (batch_size, 1), device=x_clean.device).long()
            ## B,H
            t_2d = torch.repeat_interleave(t_1d, repeats=self.horizon, dim=1)

            noise = torch.randn_like(x_clean)
            x_noisy = self.q_sample(x_start=x_clean, t_2d=t_2d, noise=noise, mask_no_noise=None)

            ## TODO: update p_losses to feed our current design, we generate x_noisy outside!
            # pdb.set_trace() ## TODO: Oct 10 1:52am, check everything around...
            x_noisy, tj_cond = self.create_train_tj_cond(x_clean, x_noisy, 
                                    t_1d[:, 0], t_1d[:,0].clone(), cond_st_gl, is_rand=True)
            # pdb.set_trace()

        else:
            raise NotImplementedError



        if self.is_train_inv:
            assert False
            inv_loss = self.compute_invdyn_loss(x_start=x)
        else:
            inv_loss = 0.
        
        if self.is_train_diff:
            b_mask_no_noise, batch_loss_w = None, None
            # pdb.set_trace()
            diffuse_loss, info = self.p_losses(x_clean[:, :, :], x_noisy, noise, t_2d, tj_cond,
                    mask_no_noise=b_mask_no_noise, batch_loss_w=batch_loss_w,)
        else:
            assert False
            diffuse_loss, info = 0, {}
            
        total_loss = inv_loss + diffuse_loss
        return total_loss, info


    
    

    def pred_x0_from_xt(self, x_t, t_2d):
        """
        from x_t to x_0, for mcmc
        """
        raise NotImplementedError
        assert x_t.shape[2] == self.observation_dim
        # pdb.set_trace()
        ## might be x0 or noise
        out_pred = self.small_model_pred(x_t, t_2d)
        ## timesteps is 2D tensor: (B, H) ; this is x0 
        x_recon = self.predict_start_from_noise(x_t, t_2d=t_2d, noise=out_pred )

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        
        return x_recon

    def small_model_pred(self, x, t_2d, tj_cond: dict, ) -> torch.Tensor:
        
        ## simple model should take in only 1d
        if self.model.input_t_type == '1d':
            # pdb.set_trace() ## check t_2d
            assert (t_2d[0] == t_2d[0,0]).all(), 'sanity check'
            t_1d = t_2d[:, 0]
            pred_out = self.model(x, t_1d, tj_cond)

        elif self.model.input_t_type == '2d':
            raise NotImplementedError
            pred_out = self.model(x, t_2d)
        else:
            raise NotImplementedError
        # pdb.set_trace() ## check the dim of t ...

        return pred_out



    def create_eval_tj_cond(self, 
                            x_et: torch.Tensor, ## x eval t
                            st_traj, # : torch.Tensor, 
                            end_traj,
                             t_1d_st: torch.Tensor, 
                             t_1d_end: torch.Tensor, 
                             t_type: str,
                             is_noisy,
                             stgl_cond:dict,):
        """
        t_1d: (B,)
        if st_traj is not None, then do st traj inpainting;
        if end_traj is not None, then do end traj inpainting;
        if 0 in stgl_cond, then do start inpainting;
        if hzn-1 in stgl_cond, then do end inpainting;
        """
        assert t_1d_st.ndim == 1 and t_1d_end.ndim == 1

        ## TODO: check the case where both end do inpainting, and using cls-free guidance

        batch_size = x_et.shape[0] # if st_traj is not None else end_traj.shape[0]
        device = x_et.device
        # batch_bool_args = dict(
            # size=(batch_size,), dtype=torch.bool, device=device)
        d_dim = x_et.shape[2]
        
        # batch_size = st_traj.shape[0]
        if st_traj == None:
            ## drop everything
            # st_traj = torch.zeros_like(end_traj)
            # st_traj = torch.zeros(size=(batch_size, self.len_ovlp_cd, d_dim), 
                                #   dtype=x_et.dtype, device=device)
            # st_is_drop = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
            st_is_drop = None
        else:
            ## keep everything ## From Here Oct 8 20:51 TODO: using tensor might be slower than np?
            # st_is_drop =  np.zeros(shape=(batch_size,), dtype=bool) ## no drop
            st_is_drop = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device) ## no drop

            assert 0 not in stgl_cond.keys()
            # batch_size = st_traj.shape[0]
        
        ## start conditioning
        if 0 in stgl_cond.keys():
            # pdb.set_trace()
            assert st_traj is None
            st_cond = {0: stgl_cond[0]}
            x_et = apply_conditioning(x_et, st_cond, 0)
            is_st_inpat = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
        else:
            is_st_inpat = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)



        
        hzn_minus1 = self.horizon - 1
        if end_traj == None:
            # end_traj = torch.zeros_like(st_traj)
            # end_is_drop = np.ones(shape=(batch_size,), dtype=bool)
            end_is_drop = None
        else:
            # end_is_drop = np.zeros(shape=(batch_size,), dtype=bool)
            end_is_drop = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)
            # batch_size = end_traj.shape[0]
            assert hzn_minus1 not in stgl_cond

        ## do inpainting conditioning
        if hzn_minus1 in stgl_cond.keys():
            # pdb.set_trace()
            assert end_traj is None
            end_cond = {hzn_minus1: stgl_cond[hzn_minus1]}
            x_et = apply_conditioning(x_et, end_cond, 0)
            is_end_inpat = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
        else:
            is_end_inpat = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)











        ## each end should have some conditions
        assert ( (st_traj is not None) or 0 in stgl_cond ) and \
                    ( (end_traj is not None) or hzn_minus1 in stgl_cond )


        
        # pdb.set_trace()

        ## t range is [0, 255(self.n_timesteps-1)], the only available range is -0 or -1
        if t_type == 'rand':
            t_1d_st = t_1d_st - torch.randint_like(t_1d_st, low=0, high=2) # [0,2)
            t_1d_end = t_1d_end - torch.randint_like(t_1d_end, low=0, high=2)
        elif t_type == '-1':
            t_1d_st = t_1d_st - torch.ones_like(t_1d_st,)
            t_1d_end = t_1d_end - torch.ones_like(t_1d_end,)
        elif t_type == '0':
            pass
        else:
            raise NotImplementedError
            # t_1d_st = t_1d.clone()
            # t_1d_end = t_1d.clone()
        t_1d_st = torch.clamp(t_1d_st, min=0, max=self.n_timesteps-1)
        t_1d_end = torch.clamp(t_1d_end, min=0, max=self.n_timesteps-1)


        t_2d_st = torch.repeat_interleave(t_1d_st[:, None], repeats=self.len_ovlp_cd, dim=1)
        t_2d_end = torch.repeat_interleave(t_1d_end[:, None], repeats=self.len_ovlp_cd, dim=1)
        # pdb.set_trace()

        ## add noise
        # detach().clone()
        # if not is_noisy:
        #     st_traj = self.q_sample(x_start=st_traj, t_2d=t_2d_st, noise=None, mask_no_noise=None)
        #     end_traj = self.q_sample(x_start=end_traj, t_2d=t_2d_end, noise=None, mask_no_noise=None)
        # else:
        #     st_traj = st_traj.clone()
        #     end_traj = end_traj.clone()
        
        ### -------
        if st_traj == None:
            pass
        elif not is_noisy:
            st_traj = self.q_sample(x_start=st_traj, t_2d=t_2d_st, noise=None, mask_no_noise=None)
        else:
            st_traj = st_traj.clone()
        
        if end_traj == None:
            pass
        elif not is_noisy:
            end_traj = self.q_sample(x_start=end_traj, t_2d=t_2d_end, noise=None, mask_no_noise=None)
        else:
            end_traj = end_traj.clone()

        # pdb.set_trace()

        tj_cond = {
            'st_ovlp_is_drop': st_is_drop,
            'end_ovlp_is_drop': end_is_drop,
            ## must be noisy
            'st_ovlp_traj': st_traj,
            'end_ovlp_traj': end_traj,
            ## TODO:
            'st_ovlp_t': t_1d_st,
            'end_ovlp_t': t_1d_end,
            ##
            'is_st_inpat': is_st_inpat,
            'is_end_inpat': is_end_inpat,

        }
        # pdb.set_trace()

        return x_et, tj_cond
    



    def predict_noise_from_start(self, x_t, t_2d, x0):
        return (
            extract_2d(self.sqrt_recip_alphas_cumprod, t_2d, x_t.shape) * x_t - x0) / \
                    extract_2d(self.sqrt_recipm1_alphas_cumprod, t_2d, x_t.shape
            )


    def model_predictions(self, x, t_2d, tj_cond: dict):
        # out_pred = self.comp_model_pred(x, t_2d)
        self.clip_noise = 20.0

        ## ------- should be similar to above -----
        if tj_cond['do_cond']:
            
            x_2, t_2d_2, tj_cond_2 = utils.batch_repeat_tensor_in_dict(x, t_2d, tj_cond, n_rp=2)

            ##
            assert (t_2d_2[0] == t_2d_2[0,0]).all(), 'sanity check'
            t_1d_2 = t_2d_2[:, 0]
            out = self.model(x_2, t_1d_2, tj_cond_2, force_dropout=True, half_fd=True)
            out_cd = out[:len(x), :, :]
            out_uncd = out[len(x):, :, :]
            out_model = out_uncd + self.condition_guidance_w * (out_cd - out_uncd)

        else:
            ## unconditional
            out_model = self.small_model_pred(x, t_2d, tj_cond)
        ## -------
        out_pred = out_model


        if self.predict_epsilon:
            pred_noise = torch.clamp(out_pred, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x_t=x, t_2d=t_2d, noise=pred_noise)
        else:
            x_start = out_pred
            pred_noise = self.predict_noise_from_start(x, t_2d, x_start)

        if self.clip_denoised:
            x_start.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        return ModelPrediction(pred_noise, x_start, out_pred)
    


    def get_total_hzn(self, num_comp):
        return num_comp * self.horizon - \
                    (num_comp - 1) * self.len_ovlp_cd
    


    def ddim_set_timesteps(self, num_inference_steps) -> np.ndarray: 

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        # e.g., 10: [90, 80, 70, 60, 50, 40, 30, 20, 10,  0]
        
        return timesteps
    

    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, g_cond, verbose=True, return_diffusion=False):
        
        utils.print_color(f'ddim steps: {self.ddim_num_inference_steps}', c='y')
        
        device = self.betas.device
        batch_size = shape[0]
        x = self.var_temp * torch.randn(shape, device=device)
        # x = apply_conditioning(x, cond, 0) # start from dim 0, different from diffuser

        if return_diffusion: diffusion = [x]
        # 100 // 20 = 5
        ## e.g., array([459, 408, 357, 306, 255, 204, 153, 102,  51,   0])
        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)


        # pdb.set_trace() ## check time_idx
        # for i in time_idx: # if np array, i is <class 'numpy.int64'>
        from tqdm import tqdm
        for i in tqdm(time_idx):
            
            
            timesteps = torch.full((batch_size, self.horizon), i, device=device, dtype=torch.long)
            
            ## this get func is not for composing
            x, tj_cond = self.get_tj_cond(x, g_cond, timesteps)

            ## From Here Dec 3, 16:04

            ## ----------------------
            ### eta=0.0
            x = self.ddim_p_sample(x, tj_cond, timesteps, self.ddim_eta, use_clipped_model_output=True)
            # x = apply_conditioning(x, cond, 0)

            if return_diffusion: diffusion.append(x)
        

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        


    @torch.no_grad()
    def ddim_p_sample(self, x, tj_cond, timesteps, eta=0.0, 
                      use_clipped_model_output=False):
        ''' NOTE follow diffusers ddim, any post-processing *NOT CHECKED yet*
        timesteps (cuda tensor [B,H]) must be same
        eta: weight for noise
        '''
        
        # # 1. get previous step value (=t-1), (B,)
        prev_timestep = timesteps - self.num_train_timesteps // self.ddim_num_inference_steps
        # # 2. compute alphas, betas
        alpha_prod_t = extract_2d(self.alphas_cumprod, timesteps, x.shape) # 
        
        # pdb.set_trace()
        
        assert torch.isclose(prev_timestep[0,],prev_timestep[0, 0]).all()
        if prev_timestep[0, 0] >= 0:
            alpha_prod_t_prev = extract_2d(self.alphas_cumprod, prev_timestep, x.shape) # tensor 
        else:
            # extract from a tensor of size 1, cuda tensor [80, 1, 1]
            alpha_prod_t_prev = extract_2d(self.final_alpha_cumprod.to(timesteps.device), torch.zeros_like(timesteps), x.shape)
            # print(f'alpha_prod_t_prev {alpha_prod_t_prev[0:3]}')
        assert alpha_prod_t.shape == alpha_prod_t_prev.shape





        beta_prod_t = 1 - alpha_prod_t

        # b, *_, device = *x.shape, x.device
        

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # 4. Clip "predicted x_0"
        ## model_mean is clipped x_0, 
        ## model_output: model prediction, should be the epsilon (noise)
        
        ## model_mean, _, model_log_variance, x_recon, model_output = self.p_mean_variance(x=x, cond=cond, t=t, walls_loc=walls_loc, return_modelout=True)

        ## TODO: from Here Dec 3 -- 16:48, finish DDIM for ours.
        model_mean, _, model_log_variance, x_recon, model_output = \
                self.p_mean_variance(x=x, t_2d=timesteps, tj_cond=tj_cond, return_modelout=True)




        ## 5. compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) \
            * ( 1 - alpha_prod_t / alpha_prod_t_prev )

        std_dev_t = eta * variance ** (0.5)

        assert use_clipped_model_output
        if use_clipped_model_output:

            sample = x
            pred_original_sample = x_recon
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
            


        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        ## ----------------------------------------------
        ## NOTE: NEW Added on Jan 19 DDIM ETA
        if self.use_eta_noise and eta > 0:
            variance_noise = torch.randn_like(model_output,)
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance




        sample = prev_sample
        
        
        return sample









    ## ----- Oct 20, copy from plan_stgl_sml_v2.ipynb -----
    @torch.no_grad()
    def resample_same_t_mcmc(self, x: torch.Tensor,
            tj_cond, timesteps,):
        x_t_mc = torch.clone(x)
        for i_mc in range(self.eval_n_mcmc):
            ## 1. from x_t to x_0; x_0 shape (B, H, tot_hzn)
            model_pred = self.model_predictions(x=x_t_mc, t_2d=timesteps, tj_cond=tj_cond )
            recon_x0 = model_pred.pred_x_start
            ## 2. apply condition to x_0; check cond's key
            # x_0 = apply_conditioning(x_0, cond, 0)
            ## 3. add noise back to x_t
            recon_x0.clamp_(-1., 1.)
            x_t_mc = self.q_sample(x_start=recon_x0, t_2d=timesteps,) # noise=None, mask_no_noise=None)
        return x_t_mc


    ## TODO: Dec 30 from This function add ddim
    @torch.no_grad()
    def comp_pred_p_loop_n(self, ##  
                        shape, stgl_cond, n_comp, do_mcmc=False, verbose=True, return_diffusion=False):
        """assume compose n trajectories"""
        assert n_comp >= 2 and not do_mcmc, 'mcmc might be bad in our case and not implemented for DDIM yet.'
        device = self.betas.device

        batch_size = shape[0]
        hzn = shape[1]

        x_p_list = [ torch.randn(shape, device=device) for _ in range(n_comp) ]

        x_dfu_all = [x_p_list,]

        assert len(stgl_cond[0]) == shape[0]

        ## --- NEW Dec 4 ---
        """1. change the ddim time_dix; 2. change the p_sample
        """
        if self.use_ddim:
            time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        else:
            time_idx = reversed(range(0, self.n_timesteps))
        ## -----------------

        from tqdm import tqdm
        for i_t in tqdm(time_idx):
            
            ## timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long) # old
            ## e.g., (B=10,H=384)
            timesteps = torch.full((batch_size, self.horizon), i_t, device=device, dtype=torch.long)

            ## iteratively denoise each sub traj
            for i_tj in range(n_comp):
                ## target traj
                x_p_i = x_p_list[i_tj]

                if i_tj == 0:
                    ## first one
                    x_p_i_plus_1 = x_p_list[i_tj+1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=timesteps[:,0],
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={0:stgl_cond[0]}
                        )
                    
                    tj_cond_p_i['do_cond'] = True
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    x_p_list[i_tj] = x_p_i
                
                elif i_tj > 0 and i_tj < n_comp-1:
                    ## intermediate one
                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i_plus_1 = x_p_list[ i_tj + 1 ]
                    st_traj_i_plus_1, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=st_traj_i_plus_1,
                        t_1d_st=timesteps[:,0]-1,
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={},
                    )

                    tj_cond_p_i['do_cond'] = True

                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    x_p_list[i_tj] = x_p_i

                elif i_tj == n_comp - 1:
                    ## last one

                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _,  end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=timesteps[:,0]-1,
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={hzn-1:stgl_cond[hzn-1]}
                        )
                    tj_cond_p_i['do_cond'] = True

                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc( x_p_i, tj_cond_p_i, timesteps )

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    x_p_list[i_tj] = x_p_i
            
            if return_diffusion:
                x_dfu_all.append([_ for _ in x_p_list])


        #### -----------
            
        
        ## Finished
        x_p_list[0] = apply_conditioning(x_p_list[0], {0:stgl_cond[0]}, 0)
        x_p_list[-1] = apply_conditioning(x_p_list[-1], {hzn-1:stgl_cond[hzn-1]}, 0)

        ## TODO: Dec 30, we can clamp the output to -1,1 as loc here

        if return_diffusion:
            ## _, a list of x_p_list
            return x_p_list, x_dfu_all
        else:
            return x_p_list
        
    ## ----- moved end -----




    ## TODO: Dec 30 from This function add ddim
    @torch.no_grad()
    def comp_pred_p_loop_n_same_t(self, ##  
                        shape, stgl_cond, n_comp, do_mcmc=False, verbose=True, return_diffusion=False):
        """assume compose n trajectories"""
        assert n_comp >= 2 and not do_mcmc, 'mcmc might be bad in our case and not implemented for DDIM yet.'
        device = self.betas.device

        batch_size = shape[0]
        hzn = shape[1]

        x_p_list = [ torch.randn(shape, device=device) for _ in range(n_comp) ]

        x_dfu_all = [x_p_list,]

        assert len(stgl_cond[0]) == shape[0]

        ## --- NEW Dec 4 ---
        """1. change the ddim time_dix; 2. change the p_sample
        """
        if self.use_ddim:
            time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        else:
            time_idx = reversed(range(0, self.n_timesteps))
        ## -----------------

        from tqdm import tqdm
        for i_t in tqdm(time_idx):
            
            ## timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long) # old
            ## e.g., (B=10,H=384)
            timesteps = torch.full((batch_size, self.horizon), i_t, device=device, dtype=torch.long)

            ## x_p_list ## to be given to the denoiser at this step
            x_p_list_next_t = [None for _ in range(n_comp)] ## after denoise this step, aka., less noisy

            ## iteratively denoise each sub traj
            for i_tj in range(n_comp):
                ## target traj
                x_p_i = x_p_list[i_tj]

                if i_tj == 0:
                    ## first one
                    x_p_i_plus_1 = x_p_list[i_tj+1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=timesteps[:,0], ## placeholder
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={0:stgl_cond[0]}
                        )
                    
                    tj_cond_p_i['do_cond'] = True
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    # x_p_list[i_tj] = x_p_i
                    ## x_p_list (old list): still save the samples
                    ## put the less noisy sample in the new list
                    x_p_list_next_t[i_tj] = x_p_i
                
                elif i_tj > 0 and i_tj < n_comp-1:
                    ## intermediate one
                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i_plus_1 = x_p_list[ i_tj + 1 ]
                    st_traj_i_plus_1, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=st_traj_i_plus_1,
                        t_1d_st=timesteps[:,0], ## same noisy level
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={},
                    )

                    tj_cond_p_i['do_cond'] = True

                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    # x_p_list[i_tj] = x_p_i
                    x_p_list_next_t[i_tj] = x_p_i

                elif i_tj == n_comp - 1:
                    ## last one

                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _,  end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=timesteps[:,0],
                        t_1d_end=timesteps[:,0], ## placeholder
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={hzn-1:stgl_cond[hzn-1]}
                        )
                    tj_cond_p_i['do_cond'] = True

                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc( x_p_i, tj_cond_p_i, timesteps )

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    # x_p_list[i_tj] = x_p_i
                    x_p_list_next_t[i_tj] = x_p_i
            
            ## important: NEW assign the next_t to cur_t for denosing at the next timestep
            x_p_list = x_p_list_next_t

            if return_diffusion:
                x_dfu_all.append([_ for _ in x_p_list])


        #### -----------
            
        
        ## Finished
        x_p_list[0] = apply_conditioning(x_p_list[0], {0:stgl_cond[0]}, 0)
        x_p_list[-1] = apply_conditioning(x_p_list[-1], {hzn-1:stgl_cond[hzn-1]}, 0)

        ## TODO: Dec 30, we can clamp the output to -1,1 as loc here

        if return_diffusion:
            ## _, a list of x_p_list
            return x_p_list, x_dfu_all
        else:
            return x_p_list
        



    ## Feb 13, Change to Parallel Sampling
    @torch.no_grad()
    def comp_pred_p_loop_n_same_t_parallel(self, ##  
                        shape, stgl_cond, n_comp, do_mcmc=False, verbose=True, return_diffusion=False):
        """assume compose n trajectories"""
        assert n_comp >= 2 and not do_mcmc, 'mcmc might be bad in our case and not implemented for DDIM yet.'
        device = self.betas.device

        batch_size = shape[0]
        hzn = shape[1]

        x_p_list = [ torch.randn(shape, device=device) for _ in range(n_comp) ]

        x_dfu_all = [x_p_list,]

        assert len(stgl_cond[0]) == shape[0]

        ## --- NEW Dec 4 ---
        """1. change the ddim time_dix; 2. change the p_sample
        """
        if self.use_ddim:
            time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        else:
            time_idx = reversed(range(0, self.n_timesteps))
        ## -----------------

        from tqdm import tqdm
        for i_t in tqdm(time_idx):
            
            ## timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long) # old
            ## e.g., (B=10,H=384)
            timesteps = torch.full((batch_size, self.horizon), i_t, device=device, dtype=torch.long)

            ## x_p_list: noisy x at the current t
            ## x_p_list_cur_t: the pre-processed input for the denoiser
            x_p_list_cur_t = [None for _ in range(n_comp)]
            ## pre-processed condition for the denoiser
            tj_cond_cur_list = [None for _ in range(n_comp)]

            ## iteratively denoise each sub traj
            for i_tj in range(n_comp):
                ## target traj
                x_p_i = x_p_list[i_tj]

                if i_tj == 0:
                    ## first one
                    x_p_i_plus_1 = x_p_list[i_tj+1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=timesteps[:,0], ## placeholder
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={0:stgl_cond[0]}
                        )
                
                elif i_tj > 0 and i_tj < n_comp-1:
                    ## intermediate one
                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i_plus_1 = x_p_list[ i_tj + 1 ]
                    st_traj_i_plus_1, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=st_traj_i_plus_1,
                        t_1d_st=timesteps[:,0], ## same noisy level
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={},
                    )

                elif i_tj == n_comp - 1:
                    ## last one

                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _,  end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=timesteps[:,0],
                        t_1d_end=timesteps[:,0], ## placeholder
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={hzn-1:stgl_cond[hzn-1]}
                        )
                
                
                ### since we update x_p_i (e.g., do inpainting), so we need to update the list
                x_p_list_cur_t[i_tj] = x_p_i ## new x_p_i for the model forward
                tj_cond_cur_list[i_tj] = tj_cond_p_i ## for the model forward

                # pdb.set_trace() ## check the data type np or tensor?




            ## ========= Finish setting up all the tj_cond ==========
            ## Now start to do denoising

            ## TODO: Feb 13, 14:01, implement parallel version and get the time.
            def merge_tj_cond_list(tj_cond_list: list):
                len_tjc = len(tj_cond_list)
                assert len_tjc >= 2
                keys_tjc = tj_cond_list[0].keys()
                mg_dict = {}
                b_s = tj_cond_list[0]['is_st_inpat'].shape[0]
                d_v = tj_cond_list[0]['is_st_inpat'].device

                ## e.g., [B=40, h=56, d=2]
                templ_ov_tj = tj_cond_list[0]['end_ovlp_traj']

                ## loop through each key
                for k_c in keys_tjc:
                    tmp_v_l = []
                    ## loop through each elem in list
                    for i_c in range(len_tjc):
                        if k_c in ['st_ovlp_is_drop', 'end_ovlp_is_drop']:
                            if tj_cond_list[i_c][k_c] == None:
                                tj_cond_list[i_c][k_c] = \
                                    torch.ones(size=(b_s,), dtype=torch.bool, device=d_v)
                                
                        if k_c in ['st_ovlp_traj', 'end_ovlp_traj']:
                            if tj_cond_list[i_c][k_c] == None:
                                ## use a placeholder with the same shape since this will be dropped anyway.
                                tj_cond_list[i_c][k_c] = torch.zeros_like(templ_ov_tj)
                                pass


                        tmp_v_l.append(tj_cond_list[i_c][k_c])
                        # print(f'{k_c=} {i_c=} {type(tj_cond_list[i_c][k_c])}')
                    
                    
                    tmp_v_l = torch.cat( tmp_v_l, dim=0 )

                    mg_dict[k_c] = tmp_v_l

                    # print(f'{k_c} {tmp_v_l.shape} {type(tmp_v_l)}') ## all tensor
                
                # pdb.set_trace()

                return mg_dict
            
            ## merge all the tj_cond
            tj_cond_mg = merge_tj_cond_list(tj_cond_cur_list)
            tj_cond_mg['do_cond'] = True
            ##
            ## NEW:
            ## concat all x_p into one batch and sample only one times
            x_p_list_cur_t = torch.cat(x_p_list_cur_t, dim=0)

            assert len(tj_cond_mg['st_ovlp_is_drop']) == len(x_p_list_cur_t)

            ## (n_comp*B,hzn)
            t_steps_mg = torch.cat([timesteps,]*n_comp, dim=0)

            # pdb.set_trace()

            if do_mcmc:
                x_p_list_cur_t = self.resample_same_t_mcmc( x_p_list_cur_t, tj_cond_mg, t_steps_mg )

            if self.use_ddim:
                x_p_list_cur_t = self.ddim_p_sample(x_p_list_cur_t, tj_cond_mg, t_steps_mg, self.ddim_eta, use_clipped_model_output=True)
            else:
                x_p_list_cur_t = self.p_sample(x_p_list_cur_t, tj_cond_mg, t_steps_mg)
            
            # # x_p_list[i_tj] = x_p_i
            # x_p_list_next_t[i_tj] = x_p_i

            ## should be (n_comp*B, H, dim) -> (n_comp, B, H, dim)
            x_p_list_cur_t = einops.rearrange(x_p_list_cur_t, 
                                              '(n_comp B) h d -> n_comp B h d', n_comp=n_comp)
            
            
            ## important: NEW assign the next_t to cur_t for denosing at the next timestep
            x_p_list = x_p_list_cur_t

            if return_diffusion:
                x_dfu_all.append([_ for _ in x_p_list])


        #### -----------
            
        
        ## Finished
        x_p_list[0] = apply_conditioning(x_p_list[0], {0:stgl_cond[0]}, 0)
        x_p_list[-1] = apply_conditioning(x_p_list[-1], {hzn-1:stgl_cond[hzn-1]}, 0)

        ## TODO: Dec 30, we can clamp the output to -1,1 as loc here

        if return_diffusion:
            ## _, a list of x_p_list
            return x_p_list, x_dfu_all
        else:
            return x_p_list
















    
    @torch.no_grad()
    def comp_pred_p_loop_n_GSC(self, ##  
                        shape, stgl_cond, n_comp, do_mcmc=False, verbose=True, return_diffusion=False):
        """
        Inpaint with Avg Values
        """
        """assume compose n trajectories"""
        assert n_comp >= 2 and not do_mcmc, 'mcmc might be bad in our case and not implemented for DDIM yet.'
        device = self.betas.device

        batch_size = shape[0]
        hzn = shape[1]

        x_p_list = [ torch.randn(shape, device=device) for _ in range(n_comp) ]

        x_dfu_all = [x_p_list,]

        assert len(stgl_cond[0]) == shape[0]

        ## --- NEW Dec 4 ---
        """1. change the ddim time_dix; 2. change the p_sample
        """
        if self.use_ddim:
            time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        else:
            time_idx = reversed(range(0, self.n_timesteps))
        ## -----------------

        from tqdm import tqdm
        for i_t in tqdm(time_idx):
            
            ## timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long) # old
            ## e.g., (B=10,H=384)
            timesteps = torch.full((batch_size, self.horizon), i_t, device=device, dtype=torch.long)

            ## x_p_list ## to be given to the denoiser at this step
            x_p_list_next_t = [None for _ in range(n_comp)] ## after denoise this step, aka., less noisy

            ## Do the Averaging First
            ## x_p_list[0].shape: Size([40, 40, 2])
            ## x_p_list[1][0][:self.len_ovlp_cd]
            ## x_p_list[0][0][-self.len_ovlp_cd:]
            # pdb.set_trace()
            x_p_list = self.avg_ovlp_chunk_GSC(x_p_list)
            # pdb.set_trace()

            ## iteratively denoise each sub traj
            for i_tj in range(n_comp):
                ## target traj
                x_p_i = x_p_list[i_tj]

                if i_tj == 0:
                    ## first one
                    x_p_i_plus_1 = x_p_list[i_tj+1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=timesteps[:,0], ## placeholder
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={0:stgl_cond[0]}
                        )
                    
                    # tj_cond_p_i['end_ovlp_is_drop'] = None
                    tj_cond_p_i['end_ovlp_is_drop'] = None
                    # pdb.set_trace() ## st_ovlp as well and is_inpat

                    ## only used in the half_fd duplicate for faster cls-free
                    tj_cond_p_i['do_cond'] = False
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    # x_p_list[i_tj] = x_p_i
                    ## x_p_list (old list): still save the samples
                    ## put the less noisy sample in the new list
                    x_p_list_next_t[i_tj] = x_p_i
                
                elif i_tj > 0 and i_tj < n_comp-1:
                    ## intermediate one
                    # x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    # _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    # x_p_i_plus_1 = x_p_list[ i_tj + 1 ]
                    # st_traj_i_plus_1, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    # x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                    #     x_et=x_p_i,
                    #     st_traj=end_traj_i_minus_1,
                    #     end_traj=st_traj_i_plus_1,
                    #     t_1d_st=timesteps[:,0], ## same noisy level
                    #     t_1d_end=timesteps[:,0], 
                    #     # is_rand=True,
                    #     t_type='0', # g_cond['t_type'], ## 0 
                    #     is_noisy=True,
                    #     stgl_cond={},
                    # )


                    ## Drop everything because for intermediate chunks,
                    ## we are using inpainting w/ uncond generation
                    tj_cond_p_i = dict(
                        st_ovlp_is_drop=None, end_ovlp_is_drop=None, 
                        is_st_inpat=torch.zeros_like(x_p_i[:,0,0]).to(torch.bool),
                        is_end_inpat=torch.zeros_like(x_p_i[:,0,0]).to(torch.bool),
                    )

                    tj_cond_p_i['do_cond'] = False

                    # pdb.set_trace()

                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    # x_p_list[i_tj] = x_p_i
                    x_p_list_next_t[i_tj] = x_p_i

                elif i_tj == n_comp - 1:
                    ## last one

                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _,  end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=timesteps[:,0],
                        t_1d_end=timesteps[:,0], ## placeholder
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={hzn-1:stgl_cond[hzn-1]}
                        )
                    

                    tj_cond_p_i['st_ovlp_is_drop'] = None
                    
                    # tj_cond_p_i['do_cond'] = True
                    tj_cond_p_i['do_cond'] = False

                    # pdb.set_trace() ## TODO: Jan 24 Check Back all tj_cond

                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc( x_p_i, tj_cond_p_i, timesteps )

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    # x_p_list[i_tj] = x_p_i
                    x_p_list_next_t[i_tj] = x_p_i
            
            ## important: NEW assign the next_t to cur_t for denosing at the next timestep
            x_p_list = x_p_list_next_t

            if return_diffusion:
                x_dfu_all.append([_ for _ in x_p_list])


        #### -----------
            
        
        ## Finished
        x_p_list[0] = apply_conditioning(x_p_list[0], {0:stgl_cond[0]}, 0)
        x_p_list[-1] = apply_conditioning(x_p_list[-1], {hzn-1:stgl_cond[hzn-1]}, 0)

        ## TODO: Dec 30, we can clamp the output to -1,1 as loc here

        if return_diffusion:
            ## _, a list of x_p_list
            return x_p_list, x_dfu_all
        else:
            return x_p_list
        
    def avg_ovlp_chunk_GSC(self, x_p_list):
        """
        given a list of (noisy) trajs [(B,H,D), ...]
        """
        # pdb.set_trace()

        assert self.horizon / self.len_ovlp_cd >= 2, 'otherwise contain overlap.'
        # return a new list
        x_p_list = [x_p.clone() for x_p in x_p_list]
        if len(x_p_list) <= 2:
            pass
        else:
            # for i_tj in range(1, len(x_p_list)-1):
            for i_tj in range( 1, len(x_p_list) ):
                ## x_0
                ##   x_1
                ##     x_2
                ##       x_3
                ##          x_4
                
                ## x_p_list[i_tj-1].shape torch.Size([B=40, H=40, D=2])
                ## x_p_list[1][0][:self.len_ovlp_cd]
                # pdb.set_trace()
                ## do avg to the tail part of the prev traj and front part of cur traj
                ## 
                tmp_tj_prev = x_p_list[i_tj-1][:, -self.len_ovlp_cd:]
                tmp_tj_cur = x_p_list[i_tj][:, :self.len_ovlp_cd]


                tmp_avg = (tmp_tj_prev + tmp_tj_cur) / 2
                
                x_p_list[i_tj-1][:, -self.len_ovlp_cd:] = tmp_avg
                x_p_list[i_tj][:, :self.len_ovlp_cd] = tmp_avg

                # print(f'{i_tj-1=}, {i_tj=}, {tmp_avg.shape=}')
                # pdb.set_trace()
        
        return x_p_list





    ### Feb 14
    @torch.no_grad()
    def comp_pred_p_loop_n_ar_backward(self, ##  
                        shape, stgl_cond, n_comp, do_mcmc=False, verbose=True, return_diffusion=False):
        # assert False, 'not started yet, just copy from our original method.'
        """assume compose n trajectories"""
        assert n_comp >= 2 and not do_mcmc, 'mcmc might be bad in our case and not implemented for DDIM yet.'
        device = self.betas.device

        batch_size = shape[0]
        hzn = shape[1]

        x_p_list = [ torch.randn(shape, device=device) for _ in range(n_comp) ]

        x_dfu_all = [x_p_list,]

        assert len(stgl_cond[0]) == shape[0]

        ## --- NEW Dec 4 ---
        """1. change the ddim time_dix; 2. change the p_sample
        """
        if self.use_ddim:
            time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        else:
            time_idx = reversed(range(0, self.n_timesteps))
        ## -----------------

        from tqdm import tqdm
        for i_t in tqdm(time_idx):
            
            ## timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long) # old
            ## e.g., (B=10,H=384)
            timesteps = torch.full((batch_size, self.horizon), i_t, device=device, dtype=torch.long)

            ## iteratively denoise each sub traj, in backward order, [n_comp-1, 0]
            for i_tj in range(n_comp-1, -1, -1):

                # print(f'{i_t=} {i_tj=}')
                # pdb.set_trace()

                ## target traj
                x_p_i = x_p_list[i_tj]

                if i_tj == 0:
                    ## first one
                    x_p_i_plus_1 = x_p_list[i_tj+1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=timesteps[:,0],
                        t_1d_end=timesteps[:,0]-1, 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={0:stgl_cond[0]}
                        )
                    
                    tj_cond_p_i['do_cond'] = True
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    x_p_list[i_tj] = x_p_i
                
                elif i_tj > 0 and i_tj < n_comp-1:
                    ## intermediate one
                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i_plus_1 = x_p_list[ i_tj + 1 ]
                    st_traj_i_plus_1, _ = self.extract_ovlp_from_full(x_p_i_plus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=st_traj_i_plus_1,
                        t_1d_st=timesteps[:,0],
                        t_1d_end=timesteps[:,0]-1, 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={},
                    )

                    tj_cond_p_i['do_cond'] = True

                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    x_p_list[i_tj] = x_p_i



                elif i_tj == n_comp - 1:
                    ## last one

                    x_p_i_minus_1 = x_p_list[ i_tj - 1 ]
                    _,  end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)

                    x_p_i, tj_cond_p_i = self.create_eval_tj_cond(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=timesteps[:,0], ## same t
                        t_1d_end=timesteps[:,0], 
                        # is_rand=True,
                        t_type='0', # g_cond['t_type'], ## 0 
                        is_noisy=True,
                        stgl_cond={hzn-1:stgl_cond[hzn-1]}
                        )
                    tj_cond_p_i['do_cond'] = True

                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc( x_p_i, tj_cond_p_i, timesteps )

                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj_cond_p_i, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    
                    x_p_list[i_tj] = x_p_i


            
            if return_diffusion:
                x_dfu_all.append([_ for _ in x_p_list])


        #### -----------
            
        
        ## Finished
        x_p_list[0] = apply_conditioning(x_p_list[0], {0:stgl_cond[0]}, 0)
        x_p_list[-1] = apply_conditioning(x_p_list[-1], {hzn-1:stgl_cond[hzn-1]}, 0)

        ## TODO: Dec 30, we can clamp the output to -1,1 as loc here

        if return_diffusion:
            ## _, a list of x_p_list
            return x_p_list, x_dfu_all
        else:
            return x_p_list