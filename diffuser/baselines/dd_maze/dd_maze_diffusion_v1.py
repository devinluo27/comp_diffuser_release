# assert False, 'Not Finished.'

import numpy as np
import torch
from torch import nn
import pdb
import torch.nn.functional as F

import diffuser.utils as utils
from diffuser.models.helpers import (
    cosine_beta_schedule, extract_2d, apply_conditioning, tensor_randint, Losses,
)
from diffuser.models.hi_helpers import MLP_InvDyn
from diffuser.models.cd_stgl_sml_dfu import Unet1D_TjTi_Stgl_Cond_V1
from diffuser.models.comp_dfu.comp_diffusion_v1 import ModelPrediction

class DD_Maze_GauDiffusion_InvDyn_V1(nn.Module):
    """
    This file contains the Decision Diffuser baseline used in the CompDiffuser paper.
    """
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

        self.is_inv_dyn_dfu = True
        self.dfu_name = 'dd_maze'

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
        self.var_temp = 1.0

        ## ----- Dec 3, for DDIM -----
        ddim_set_alpha_to_one = self.diff_config.get('ddim_set_alpha_to_one', True)
        self.final_alpha_cumprod = torch.tensor([1.0,], ) \
            if ddim_set_alpha_to_one else torch.clone(self.alphas_cumprod[0:1]) # tensor of size (1,)
        self.num_train_timesteps = self.n_timesteps
        self.ddim_num_inference_steps = self.diff_config.get('ddim_steps', 50)
        ## Hyper-parameters
        self.ddim_eta = 1.0 
        self.use_ddim = True
        ## ----------------------------
        
        
        


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
        # pdb.set_trace()

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

        
        if False:
            ## NOTE: we do not need classifier-free guidance in DD for maze setup
            
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
            ## already doing inpainting to x
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
        
        # no noise when t == 0, shape: B,H,1
        nonzero_mask = (1 - (timesteps == 0).float()).reshape( b, self.horizon, *((1,) * (len(x.shape) - 2)) )
        ## check model_log_variance
        # pdb.set_trace() # check shape
        x_t_minus_1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
        if mask_same_t is not None:
            assert False, 'no bug, not used'
        else:
            ## normal diffusion
            return x_t_minus_1

    
    




    

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
            
            # pdb.set_trace() ## check out what is inside g_cond

            # x, tj_cond = self.get_tj_cond(x, g_cond, timesteps)

            x = apply_conditioning(x, g_cond['stgl_cond'], 0)
            tj_cond = {}
            

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
        # pdb.set_trace()

        device = self.betas.device
        batch_size = len(g_cond['stgl_cond'][0]) ## TODO: check

        

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
            assert False, 'not used'
            ## (B H C)[(B,H)]
            assert mask_no_noise.shape == x_start.shape[:2] and mask_no_noise.dtype == torch.bool
            ## TODO: why the sample is so large? so :2 larger than -1? solved, due to luotest subset
            # pdb.set_trace()
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

        ## check loss weight for inpaint part!
        # pdb.set_trace()

        if self.w_loss_type == 'no_unused':
            assert False
            batch_loss_w = batch_loss_w[:, :, None] ## B,H,1
        elif self.w_loss_type == 'all':
            ## important: set the loss of the start state and last state to  0
            # batch_loss_w = 1.
            ## B,H,1
            batch_loss_w = torch.ones_like(x_start[:, :, :1])
            ## (B,), no need loss for the inpainting state
            batch_loss_w[:, 0] = 0.
            batch_loss_w[:, self.horizon-1] = 0.
            # pdb.set_trace()
            assert x_start.shape[1] == self.horizon
            

        else:
            raise NotImplementedError()
        
        ## x_recon might be the predicited x0 (default) or the predicted noise 
        x_recon = self.small_model_pred(x_noisy, t_2d, tj_cond)

        assert x_noisy.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, ext_loss_w=batch_loss_w)
        else:
            loss, info = self.loss_fn(x_recon, x_start, ext_loss_w=batch_loss_w)

        return loss, info
    
    



    



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

            ## we generate x_noisy outside!
            # pdb.set_trace()
            x_noisy = apply_conditioning(x_noisy, cond_st_gl, 0)
            tj_cond = {}

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

        stgl_cond = g_cond['stgl_cond']
        # pdb.set_trace()

        x = apply_conditioning(x, stgl_cond, 0) # start from dim 0, different from diffuser

        if return_diffusion: diffusion = [x]
        # 100 // 20 = 5
        ## e.g., array([459, 408, 357, 306, 255, 204, 153, 102,  51,   0])
        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)

        # for i in time_idx: # if np array, i is <class 'numpy.int64'>
        from tqdm import tqdm
        for i in tqdm(time_idx):
            
            
            timesteps = torch.full((batch_size, self.horizon), i, device=device, dtype=torch.long)
            
            ## this get func is not for composing
            # x, tj_cond = self.get_tj_cond(x, g_cond, timesteps)

            # pdb.set_trace()

            x = apply_conditioning(x, stgl_cond, 0) # start from dim 0, different from diffuser
            tj_cond = {}

            ## From Here Dec 3, 16:04

            ## ----------------------
            ### eta=0.0
            x = self.ddim_p_sample(x, tj_cond, timesteps, self.ddim_eta, use_clipped_model_output=True)
            # x = apply_conditioning(x, cond, 0)

            if return_diffusion: 
                x = apply_conditioning(x, stgl_cond, 0)
                diffusion.append(x)
        

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


    
