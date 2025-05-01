import numpy as np
import torch, einops, pdb, time
import diffuser.utils as utils
from diffuser.guides.comp.cd_sml_policies import Trajectories_invdyn

from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.models.helpers import apply_conditioning
from diffuser.guides.comp.traj_blender import Traj_Blender
from diffuser.utils.cp_utils.plan_utils import split_trajs_list_by_prob
        

class Stgl_Sml_Policy_V1:

    def __init__(self, diffusion_model, 
                 normalizer, 
                 pol_config,
                 tj_blder_config,
                 ):
        """
        pick_type: how to pick from top_n
        """
        self.diffusion_model: Stgl_Sml_GauDiffusion_InvDyn_V1 = diffusion_model
        self.diffusion_model.eval() ## NOTE: must be the ema one
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        
        self.n_comp = pol_config['ev_n_comp']
        self.top_n = pol_config['ev_top_n'] ## 5
        self.pick_type = pol_config['ev_pick_type'] ## thresholding is also fine?
        assert self.pick_type in ['first', 'rand'] ## smallest dist or randomly pick one
        
        self.cp_infer_t_type = pol_config.get('ev_cp_infer_t_type', 'interleave')

        ## blender_type, exp_beta
        self.tj_blder = Traj_Blender(diffusion_model, normalizer, **tj_blder_config)
        self.ncp_pred_time_list = [] ## a list of tuple [n_comp, sampling_time]

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device
    



    def gen_cond_stgl_parallel(self, g_cond, debug=False, b_s=1):
        """
        st_gl: *not normed*, np2d [2, ndim], e.g., [ [st], [end] ], [[2,1], [3,4]],
        b_s: batch_size, 10-20+
        """
        
        hzn = self.diffusion_model.horizon
        o_dim = self.diffusion_model.observation_dim ## TODO: obs_dim only?
        # c_shape = [b_s, hzn, o_dim] ## e.g.,(20,160,2)
        

        st_gl = g_cond['st_gl']
        st_gl = torch.tensor(self.normalizer.normalize(st_gl, 'observations'))
        
        
        ## shape: 2, n_probs, dim
        assert st_gl.ndim == 3 and st_gl.shape[0] == 2
        n_probs = st_gl.shape[1]
        
        c_shape = [b_s*n_probs, hzn, o_dim] ## e.g.,(b_s*n_p: 20*10=200,160,2)

        ## 0: tensor (n_parallel_probs,2); hzn-1: same
        ## make sure return is not a view
        stgl_cond = {
            0: einops.repeat(st_gl[0,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
            hzn-1: einops.repeat(st_gl[1,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
        }
        # pdb.set_trace() ## check if repeat is correct




        ## Run GPU Planning, x_dfu_all
        ## a list of len n_comp, elem: cuda tensor (b_s*n_p=200 or 400,sm_hzn,dim)
        if self.cp_infer_t_type == 'interleave': ## original our
            trajs_list_lg_b = self.diffusion_model.comp_pred_p_loop_n(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'gsc': ## Jan 23
            trajs_list_lg_b = self.diffusion_model.comp_pred_p_loop_n_GSC(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        else:
            raise NotImplementedError

        ## reshape to a list of len n_probs, each elem is a trajs_list

        ## a list of trajs_list(list) for each prob
        trajs_list_acc = split_trajs_list_by_prob(trajs_list_lg_b, n_probs)
        
        out_list = [] ## store output for each problems
        pick_traj_acc = []


        for i_pb in range(n_probs):
            trajs_list = trajs_list_acc[i_pb]

            ## traj picking and merging
            ## get unnormed numpy list, same format
            trajs_list_np_un = utils.get_np_trajs_list(trajs_list, do_unnorm=True, 
                                                    normalizer=self.normalizer)
            ## ranking of all the traj candiates based on the distance of ovlp parts
            s_idxs, dist_per_sam = utils.compute_ovlp_dist(trajs_list_np_un, 
                                                        self.diffusion_model.len_ovlp_cd)
            
            ## list, pick out the topn, from un-normed traj
            trajs_list_topn_np_un = utils.pick_top_n_trajs(trajs_list_np_un, s_idxs, self.top_n)
            ## np, un-normed, shape (B, tot_hzn, dim)
            trajs_list_topn_bl = self.tj_blder.blend_traj_lists(trajs_list_topn_np_un, do_unnorm=False)

            ## pick one traj to execute
            if self.pick_type == 'first':
                pick_traj = trajs_list_topn_bl[0]
            elif self.pick_type == 'rand':
                p_idx = np.random.randint(low=0, high=self.top_n)
                pick_traj = trajs_list_topn_bl[p_idx]
            else:
                raise NotImplementedError

            out = Stgl_Sml_Ev_Pred(pick_traj, trajs_list_topn_bl, trajs_list_np_un)

            out_list.append(out)
            pick_traj_acc.append(out.pick_traj)

        ##
        # pdb.set_trace()
        ## return a list of out and pick_traj
        return out_list, pick_traj_acc







    

    ### -------- Oct 20 For Stgl Quan Planning ------
    def gen_cond_stgl(self, g_cond, debug=False, b_s=1):
        """
        Jan 21: Default Version that Support Replan
        st_gl: *not normed*, np2d [2, ndim], e.g., [ [st], [end] ], [[2,1], [3,4]],
        b_s: batch_size, 10-20+
        """
        if self.n_comp == 1:
            return self.gen_1_cond_stgl(g_cond, b_s=b_s)
        
        hzn = self.diffusion_model.horizon
        o_dim = self.diffusion_model.observation_dim ## TODO: obs_dim only?
        c_shape = [b_s, hzn, o_dim] ## e.g.,(20,160,2)
        
        # pdb.set_trace() ## check format

        st_gl = g_cond['st_gl']
        st_gl = torch.tensor(self.normalizer.normalize(st_gl, 'observations'))
        
        ## shape: 2, n_probs, dim
        assert st_gl.ndim == 3 and st_gl.shape[0] == 2
        
        ## make sure return is not a view
        stgl_cond = {
            0: einops.repeat(st_gl[0,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
            hzn-1: einops.repeat(st_gl[1,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
        }

        cur_time = time.time()

        ## Run GPU Planning, x_dfu_all
        ## a list of len n_comp, elem: cuda tensor (B,sm_hzn,dim)
        if self.cp_infer_t_type == 'interleave': ## original our
            trajs_list = self.diffusion_model.comp_pred_p_loop_n(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'same_t': ## Same t denoising, but not parallel
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_same_t(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
            
        elif self.cp_infer_t_type == 'gsc': ## baseline
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_GSC(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'same_t_p': ## Same t denoising and *parallel*
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_same_t_parallel(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        
        elif self.cp_infer_t_type == 'ar_back': ## backward autoregressive denosing
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_ar_backward(
                c_shape, stgl_cond, n_comp=self.n_comp, return_diffusion=False)
        else:
            raise NotImplementedError
        
        self.ncp_pred_time_list.append( [self.n_comp,  time.time() - cur_time] ) ## unit: sec


        
        ## note that we can return a lof of stuff
        ## get unnormed numpy list, same format
        trajs_list_np_un = utils.get_np_trajs_list(trajs_list, do_unnorm=True, 
                                                   normalizer=self.normalizer)
        ## ranking of all the traj candiates based on the distance of ovlp parts
        s_idxs, dist_per_sam = utils.compute_ovlp_dist(trajs_list_np_un, 
                                                       self.diffusion_model.len_ovlp_cd)
        
        ## list, pick out the topn, from un-normed traj
        trajs_list_topn_np_un = utils.pick_top_n_trajs(trajs_list_np_un, s_idxs, self.top_n)
        ## np, un-normed, shape (B, tot_hzn, dim)
        trajs_list_topn_bl = self.tj_blder.blend_traj_lists(trajs_list_topn_np_un, do_unnorm=False)

        ## pick one traj to execute
        if self.pick_type == 'first':
            pick_traj = trajs_list_topn_bl[0]
        elif self.pick_type == 'rand':
            p_idx = np.random.randint(low=0, high=self.top_n)
            pick_traj = trajs_list_topn_bl[p_idx]
        else:
            raise NotImplementedError

        out = Stgl_Sml_Ev_Pred(pick_traj, trajs_list_topn_bl, trajs_list_np_un)

        return out
    


    def gen_1_cond_stgl(self, g_cond, debug=False, b_s=1):
        """
        use start goal as condition, do not compose model, just like vanilla DD

        st_gl: *not normed*, np2d [2, ndim], e.g., [ [st], [end] ], [[2,1], [3,4]],
        b_s: batch_size, 10-20+
        """
        
        hzn = self.diffusion_model.horizon
        o_dim = self.diffusion_model.observation_dim
        c_shape = [b_s, hzn, o_dim] ## e.g.,(20,160,2)
        
        # pdb.set_trace() ## check format

        st_gl = g_cond['st_gl']
        st_gl = torch.tensor(self.normalizer.normalize(st_gl, 'observations'))
        
        n_probs = st_gl.shape[1]
        ## shape: 2, n_probs, dim, now only support planning for one problem
        assert st_gl.ndim == 3 and st_gl.shape[0] == 2 and n_probs == 1

        # pdb.set_trace()

        stgl_cond = {
            0: einops.repeat(st_gl[0,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
            hzn-1: einops.repeat(st_gl[1,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
        }

        g_cond_1 = {}
        g_cond_1['do_cond'] = 'both_stgl'
        g_cond_1['stgl_cond'] = stgl_cond
        ## uesless placeholder
        g_cond_1['t_type'] = '0'
        g_cond_1['traj_full'] = np.random.random( size=(stgl_cond[0].shape[0],) )

        pred_trajs = self.diffusion_model.conditional_sample(g_cond=g_cond_1)

        pred_trajs = apply_conditioning(pred_trajs, stgl_cond, 0)



        pred_trajs = utils.to_np(pred_trajs)
        pred_trajs_un = self.normalizer.unnormalize(pred_trajs, 'observations')

        out = Stgl_Sml_Ev_Pred(pred_trajs_un[0], 
                               pred_trajs_un, 
                               pred_trajs_un)
        ##
        # pdb.set_trace()

        return out





    




    def _format_g_cond(self, g_cond, batch_size):
        
        traj_f = g_cond['traj_full'] 
        ## normalize the traj
        traj_f =  self.normalizer.normalize(traj_f, 'observations')
        traj_f = torch.tensor(traj_f, dtype=torch.float32, device='cuda:0')

        traj_f = einops.repeat(traj_f, 'b h d -> (repeat b) h d', repeat=batch_size)
        
        g_cond['traj_full'] = traj_f
        
        return g_cond


    def gen_cond(self, g_cond, debug=False, batch_size=1):
        '''conditional sampling
        conditioned on start and end chunks, just for sanity test
        '''

        g_cond = self._format_g_cond(g_cond, batch_size)

        
        sample = self.diffusion_model.conditional_sample(g_cond)

        
        actions = np.zeros(shape=(*sample.shape[0:2], self.action_dim))
        sample = utils.to_np(sample)
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)
        
        ## extract first action
        action = actions[0, 0]

        # pdb.set_trace()

        # if debug:
        normed_observations = sample[:, :, 0:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories_invdyn(actions, observations)
        return action, trajectories
        




    

class Stgl_Sml_Ev_Pred:
    def __init__(self, pick_traj, 
                 trajs_list_topn_bl, trajs_list_np_un) -> None:
        '''
        pick_traj: np2d unnormed (tot_hzn,dim), the traj to follow
        trajs_list_topn_bl: np3d unnormed (B,tot_hzn,dim), all topn


        '''
        self.pick_traj = pick_traj
        self.trajs_list_topn_bl = trajs_list_topn_bl
        self.trajs_list_np_un = trajs_list_np_un