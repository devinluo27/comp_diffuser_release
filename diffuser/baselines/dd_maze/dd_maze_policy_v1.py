import numpy as np
import torch, einops, pdb, time
import diffuser.utils as utils
from diffuser.guides.comp.cd_sml_policies import Trajectories_invdyn

# from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.baselines.dd_maze.dd_maze_diffusion_v1 import DD_Maze_GauDiffusion_InvDyn_V1
from diffuser.models.helpers import apply_conditioning
from diffuser.models.cd_stgl_sml_dfu.stgl_sml_policy_v1 import Stgl_Sml_Ev_Pred


class DD_Maze_Policy_V1:

    def __init__(self, diffusion_model, 
                 normalizer, 
                 pol_config,
                 ):
        """
        pick_type: how to pick from top_n
        """
        self.diffusion_model: DD_Maze_GauDiffusion_InvDyn_V1 = diffusion_model
        self.diffusion_model.eval() ## NOTE: must be the ema one
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        self.pl_hzn = pol_config.get('ev_pl_hzn', diffusion_model.horizon,)
        ## if not set,  cause error 
        self.diffusion_model.horizon = self.pl_hzn
        # pdb.set_trace()
        self.ncp_pred_time_list = []
        self.return_diffusion = False
        
        

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
        
        # pdb.set_trace() ## Oct 24 TODO: From Here, check the format of repeat
        ## shape: 2, n_probs, dim
        assert st_gl.ndim == 3 and st_gl.shape[0] == 2
        n_probs = st_gl.shape[1]

        ## NOTE: in Decision Diffuser Baseline, ignore the input b_s
        b_s = 1


        ## TODO: Oct 25: 14:08 pm check c_shape
        # c_shape = [b_s*n_probs, hzn, o_dim] ## e.g.,(b_s*n_p: 20*10=200,160,2)

        ## 0: tensor (n_parallel_probs,2); hzn-1: same
        ## make sure return is not a view
        stgl_cond = {
            0: einops.repeat(st_gl[0,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
            hzn-1: einops.repeat(st_gl[1,:,:], 'n_p d -> (n_p rr) d', rr=b_s).clone(),
        }

        # pdb.set_trace() ## check if repeat is correct

        g_cond = dict(stgl_cond=stgl_cond)

        sample = self.diffusion_model.conditional_sample(g_cond, horizon=self.pl_hzn, verbose=False,
                                                         return_diffusion=self.return_diffusion)
        if self.return_diffusion:
            self.dfu_proc_all = sample[1]
            sample = sample[0]

        ## NOTE: NEW inpaint start and goal again
        sample = apply_conditioning(sample, stgl_cond, 0) # start from dim 0, different from diffuser

        sample = utils.to_np(sample)

        # pdb.set_trace()

        normed_observations = sample[:, :, 0:] ## act_dim
        pred_obs_trajs_un = self.normalizer.unnormalize(normed_observations, 'observations')

        



        
        
        out_list = [] ## store output for each problems
        pick_traj_acc = []

        # pdb.set_trace()
        assert len(pred_obs_trajs_un) == n_probs

        for i_pb in range(n_probs):

            ######
            ## an unnormed H,D np traj
            pick_traj = pred_obs_trajs_un[i_pb]
            out = Stgl_Sml_Ev_Pred(pick_traj, None, None)

            out_list.append(out)
            pick_traj_acc.append(out.pick_traj)



        ##
        # pdb.set_trace()
        ## return a list of out and pick_traj
        return out_list, pick_traj_acc





    def gen_cond_stgl(self, g_cond, debug=False, b_s=1):

        cur_time = time.time()
        out_list, pick_traj_acc = self.gen_cond_stgl_parallel(g_cond, debug, b_s,)
        
        # pdb.set_trace() ## TODO: Feb 15 20:20 start from test time the time on L40s

        assert len(out_list) == 1

        self.ncp_pred_time_list.append( [1,  time.time() - cur_time] ) ## unit: sec
        
        return out_list[0]

    


