import numpy as np
from os.path import join
from diffuser.models.cd_stgl_sml_dfu.stgl_sml_policy_v1 import Stgl_Sml_Policy_V1
from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.models.helpers import apply_conditioning
import diffuser.datasets as datasets
import diffuser.utils as utils
from datetime import datetime
import os.path as osp
import copy, pdb, json, pdb, torch, os, mujoco, math, socket
from diffuser.guides.render_m2d import Maze2dRenderer_V2
from collections import OrderedDict
from diffuser.datasets.d4rl import Is_Gym_Robot_Env, Is_OgB_Robot_Env
from diffuser.datasets.ogb_dset.ogb_utils import ogb_load_env, ogb_xy_to_ij, \
        ogb_get_rowcol_obs_trajs_from_xy, ogb_get_rowcol_obs_trajs_from_xy_list, ogb_load_env_kwargs


class OgB_Stgl_Sml_MazeEnvPlanner_V1:
    """
    A Class to support evaluation using CompDiffuser.
    This class can be used for OGBench Maze and AntSoccer
    A high level planner that:
        1. load the model out
        2. loop through all the required episodes.
        3. summarize the metrics results
    """
    def __init__(self, args_train, args) -> None:
        self.args_train = args_train
        self.args = args

        self.plan_n_ep = args.plan_n_ep
        self.b_size_per_prob = args.b_size_per_prob

        
        self.n_batch_acc_probs = args.n_batch_acc_probs
        self.is_replan = args.is_replan
        self.repl_wp_cfg = args.repl_wp_cfg
        self.repl_ada_dist_cfg = getattr(self.args, 'repl_ada_dist_cfg', {})
        ## might enable inv dyn dropout in soccer env
        self.is_inv_train_mode = getattr(self.args, 'is_inv_train_mode', False)
        self.ep_st_idx = getattr(self.args, 'ep_st_idx', 0)
        
        self.is_save_pkl = getattr(self.args, 'is_save_pkl', False)
        self.rd_resol = getattr(self.args, 'rd_resol', 200)
        self.is_use_subgoal_marker = getattr(self.args, 'is_use_subgoal_marker', True)
        self.is_rd_agv = getattr(self.args, 'is_rd_agv', False)

        if self.plan_n_ep == 100:
            assert self.ep_st_idx == 0

        self.vis_trajs_per_img = 10
        self.score_low_limit = 100
        # np.set_printoptions(precision=3, suppress=True)
        self.act_control = 'pred_inv' # 'dfu_force'
        self.n_act_per_waypnt = args.n_act_per_waypnt
        # self.eval_cfg = eval_cfg
        tmp_fps = getattr(self.args, 'vid_fps', False)
        if tmp_fps:
            pass ## just use the given one
        elif 'soccer' in self.args.dataset:
            tmp_fps = 80
        else:
            tmp_fps = 120

        self.vid_fps = tmp_fps * self.n_act_per_waypnt ## ori=200
        self.is_soccer_task = False
        ## some hyperparam might be updated in the later code
        if 'antmaze' in args_train.dataset:
            self.extra_env_steps = 150
        elif 'humanoidmaze' in args_train.dataset:
            self.extra_env_steps = 400 # 300
        elif 'antsoccer' in args_train.dataset:
            self.extra_env_steps = 300 # 300
            self.is_soccer_task = True
        elif 'pointmaze' in args_train.dataset:
            self.extra_env_steps = 50 ## Jan 12
            self.act_control = 'pd_ogb' # 'dfu_force'
        else:
            raise NotImplementedError


    def setup_load(self, ld_config):
        """
        used in a separate launch evaluation, where model should be loaded from file
        """
        args_train = self.args_train
        args = self.args

        

        if self.args.env_n_max_steps is not None: # No Need
            ## before set, need to check if ogb use this variable
            assert self.is_replan == 'ada_dist'
        
        dfu_exp = utils.load_stgl_sml_diffusion(
            args.logbase, args_train.dataset, 
            args_train.exp_name, epoch=args.diffusion_epoch,
            ld_config=ld_config)


        ## check if custom normalizer load is ok.
        ## add non-load normalizer!!!
        ## the normalization value is hard-coded according to dataset.
        self.train_normalizer = utils.load_ogb_maze_datasetNormalizer(args_train,)
        ## for inv dyn model
        self.full_normalizer = utils.load_ogb_maze_datasetNormalizer(args_train, obs_dim_idxs='full')

        self.check_is_same_nmlizer()
        # pdb.set_trace() ## check normalizer 


        self.diffusion: Stgl_Sml_GauDiffusion_InvDyn_V1 = dfu_exp.ema ## should an ema model
        self.diffusion.var_temp = args.var_temp
        self.diffusion.condition_guidance_w = args.cond_w
        ## NEW DDIM hyper-param
        self.diffusion.use_ddim = args.use_ddim
        self.diffusion.ddim_eta = args.ddim_eta
        self.diffusion.ddim_num_inference_steps = args.ddim_steps


        self.dataset = dfu_exp.dataset
        # self.train_normalizer = self.dataset.normalizer
        self.renderer: Maze2dRenderer_V2
        self.renderer = dfu_exp.renderer
        self.trainer = dfu_exp.trainer
        
        if self.diffusion.is_inv_dyn_dfu:
            ## NOTE: to support DD baseline...
            dfu_name =  getattr(self.diffusion, 'dfu_name', 'our_stgl_sml')

            if dfu_name == 'dd_maze':
                # pdb.set_trace()
                
                ## Jan 18 For the Decision Diffuser Baseline
                self.pol_config = {}
                self.tj_blder_config = {}
                for k in args.__dict__.keys():
                    if 'ev_' in k:
                        self.pol_config[k] = args.__dict__[k]
                
                from diffuser.baselines.dd_maze.dd_maze_policy_v1 import DD_Maze_Policy_V1
                self.policy = DD_Maze_Policy_V1(self.diffusion, self.train_normalizer,
                                                 self.pol_config)

            else:
                ## ---------- Our Method ----------
                ## top_n: int, pick_type: str,tj_blder_config,
                ## Setup the config to init policy
                self.pol_config = {}

                for k in args.__dict__.keys():
                    if 'ev_' in k:
                        self.pol_config[k] = args.__dict__[k]
                
                self.tj_blder_config = dict(blend_type=args.tjb_blend_type,
                                    exp_beta=args.tjb_exp_beta,)

                self.policy = Stgl_Sml_Policy_V1(self.diffusion, self.train_normalizer, 
                                                self.pol_config, self.tj_blder_config)
                
                # pdb.set_trace()

        else:
            assert False, 'not implemented'
        
        self.savepath = args.savepath
        self.epoch = dfu_exp.epoch

        utils.print_color(f'Load From {self.epoch=}', c='y')

        ## create the env instance for the actual rollout
        self.create_env()

        self.setup_general()
        self.dataset_config = args_train.dataset_config
        self.obs_select_dim = self.dataset_config['obs_select_dim']
        self.dfu_ndim = len(self.obs_select_dim)
        self.dset_type = self.dataset_config['dset_type'] # should be 'ogb', not 'ours'
        self.ld_config = ld_config ## load_config
        self.load_inv_model()


        # self.diffusion.update_eval_config(eval_cfg=args.as_dict())



    
    def create_env(self):
        if hasattr(self, 'env'):
            del self.env
        if Is_OgB_Robot_Env:
            # self.env = ogb_load_env(args.dataset)
            self.env = ogb_load_env_kwargs(self.args.dataset, 
                                           height=self.rd_resol, width=self.rd_resol)
            
            self.env.set_seed_addn(0) ## So st/gl are the same across different runs
        else:
            raise NotImplementedError
        utils.print_color(f'[setup_load] {self.env.max_episode_steps=}')



    def setup_general(self):
        '''general stuff for both types of setup'''
        utils.mkdir(self.savepath)
        self.savepath_root = self.savepath
        ## We use pre-sampled problems so can do parallel planning
        self.load_ev_problems()


    def load_inv_model(self):
        """load the inv dyn model out"""
        if hasattr(self.args, 'inv_model_path'):
            ## parse from outside
            self.inv_model_path = self.args.inv_model_path
        else:
            ## use default one
            self.inv_model_path = utils.ogb_get_inv_model_path(self.env.name, gl_dim=self.dfu_ndim)

        if 'pointmaze' in self.args.dataset:
            import types
            ## we do not use a inv dyn model for point maze
            self.inv_epoch = None
            self.inv_model = types.SimpleNamespace()
            self.inv_model.training = None
            utils.print_color(f'\n{self.args.dataset=}. *** No Inv Dyn Model *** \n', c='y')
            # pdb.set_trace()

        else:

            from diffuser.ogb_task.og_inv_dyn.og_invdyn_helpers import ogb_load_invdyn_maze_v1
            inv_model, inv_ema, inv_epoch = ogb_load_invdyn_maze_v1(self.inv_model_path, 
                                                                    epoch=self.args.inv_epoch)
            self.inv_epoch = inv_epoch
            ## self.diffusion.inv_model = inv_model
            self.inv_model = inv_model
            

            if self.is_inv_train_mode:
                self.inv_model.train()
                assert 'soccer' in self.args.dataset, 'temporary, can be removed'
            utils.print_color(f'\n{self.inv_model.training=}\n', c='y')
        
        # pdb.set_trace()



    def load_ev_problems(self):
        """
        load the pre-collected evaluation problems (start/goal) pair out
        """
        ## get the file name and load the dict out
        self.problems_h5path = utils.get_ogb_maze_ev_probs_fname(self.env.name)
        self.problems_dict = utils.load_stgl_lh_ev_probs_hdf5(h5path=self.problems_h5path)        
        
        return
    

    
    def ogb_env_get_obs(self) -> np.ndarray:
        obs = self.env.get_ob() ## info is empty
        return obs




    def ogb_env_interact_1_ep(self, pick_traj, start_state, target):
        """
        ** OGBench **
        Interact for one episode, OgB's Version
        pick_traj: the traj to follow, unnormed
        """
        ## make sure robot state is aligned with the outer loop
        # self.env: <ogbench.locomaze.maze.make_maze_env.<locals>.MazeEnv object at 0x7f671256b1c0>

        ## make sure pick_traj is correct
        assert np.isclose( start_state, self.env.get_ob() ).all()
        assert np.isclose( target, self.env.cur_goal_xy ).all()

        ## Oct 30 uncomment
        # assert np.isclose(pick_traj[0], start_state[ self.obs_select_dim, ], atol=0.05).all()
        # assert np.isclose(pick_traj[-1], target, atol=0.05).all()

        ## metrics to return
        is_suc = False
        total_reward = 0
        rollout = [start_state.copy(),]
        imgs_rout = [self.env.render(),]


        # n_max_steps = self.env.max_episode_steps
        self.n_max_steps = len(pick_traj) * self.n_act_per_waypnt + 30

        for i_et in range(self.n_max_steps):
            ## maze2d: np (4,)
            obs_cur = self.env.get_ob().copy()
            obs_cur_nm = self.full_normalizer.normalize(obs_cur, 'observations')
            
            assert len(obs_cur) in [4, 29, 69], '2d, ant, humanoid'

            ## ----------- Controller ------------
            
            ## can use actions or define a simple controller based on state predictions
            if self.act_control == 'pred_inv':
                wp_idx = i_et // self.n_act_per_waypnt
                wp_idx = min( wp_idx, len(pick_traj)-1 )
                
                # pdb.set_trace()

                ## put a virtual cylinder for vis
                goal_cur = pick_traj[ wp_idx ] # [:2]
                ## NOTE: train and full is the same normalizer but different dimension
                goal_cur_nm = self.train_normalizer.normalize(goal_cur, 'observations')

                self.env.set_subgoal_waypnt( goal_cur[:2] )

                ## automatically to gpu, after expand [1,obs_dim]
                obs_cur_nm = utils.to_torch(obs_cur_nm)[None,]
                goal_cur_nm = utils.to_torch(goal_cur_nm)[None,]
                act_pred = self.inv_model(obs_cur_nm, goal_cur_nm).cpu().numpy()[0]

                
            elif self.act_control == 'dfu_force': ## from diffusion forcing
                ## can copy from other planner later, remove for simplicity.
                raise NotImplementedError()

            
            ## ------------------------------------


            ## obs_cur: a vector, e.g., (29,)
            obs_cur, rew, terminated, truncated, info = self.env.step(act_pred)


            is_suc = bool(info['success']) or is_suc ## sparse reward

            total_reward += info['success']
            score = 0 # not used

            if i_et % 100 == 0:
            # if i_et == n_max_steps - 1: ## action: {act_pred}
                print(
                    f't: {i_et} | r: {rew:.2f} |  R: {total_reward:.2f} | '
                    f'pos: {obs_cur[:2]} | '
                    f'Max Steps: {self.n_max_steps}'
                )

            ## update rollout observations
            rollout.append(obs_cur.copy())
            imgs_rout.append(self.env.render())


            
        

        ## -----------------------------------------------------------
        ## ------------ Finished one env interact episode ------------
        # pdb.set_trace()
        imgs_rout = np.array(imgs_rout) ## a list of 200,200,3
        rollout = np.array(rollout) ## a list of 29/69

        return is_suc, total_reward, score, rollout, imgs_rout

        






        

    ## ---------------------------------------
    ## ---------------------------------------
    ## Old Slow version, but can support replanning because we plan one traj each time

    def ogb_plan_once(self, pl_seed=None, given_probs=None): ## num_ep
        '''
        NOTE this func is proabbly for replan only, please refer to 
        'trash/backup/2024_Jan3_ogb_stgl_sml_planner_v1.py' for no replan version
        code to launch planning
        given_probs: dict to specify start/goal, len is ??
        '''
        assert Is_OgB_Robot_Env
        n_comp_full = self.pol_config['ev_n_comp'] ## full num of composed trajs

        if pl_seed is not None:
            utils.set_seed(pl_seed) ## seed everything
        if given_probs is not None: # if given, just evaluate the given states
            num_ep = len(given_probs)
        else:
            num_probs = len(self.problems_dict['start_state'])
            num_ep = num_probs if self.plan_n_ep == -100 else self.plan_n_ep
        
        utils.print_color(f'[ogb_plan_once]: {num_ep=}')

        ep_scores = []
        ep_total_rewards = []
        ep_pred_obss = [] ## just the 2d agent xy 
        ep_pred_obss_full = [] ## full pred obs state
        ep_rollouts = []
        ep_rollouts_full = []
        ep_targets = []
        ep_is_suc = [] ## a list of bool
        ep_titles_obs, ep_titles_act = [], []
        ep_cnt_repls = []
        ep_cnt_env_steps = []
        ep_all_plan_trajs_100 = {}
        trajs_per_img = min(self.vis_trajs_per_img, num_ep)
        n_col = min(5, trajs_per_img)
        

        ## by default, ep_st_idx should be 0, so all probs will be evaluated
        for i_ep in range(self.ep_st_idx, self.ep_st_idx+num_ep):
            
            is_suc = False
            if given_probs is not None: ## set to given value
                raise NotImplementedError
            else:
                ## 2d, (n_probs, 4/29/69) --> 1d, (4/29/69,) ogb obs dim
                st_state = self.problems_dict['start_state'][i_ep,]
                gl_pos = self.problems_dict['goal_pos'][i_ep]

            self.check_obs_dim(gl_pos, 'full') ## sanity check

            ## ------------- Reset the Env -----------------
            # pdb.set_trace() ## check current state
            # self.save_env_cur_img(sv_idx=1) ## commented out

            ## put the agent to the start state
            self.env.reset() ## reset dynamics changed by last rollout
            if 'antmaze' in self.env.name:
                self.env.set_state_with_obs(st_state)
            elif 'humanoidmaze' in self.env.name:
                self.env.set_state_with_full(st_state)
            elif 'antsoccer' in self.env.name: 
                ## set the 42D start state, both ant and ball are set
                self.env.set_state_with_obs(st_state)
            elif 'pointmaze' in self.env.name:
                # pdb.set_trace()
                self.env.set_xy_with_0vel(xy=st_state)
            else: 
                raise NotImplementedError
            
            # pdb.set_trace() ##

            if 'antsoccer' in self.env.name:
                self.env.set_goal(goal_xy=gl_pos[(15,16),]) ## gl_pos of the ball
                self.env.set_ball_start_marker( st_state[(15,16),] )
            else:
                self.env.set_goal(goal_xy=gl_pos[:2]) ## gl_pos is in mj xy coordinate

            mujoco.mj_forward(self.env.model, self.env.data)

            # self.save_env_cur_img(sv_idx=2)

            ## get the value from env after setting
            st_state_mj = self.env.get_ob()
            target_mj = self.env.cur_goal_xy
            
            # pdb.set_trace() ## sanity check

            if 'antmaze' in self.env.name:
                assert np.isclose(st_state_mj, st_state).all()
            elif 'antsoccer' in self.env.name:
                assert np.isclose(st_state_mj, st_state).all()
            elif 'pointmaze' in self.env.name:
                assert np.isclose(st_state_mj, st_state).all()
            else:
                assert np.isclose(self.env.get_state_full(), st_state).all()

            self.env.set_start_marker( st_state_mj[:2] )

            ## ------------------------------------------------

            obs_cur = self.env.get_ob().copy()
            self.check_obs_dim(obs_cur, 'obs')

            # pdb.set_trace() ## check current state

            ## --------- add replanning --------
            repl_wp_cfg = self.repl_wp_cfg
            repl_wp_list = sorted(repl_wp_cfg.keys()) ## a list of wp_idx where replan will happen
            self.policy.n_comp = n_comp_full ## set back the value after each episode

            # pdb.set_trace()
            
            ## ------------------------------------------------
            if self.is_replan == 'at_given_t':
                ## compute the full hzn considering the replanning
                last_repl_wpnt = repl_wp_list[-1]
                full_hzn_with_repl = last_repl_wpnt + self.get_comp_hzn(repl_wp_cfg[last_repl_wpnt])

                utils.print_color(f'[ {i_ep=} ] {repl_wp_cfg=}\n'
                                f'{last_repl_wpnt=} {full_hzn_with_repl=}', c='y')

                # pdb.set_trace()
                tot_hzn = full_hzn_with_repl
            elif self.is_replan == 'ada_dist':
                ## Setup
                self.ada_dist_max_n_repl = self.repl_ada_dist_cfg['max_n_repl']
                self.ada_dist_thres = self.repl_ada_dist_cfg['thres']
                self.ada_dist_type = self.repl_ada_dist_cfg['type']
                ## a dict of int: e.g., 
                ## if {1:4, 2:3, 3:2, 4:1 }, compose 3 models at the second repl
                if self.ada_dist_type == 'm_1':
                    self.ada_dist_comp_sch = self.repl_ada_dist_cfg.get('comp', {})
                    assert len(self.ada_dist_comp_sch) == self.ada_dist_max_n_repl
                elif self.ada_dist_type == 'm_2':
                    ## e.g., [100->300]
                    self.ada_dist_minus_n_wp = self.repl_ada_dist_cfg['ada_dist_minus_n_wp']
                    self.ada_dist_cond_2_extra = self.repl_ada_dist_cfg['cond_2_extra']
                    
                    ## NOTE: by default use all pred state
                    if 'used_idxs' not in self.repl_ada_dist_cfg:
                        self.repl_ada_dist_cfg['used_idxs'] = tuple(range(len(self.obs_select_dim)))
                    self.ada_dist_used_idxs = self.repl_ada_dist_cfg['used_idxs']
                    assert len(self.ada_dist_used_idxs) in [2, 4], 'temporary sanity check'
                    # pdb.set_trace()
                    
                
                ## FIXME: temporary set to 10k
                tot_hzn = 10000 # 5000
                utils.print_color(f'{tot_hzn=}', c='c')
                utils.print_color(f'{self.repl_ada_dist_cfg=}', c='c')
                


            elif self.is_replan == False:
                tot_hzn = self.get_comp_hzn(self.policy.n_comp)
                assert self.repl_wp_cfg == {}
            else:
                raise NotImplementedError
            
            # pdb.set_trace()
            
            ## the fused trajectory (considering replanning)
            fused_traj_ep = np.zeros(shape=(tot_hzn, self.dfu_ndim), dtype=np.float32)
            all_plan_trajs_ep = [] ## each plan_traj of one ep
            imgs_rout = [] ## save the env rendered img
            imgs_rout_agv = [] ## Feb 16

            rollout = [self.env.get_qpos_qvel(),]
            target_xy = self.env.cur_goal_xy ## 2D goal in mj scale

            ep_targets.append(target_xy)
            
            # pdb.set_trace() ## check target
            assert len(target_xy) == 2 ## just the mj xy-coordinate

            
            cnt_extras = 0 ## extra steps after suc
            total_reward = 0 ## accumulate reward of one episode

            
            ## Set Num of Env Steps
            if self.is_replan == 'ada_dist':
                self.n_max_steps = self.repl_ada_dist_cfg['n_max_steps']
            else:
                self.n_max_steps = tot_hzn * self.n_act_per_waypnt + self.extra_env_steps
            wp_idx = 0
            cnt_repl = 0

            # pdb.set_trace()

            
            ## -------------- Do One Eval Episode in a For Loop ------------
            ## loop through timesteps
            for i_et in range(self.n_max_steps):
                ## get current obs
                obs_cur = self.env.get_ob().copy()
                obs_cur_nm = self.full_normalizer.normalize(obs_cur, 'observations')


                ## -----------------------------------------------------
                ## e.g., if wp_idx=104, then it have finished [0,103] wp, in total 104 wp 
                wp_idx = i_et // self.n_act_per_waypnt

                ## do replanning only when it have finished e.g., the first 104 waypoints
                ## and is about to rollout using the 105th waypoints.
                ## So we replace the fused traj from the 105th waypoint.
                
                ## the num of finished of wp is wp_idx
                ## whether this wp have not be used as subgoal yet
                is_wp_not_start = (wp_idx * self.n_act_per_waypnt) == i_et

                ## -------- prepare for corresponding replan strategy --------
                assert len(all_plan_trajs_ep) <= 20, 'sanity check, never use so much repl'
                if i_et > 0:
                    ## check if both at mj scale: Yes, in mj scale (* 4)
                    ## the scale should be very different if we are in higher dim
                    cur_obs_subgl_l2_dist = np.linalg.norm(
                        obs_cur[self.obs_select_dim,][self.ada_dist_used_idxs,] - goal_cur[self.ada_dist_used_idxs,])
                    # pdb.set_trace()
                    ## dist for ada replan
                    ada_used_dist = cur_obs_subgl_l2_dist
                    
                
                if self.is_replan == 'at_given_t':
                    is_do_repl_et = wp_idx in repl_wp_cfg
                elif self.is_replan == 'ada_dist' and i_et > 0:
                    
                    ## distance larger than threshold and within repl limit
                    repl_ada_cond_1 = ada_used_dist > self.ada_dist_thres
                    
                    repl_ada_cond_2 = (wp_idx - prev_dfu_wp_idx - self.ada_dist_cond_2_extra) > len(all_plan_trajs_ep[-1])
                    # pdb.set_trace()
                    ## last pred traj is finished
                    is_do_repl_et = (repl_ada_cond_1 or repl_ada_cond_2) and \
                                    cnt_repl < self.ada_dist_max_n_repl
                    
                    # pdb.set_trace()
                else:
                    is_do_repl_et = False ## no replan

                ## because we might use same wp_idx multiple times
                is_do_repl_et = is_do_repl_et and is_wp_not_start
                ## ------------------------------------------------------------
                
                ## --- Debug Only to be removed ----
                # if 101 <= wp_idx <= 105:
                    # print(f'{i_et=} {wp_idx=}')
                    # pdb.set_trace()

                ## ------

                ## do the compositional planning inside this if-block
                # if i_et == 0:
                if i_et == 0 or (is_do_repl_et and self.is_replan):
                    ## run diffusion model
                    ## pick obs_dims for diffusion planner
                    input_st = obs_cur[ self.obs_select_dim, ]
                    gl_pos_for_dfu = gl_pos[ self.obs_select_dim, ]

                    # pdb.set_trace()

                    g_cond = {
                        ## (2, dim), not normalized
                        'st_gl': np.array( 
                            [input_st[None,], 
                             gl_pos_for_dfu[None,]], dtype=np.float32 )
                    }

                    if is_do_repl_et: ## this is doing a replanning
                        cnt_repl += 1
                        if self.is_replan == 'at_given_t':
                            self.policy.n_comp = repl_wp_cfg[wp_idx]
                        elif self.is_replan == 'ada_dist':
                            
                            if self.ada_dist_type == 'm_1':
                                raise NotImplementedError
                                # pdb.set_trace() ## not checked yet
                                ## a dict of schedule for n_comp
                                tmp_n_comp = self.ada_dist_comp_sch[cnt_repl]
                            elif self.ada_dist_type == 'm_2':
                                ## ** decide the n_comp according to wp_idx **
                                tmp_cnt_wp = wp_idx - prev_dfu_wp_idx
                                
                                tmp_v1 = max(0, (tmp_cnt_wp - self.ada_dist_minus_n_wp)) 
                                prev_hzn = len(all_plan_trajs_ep[-1])
                                tmp_n_comp = math.ceil( (1 - tmp_v1 / prev_hzn) * prev_n_comp )
                                ## BUG FIX
                                tmp_n_comp = max(1, tmp_n_comp)
                                tmp_cur_hzn = self.get_comp_hzn(num_comp=tmp_n_comp)
                                
                                utils.print_color(f'{i_et=} {wp_idx=} {prev_dfu_wp_idx=} {tmp_cnt_wp=} {tmp_v1=}')
                                utils.print_color(f'{i_et=} {wp_idx=} {prev_hzn=} {tmp_n_comp=} {tmp_cur_hzn=} {ada_used_dist=:.2f}')
                                
                                # pdb.set_trace()

                            self.policy.n_comp = tmp_n_comp
                            # pdb.set_trace()
                        else:
                            raise NotImplementedError
                        
                    else:
                        assert self.policy.n_comp == n_comp_full


                    m_out = self.policy.gen_cond_stgl(g_cond=g_cond, b_s=self.b_size_per_prob)

                    ## (tot_hzn, dfu_ndim), e.g., (992, 2)
                    pick_traj = m_out.pick_traj
                    
                    utils.print_color(f'[ Run Planner ]{i_et=} {wp_idx=}', c='y')
                    utils.print_color(f'{pick_traj.shape=}')
                    

                    tmp_tj_end_idx = wp_idx + len(pick_traj)
                    # assert tmp_tj_end_idx <= len(fused_traj_ep)
                    ## TODO: Temporary solution, drop them, not very good
                    if tmp_tj_end_idx > len(fused_traj_ep):
                        pick_traj = pick_traj[:len(fused_traj_ep)-wp_idx] ## discard redundant part

                    if self.is_replan == 'at_given_t' and repl_wp_list[-1] == wp_idx: ## if is last repl traj
                        assert tmp_tj_end_idx == len(fused_traj_ep)
                    elif self.is_replan == 'ada_dist':
                        ## can add some sanity check
                        pass
                    elif self.is_replan == False:
                        assert tmp_tj_end_idx == len(fused_traj_ep)
                    ## this wp_idx in fused traj should not be used
                    ## wp_idx is 0 at the begining
                    fused_traj_ep[ wp_idx:tmp_tj_end_idx ] = pick_traj
                    ## NOTE: just for vis, pad with the last state
                    fused_traj_ep[tmp_tj_end_idx:] = pick_traj[-1]

                    all_plan_trajs_ep.append(pick_traj)

                    prev_dfu_wp_idx = wp_idx ## last wp_idx that run a diffusion planner
                    prev_n_comp = self.policy.n_comp
                    # pdb.set_trace()

                    


                    ## ----- TODO: Just for vis, can delete  -------
                    # if repl_wp_list[-1] == wp_idx or not self.is_replan: ## just save once
                    # if True:
                    if False:
                        tmp_dir_path = self.get_sample_savedir(i_ep)
                        ## func that gets the coordinate for vis
                        tmp_obs_trajs = ogb_get_rowcol_obs_trajs_from_xy_list(self.env, all_plan_trajs_ep)
                        
                        ## None if is no ball env 
                        tmp_trajs_ball = self.extract_ball_trajs_ev(all_plan_trajs_ep, do_to_ij=True)
                        # pdb.set_trace() ## Check Back

                        ## save the model pred for each step
                        img_obs = self.renderer.composite(
                            f'{tmp_dir_path}/ep{i_ep}_et{i_et}_wp{wp_idx}_cp{self.policy.n_comp}_h{len(pick_traj)}_re{cnt_repl}_allpred.png', tmp_obs_trajs, ncol=n_col, trajs_ball=tmp_trajs_ball)

                        ## save the fused pred traj
                        # tmp_obs_trajs = ogb_get_rowcol_obs_trajs_from_xy(self.env, fused_traj_ep[None,])
                        # img_obs = self.renderer.composite(f'{tmp_dir_path}/ep{i_ep}_et{i_et}_wp{wp_idx}_fused.png', tmp_obs_trajs, ncol=n_col)
                    ## ---------------------------------------------
                    
                    # pdb.set_trace()


                ## ---------------- A Simple Controller -----------------
                ## can use inv dyn actions or define a simple controller based on state predictions
                if self.act_control == 'pred_inv':
                    ## *note* fused_traj_ep is the subgoal traj
                    wp_idx = min( wp_idx, len(fused_traj_ep)-1 ) ## make sure last wp is the goal

                    ## pick out the subgoal
                    goal_cur = fused_traj_ep[ wp_idx ] # [:2] can be 2d/29d/69d etc
                    ## NOTE: full_normalizer and train_normalizer must be the same normalizer but different dimension
                    goal_cur_nm = self.train_normalizer.normalize(goal_cur, 'observations')

                    if self.is_use_subgoal_marker:
                        ## put a virtual cylinder for vis
                        self.env.set_subgoal_waypnt( goal_cur[:2] )

                    ## automatically to gpu, after expand [1,obs_dim]
                    obs_cur_nm = utils.to_torch(obs_cur_nm)[None,]
                    goal_cur_nm = utils.to_torch(goal_cur_nm)[None,]
                    ## note that the act_pred is not clipped, e.g., might be > 1.0
                    act_pred_nm = self.inv_model(obs_cur_nm, goal_cur_nm).cpu().numpy()[0]
                    ## normalizer is also [-1, 1], so should be the same but only clip the acts
                    act_pred = self.full_normalizer.unnormalize(act_pred_nm, 'actions')
                    # pdb.set_trace()
                    
                elif self.act_control == 'pd_ogb':
                    # pdb.set_trace()
                    
                    wp_idx = min( wp_idx, len(fused_traj_ep)-1 ) ## make sure last wp is the goal

                    ## pick out the subgoal
                    goal_cur = fused_traj_ep[ wp_idx ] # [:2] can be 2d/29d/69d etc

                    ## put a virtual cylinder for vis
                    self.env.set_subgoal_waypnt( goal_cur[:2] )

                    act_pred_nm = ( goal_cur - obs_cur ) * 5 ## because ogbench code *0.2 for the action
                    act_pred = self.full_normalizer.unnormalize(act_pred_nm, 'actions')


                elif self.act_control == 'dfu_force': ## from diffusion forcing
                    raise not NotImplementedError

                ## ------------------------------------------------------

                # pdb.set_trace()

                obs_cur, rew, terminated, truncated, info = self.env.step(act_pred)

                is_suc = bool(info['success']) or is_suc ## sparse reward

                total_reward += rew
                score = 0 # not used

                if i_et % 100 == 0 and i_et != 0:
                    tmp_dist_1 = np.linalg.norm(obs_cur[self.obs_select_dim,] - goal_cur).item()
                    # r: {rew:.2f} |  R: {total_reward:.2f} |
                    print(
                        f't: {i_et} |'
                        f'pos: { ogb_get_rowcol_obs_trajs_from_xy(self.env, obs_cur[None,:2])[0] } | '
                        f'obs_subgl_dist: {tmp_dist_1:.3f} | '
                        f'Max Steps: {self.n_max_steps}'
                    )

                ## update rollout observations
                rollout.append( self.env.get_qpos_qvel() )
                imgs_rout.append(self.env.render())
                if self.is_rd_agv:
                    imgs_rout_agv.append( self.render_agv_img() )
                    

                if is_suc: ## early termination
                    utils.print_color(f'{i_ep=} {i_et=} {is_suc=}')
                    cnt_extras += 1
                    if cnt_extras == 30:
                        break

            

            ## ---------------------------------------------------
            ## ------------ Finished one eval episode ------------
            ep_all_plan_trajs_100[i_ep] = all_plan_trajs_ep

            ## Jan 17
            ## move the save imgs of all planned trajs to here
            if True:
                tmp_dir_path = self.get_sample_savedir(i_ep)
                ## func that gets the coordinate for vis
                tmp_obs_trajs = ogb_get_rowcol_obs_trajs_from_xy_list(self.env, all_plan_trajs_ep)
                
                ## None if is no ball env 
                tmp_trajs_ball = self.extract_ball_trajs_ev(all_plan_trajs_ep, do_to_ij=True)
                # pdb.set_trace() ## Check Back

                ## save the model pred for each step
                img_obs = self.renderer.composite(
                    f'{tmp_dir_path}/ep{i_ep}_et{i_et}_wp{wp_idx}_cp{self.policy.n_comp}_h{len(pick_traj)}_re{cnt_repl}_allpred.png', tmp_obs_trajs, ncol=n_col, trajs_ball=tmp_trajs_ball)








            # pdb.set_trace()
            rollout = np.array(rollout) ## e.g., (2000+, 29)
            imgs_rout = np.array(imgs_rout) ## e.g., (2000+,200,200,3)

            ## save the interaction video
            tmp_dir_path = self.get_sample_savedir(i_ep)
            utils.save_imgs_to_mp4(imgs=imgs_rout, 
                        save_path=f'{tmp_dir_path}/ep{i_ep}_{is_suc}.mp4', fps=self.vid_fps, n_repeat_first=10)
            if self.is_rd_agv and is_suc:
                utils.save_imgs_to_mp4(imgs=imgs_rout_agv, 
                        save_path=f'{tmp_dir_path}/ep{i_ep}_{is_suc}_agv.mp4', fps=self.vid_fps, n_repeat_first=10)


            ## TODO: please update and check, looks like the mj actual coordinate 
            ## is slightly different with our coordinate?

            ## record the unnormed full pred as well, e.g., (h,4) soccer
            ep_pred_obss_full.append(fused_traj_ep)
            ep_rollouts_full.append(rollout) ## e.g., (1000+,42)
            
            ## fused_traj_ep is already unnormed
            fused_traj_ep = ogb_get_rowcol_obs_trajs_from_xy(self.env, fused_traj_ep)
            rollout_ij2d = ogb_get_rowcol_obs_trajs_from_xy(self.env, rollout[:, :2])

            # pdb.set_trace() ##
            

            ## these two are just 2D
            ep_pred_obss.append( fused_traj_ep ) # shoule be unnormed, w/ only xy, e.g., (992,2,)
            ep_rollouts.append(rollout_ij2d)
            ep_titles_obs.append( f'PredObs: {i_ep}_o{self.dfu_ndim}_{is_suc}' )
            ep_titles_act.append( f'Act: {i_ep}_{is_suc}' )

            ep_is_suc.append(is_suc)
            # pdb.set_trace()
           
            ep_scores.append(score)
            ep_total_rewards.append( total_reward )
            ep_cnt_repls.append(cnt_repl)

            ## --- save multiple trajs in one large image ---
            if len(ep_pred_obss) % trajs_per_img == 0 or i_ep == num_ep - 1:
                ## the direct obs prediction
                tmp_st_idx = (i_ep // trajs_per_img) * trajs_per_img
                tmp_end_idx = tmp_st_idx + trajs_per_img # not inclusive
                tmp_st_idx  -= self.ep_st_idx
                tmp_end_idx  -= self.ep_st_idx
                tmp_tgts = np.array(ep_targets[tmp_st_idx:tmp_end_idx])
                tmp_tgts = ogb_get_rowcol_obs_trajs_from_xy(self.env, tmp_tgts)

                tmp_tls_obs = ep_titles_obs[tmp_st_idx:tmp_end_idx]
                tmp_tls_act = ep_titles_act[tmp_st_idx:tmp_end_idx]

                tmp_scs = np.array(ep_scores[tmp_st_idx:tmp_end_idx])
                tmp_avg_sc = int(tmp_scs.mean())
                tmp_num_f = (tmp_scs < 100).sum() # not suc
                tmp_avg_sr = int(np.mean(ep_is_suc[tmp_st_idx:tmp_end_idx])*100)

                # pdb.set_trace()
                get_is_non_keypt = getattr(self.diffusion, 'get_is_non_keypt', None)

                if get_is_non_keypt is not None:
                    raise NotImplementedError
                    # is_non_keypt = get_is_non_keypt(b_size=trajs_per_img, idx_keypt=None, 
                        # n_comp=self.comp_diffusion.eval_num_cp_trajs)
                else:
                    is_non_keypt = None

                ## check everything whether soccer env will affect locomotion
                tmp_trajs_ball_pred = self.extract_ball_trajs_ev(ep_pred_obss_full[tmp_st_idx:tmp_end_idx], do_to_ij=True)
                tmp_trajs_ball_rout = self.extract_ball_trajs_ev(ep_rollouts_full[tmp_st_idx:tmp_end_idx], do_to_ij=True)

                # pdb.set_trace()

                img_obs, rows_obs = self.renderer.composite(None, np.array(ep_pred_obss[tmp_st_idx:tmp_end_idx]), 
                                ncol=n_col, goal=tmp_tgts, titles=tmp_tls_obs, return_rows=True, is_non_keypt=is_non_keypt,
                                trajs_ball=tmp_trajs_ball_pred)

                ## elem in ep_rollouts have different len
                img_act, rows_act = self.renderer.composite(None, ep_rollouts[tmp_st_idx:tmp_end_idx], 
                                ncol=n_col, goal=tmp_tgts, titles=tmp_tls_act, return_rows=True, 
                                trajs_ball=tmp_trajs_ball_rout)
                
                f_path_3 = join(self.savepath, f'total/{tmp_st_idx}_act_obs_nns{tmp_num_f}_sr{tmp_avg_sr}.png')
                n_rows = len(rows_obs)
                img_whole = []
                ## cat (act,obs) pairs
                for i_r in range(n_rows):
                    img_whole.append( np.concatenate([rows_act[i_r], rows_obs[i_r]], axis=0) ) # 2H,W,C
                img_whole = np.concatenate(img_whole)
                utils.save_img(f_path_3, img_whole)


                

        
        ## ----------------------------------------------------------------
        ## ------------------ Finish All Eval Episodes --------------------

        self.policy.n_comp = n_comp_full ## reset

        utils.print_color(self.env.name,)

        ## metrics based on if success 
        ep_is_suc = np.array(ep_is_suc)
        ep_srate = ep_is_suc.mean() * 100
        ep_fail_idxs = np.where(ep_is_suc == False, )[0]
        assert len(ep_is_suc) == num_ep
        
        # pdb.set_trace()

        ## useless
        avg_ep_scores = np.mean(ep_scores)
        avg_ep_rewards = np.mean(ep_total_rewards)
        ##
        utils.print_color(f'[avg suc rate] {ep_srate=}')
        ## save result as a json file
        json_path = join(self.savepath, '00_rollout.json')

        sc_low_idxs = np.where(np.array(ep_scores) < self.score_low_limit)[0].tolist() # np -> list
        sc_low_idxs_d=dict(zip(sc_low_idxs, np.round(ep_scores, decimals=2)[sc_low_idxs].tolist()))
        print(f'{sc_low_idxs_d=}')

        ## get avg time
        avg_t_dict = self.get_avg_sampling_time()

        # ep_range = range(1, len(ep_scores)+1)
        ep_range = range(len(ep_scores)) ## from 0
        json_data = OrderedDict([
            ('num_ep', num_ep),
            ('ep_srate', ep_srate), ## success rate
            ('avg_ep_scores', avg_ep_scores),
            ('avg_ep_rewards', avg_ep_rewards),
            ('pl_seed', pl_seed),
            # ('', ),
        ])
        json_data = self.update_j_data(json_data)
        json_data.update([
            ('p_type', 'plan_once'),
            ## ----
            ('avg_t_dict', avg_t_dict),
            ## ----
            ('ep_fail_idxs', ep_fail_idxs.tolist()),
            ('sc_low_idx', sc_low_idxs_d),
            ##
            ('ep_is_suc', ep_is_suc.tolist()),
            ('ep_cnt_repls', ep_cnt_repls), ## already list
            ('ep_scores', dict(zip(ep_range, ep_scores)) ),
            ('ep_total_rewards', dict(zip(ep_range, ep_total_rewards)) ),
            ('ncp_pred_time_list', self.policy.ncp_pred_time_list),
        ])
        
        utils.save_json(json_data, json_path)

        ## 
        if self.is_save_pkl:
            self.save_results_to_pkl(
                ep_all_plan_trajs_100=ep_all_plan_trajs_100, 
                ep_pred_obss_full=ep_pred_obss_full, 
                ep_rollouts_full=ep_rollouts_full,
            )


        new_savepath = f'{self.savepath.rstrip(os.sep)}-sr{int(ep_srate)}/'
        utils.rename_fn(self.savepath, new_savepath)
        new_json_path = json_path.replace(self.savepath, new_savepath)
        utils.print_color(f'new_json_path: {new_json_path} \n',)

        
        
        return json_data
    

    def save_results_to_pkl(self, **kwargs):
        import pickle

        num_ep = len( kwargs['ep_rollouts_full'] )

        pkl_path = join(self.savepath, '00_rout.pkl')

        for i_ep in range(num_ep):
            tmp_tj = kwargs['ep_pred_obss_full'][i_ep]
            kwargs['ep_pred_obss_full'][i_ep] = tmp_tj[ :len(kwargs['ep_rollouts_full'][i_ep]) ]

        
        # Open a file in write-binary mode ('wb') to store the pickled object
        with open(f"{pkl_path}", "wb") as file:
            pickle.dump(kwargs, file)
        
        print(f'[save to pickle] {pkl_path}')
    
    def get_avg_sampling_time(self):
        """for appendix, added on Feb 13"""
        ncp_times = np.array(self.policy.ncp_pred_time_list)
        max_ncp = np.unique(ncp_times[:, 0]).max()
        is_max_ncp = np.isclose( max_ncp, ncp_times[:, 0],  )
        n_max_ncp = is_max_ncp.sum().item()
        max_idxs = np.where( is_max_ncp )[0]

        out_dict = {}
        n_rm = 2 ## remove the first one

        tmp_t_list = ncp_times[max_idxs[n_rm:], 1]
        if len(tmp_t_list) == 0:
            tmp_t_list = [0,]
        ## n_eval sampling, times
        out_dict[max_ncp] = {'n': n_max_ncp - n_rm, 
                             't': np.round(np.mean( tmp_t_list ), 4).item(),
                             'e': np.round(np.std( tmp_t_list ), 4).item(),}

        return out_dict

        


    def get_sample_savedir(self, i_ep):
        div_freq = 10
        subdir = str( (i_ep // div_freq) * div_freq )
        sample_savedir = os.path.join(self.savepath, subdir)
        if not os.path.isdir(sample_savedir):
            os.makedirs(sample_savedir)
        return sample_savedir



    def check_is_same_nmlizer(self):
        ## current code assume the normalizer have the same value;
        ## otherwise should unnorm and norm again
        tmp_1 = self.train_normalizer.normalizers['observations'].mins[:2]
        tmp_2 = self.full_normalizer.normalizers['observations'].mins[:2]
        assert np.isclose(tmp_1, tmp_2).all()

    def extract_ball_trajs_ev(self, all_plan_trajs_ep, do_to_ij):
        """
        eval time version of extracting the ball trajs out, see trainer version as well
        all_plan_trajs_ep (bad name): a list of np unnormed traj of different len
        do_to_ij: convert to our cell ij coordinate for vis
        """
        if self.is_soccer_task: ## ball trajs
            ## TODO: this -2 is good now but might cause BUG upon future update
            if all_plan_trajs_ep[0].shape[1] == 42:
                tmp_all_ball_tj_list = [ tmp_tj[:, (15,16)] for tmp_tj in all_plan_trajs_ep]
            else:
                tmp_all_ball_tj_list = [ tmp_tj[:, -2:] for tmp_tj in all_plan_trajs_ep]
            ## a list of np 2d
            if do_to_ij:
                tmp_trajs_2 = ogb_get_rowcol_obs_trajs_from_xy_list(self.env, tmp_all_ball_tj_list)
            else:
                ## not checked but should be in mj xy coordinate
                tmp_trajs_2 = tmp_all_ball_tj_list 
        else:
            tmp_trajs_2 = None
        
        return tmp_trajs_2



    
    def update_j_data(self, json_data: OrderedDict):
        """update the result data dict to be saved to a json"""
        json_data.update([
            ('epoch_diffusion', self.epoch,),
            ('cond_w', self.diffusion.condition_guidance_w,),
            ## ---- New Oct 23
            ('p_h5path', self.problems_h5path),
            ('var_temp', self.diffusion.var_temp),
            ###
            ('use_ddim', self.diffusion.use_ddim),
            ('ddim_eta', self.diffusion.ddim_eta),
            ('ddim_steps', self.diffusion.ddim_num_inference_steps),
            ('is_replan', self.is_replan),
            ('repl_wp_cfg', self.repl_wp_cfg),
            ('repl_ada_dist_cfg', self.repl_ada_dist_cfg),
            ('extra_env_steps', self.extra_env_steps),

            ###
            ('b_size_per_prob', self.b_size_per_prob),
            ('pol_config', self.pol_config),
            ('tj_blder_config', self.tj_blder_config),
            ('n_batch_acc_probs', self.n_batch_acc_probs),
            ##
            ('max_episode_steps', self.n_max_steps),
            ('act_control', self.act_control),
            ('n_act_per_waypnt', self.n_act_per_waypnt),
            ('inv_model_path', self.inv_model_path),
            ('inv_epoch', self.inv_epoch),
            ('is_inv_train_mode', self.inv_model.training),
            ('ep_st_idx', self.ep_st_idx),
            ('hostname', socket.gethostname()),
        ])
        return json_data
    
    
    def check_obs_dim(self, obs_in, o_type='obs'):
        # pdb.set_trace()
        ## sanity check if dim matches env, can ignore
        if 'antmaze' in self.env.name:
            assert obs_in.shape == (29,)
        elif 'human' in self.env.name:
            if o_type == 'obs':
                assert obs_in.shape == (69,)
            else: ## 'hum_full'
                assert obs_in.shape == (55,)
        elif 'antsoccer' in self.env.name:
            assert obs_in.shape == (42,)
        elif 'pointmaze' in self.env.name:
            assert obs_in.shape == (2,)
        else: 
            raise NotImplementedError
        
    def save_env_cur_img(self, sv_idx):
        tmp_img = self.env.render()
        utils.save_img(f'./luotest_{sv_idx}.png', tmp_img)

    def get_comp_hzn(self, num_comp):
        return self.diffusion.get_total_hzn(num_comp=num_comp)
    

    def render_agv_img(self):
        ### setup the agv rendering videos
        self.env.camera_name = 'back_luo_v3'
        

        vc_name = f'visual_circle'
        rgba_ori = self.env.model.geom(vc_name).rgba.copy()
        ## set the cylinder invisible
        self.env.model.geom(vc_name).rgba = np.array([0. , 0. , 0. , 0.])
        img_r = self.env.render()
        
        ## ---------- set everything back ----------
        self.env.camera_name = None
        self.env.model.geom(vc_name).rgba = rgba_ori

        return img_r
