import numpy as np
from os.path import join
from diffuser.models.cd_stgl_sml_dfu.stgl_sml_policy_v1 import Stgl_Sml_Policy_V1
from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.models.helpers import apply_conditioning
import diffuser.datasets as datasets
import diffuser.utils as utils
from datetime import datetime
import os.path as osp
import copy, pdb, json
import pdb, torch, os
from diffuser.guides.render_m2d import Maze2dRenderer_V2
from collections import OrderedDict
from diffuser.datasets.d4rl import Is_Gym_Robot_Env


class Stgl_Sml_Maze2DEnvPlanner_V1:
    """
    Oct 19, A high level planner that 
    1. load the model out
    2. loop through all the required episodes.
    3. summarize the metrics results
    """
    def __init__(self, args_train, args) -> None:
        self.args_train = args_train
        self.args = args

        self.plan_n_ep = args.plan_n_ep
        self.b_size_per_prob = args.b_size_per_prob

        # self.num_episodes = num_ep
        # self.pl_seed = None
        self.n_batch_acc_probs = args.n_batch_acc_probs
        self.vis_trajs_per_img = 10
        self.score_low_limit = 100
        # np.set_printoptions(precision=3, suppress=True)
        self.act_control = 'dfu_force' # 'pred'
        # self.eval_cfg = eval_cfg
        self.is_vis_single = args.is_vis_single


    def setup_load(self, ld_config):
        """
        used in a separate launch evaluation, where model should be loaded from file
        """
        args_train = self.args_train
        args = self.args

        if Is_Gym_Robot_Env:
            self.env = datasets.load_env_gym_robo(args.dataset)
        else:
            self.env = datasets.load_environment(args.dataset)
            self.env.seed(0) ## default

        assert self.env.reward_type == 'sparse'
        self.env.max_episode_steps = self.args.env_n_max_steps
        
        ## TODO: Oct 21 21:23
        ## loading the dataset is too slow...
        # dfu_exp = utils.load_cd_sml_diffusion(
        #     args.logbase, args_train.dataset, 
        #     args_train.exp_name, epoch=args.diffusion_epoch,
        #     ld_config=ld_config)
        
        dfu_exp = utils.load_stgl_sml_diffusion(
            args.logbase, args_train.dataset, 
            args_train.exp_name, epoch=args.diffusion_epoch,
            ld_config=ld_config)


        # pdb.set_trace() ## check args.diffusion_epoch controllable? No, If inside parser
        ## NOTE:
        ## TODO: Oct 21 22:15 From Here, add non-load normalizer!!!
        self.train_normalizer = utils.load_stgl_sml_datasetNormalizer(args_train,)

        self.diffusion: Stgl_Sml_GauDiffusion_InvDyn_V1 = dfu_exp.ema ## should an ema model
        self.diffusion.var_temp = args.var_temp
        self.diffusion.condition_guidance_w = args.cond_w
        ## NEW
        self.diffusion.use_ddim = getattr(args, 'use_ddim', True)
        self.diffusion.ddim_eta = getattr(args, 'ddim_eta', self.diffusion.ddim_eta)

        self.dataset = dfu_exp.dataset
        # self.train_normalizer = self.dataset.normalizer
        self.renderer: Maze2dRenderer_V2
        self.renderer = dfu_exp.renderer
        self.trainer = dfu_exp.trainer
        
        if self.diffusion.is_inv_dyn_dfu:
            dfu_name =  getattr(self.diffusion, 'dfu_name', 'our_stgl_sml')
            if dfu_name == 'dd_maze':
                ## Jan 18 For the Decision Diffuser Baseline
                self.pol_config = {}
                self.tj_blder_config = {}
                for k in args.__dict__.keys():
                    if 'ev_' in k:
                        self.pol_config[k] = args.__dict__[k]
                ## TODO: NEW
                from diffuser.baselines.dd_maze.dd_maze_policy_v1 import DD_Maze_Policy_V1
                self.policy = DD_Maze_Policy_V1(self.diffusion, self.train_normalizer,
                                                 self.pol_config)
                
            else:
                # pdb.set_trace() ## TODO: Oct 21 16:31pm, From Here
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
            assert False
            # self.policy = Policy(self.comp_diffusion, self.dataset.normalizer)
        
        self.savepath = args.savepath
        self.epoch = dfu_exp.epoch

        utils.print_color(f'Load From {self.epoch=}', c='y')

        self.setup_general()
        self.dataset_config = args_train.dataset_config
        self.obs_select_dim = self.dataset_config['obs_select_dim']
        self.dset_type = self.dataset_config.get('dset_type', 'ours')


        # self.diffusion.update_eval_config(eval_cfg=args.as_dict())
        # pdb.set_trace()
        # print(f'{self.diffusion.eval_num_cp_trajs=}')



    def setup_general(self):
        '''general stuff for both types of setup'''
        utils.mkdir(self.savepath)
        self.savepath_root = self.savepath
        self.load_ev_problems()



    def load_ev_problems(self):
        ## Oct 21: get the file name and load the dict out
        self.problems_h5path = utils.get_stgl_lh_ev_probs_fname(self.env.name)
        self.problems_dict = utils.load_stgl_lh_ev_probs_hdf5(h5path=self.problems_h5path)        
        # pdb.set_trace()
        return
    


    def plan_once_parallel(self, pl_seed=None, given_probs=None): ## num_ep
        '''
        code to launch planning,
        Faster version of plan_one.
        20 * 20 (b_s_per_sample) = 400, parallelly generate trajectories for 20 problems,
        but cannot support replanning during the rollout of one episode
        given_probs: dict to specify start/goal, len is ??
        '''
        if pl_seed is not None:
            utils.set_seed(pl_seed) ## seed everything
        if given_probs is not None: # if given, just evaluate the given states
            num_ep = len(given_probs)
        else:
            num_probs = len(self.problems_dict['start_state'])
            num_ep = num_probs if self.plan_n_ep == -100 else self.plan_n_ep
        
        utils.print_color(f'[plan_once_parallel]: {num_ep=}')

        ep_scores = []
        ep_total_rewards = []
        ep_pred_obss = []
        ep_rollouts = []
        ep_targets = []
        ep_is_suc = [] ## a list of bool
        ep_titles_obs, ep_titles_act = [], []
        trajs_per_img = min(self.vis_trajs_per_img, num_ep)
        n_col = min(5, trajs_per_img)
        # pdb.set_trace()


        ## loop through all eval episodes
        for i_ep in range(num_ep):

            ## ------- generate plans via diffusion -------
            if i_ep % self.n_batch_acc_probs == 0:
                tmp_last_p_idx = min(num_ep, i_ep + self.n_batch_acc_probs)
                ## (n_probs, 4)
                input_st_acc = self.problems_dict['start_state'][i_ep:tmp_last_p_idx, self.obs_select_dim]
                ## (n_probs, 2)
                gl_pos_acc = self.problems_dict['goal_pos'][i_ep:tmp_last_p_idx]

                g_cond = {
                        ## (2, n_probs, dim)
                        'st_gl': np.array( [
                            input_st_acc, 
                            gl_pos_acc,], dtype=np.float32 )
                    }
                # pdb.set_trace() ## TODO: check above
                ## two lists, of len n_probs
                ## [out1, out2, ...] ; [traj1 2d (h,d), traj2, ...]
                m_out_list, pick_traj_acc = self.policy.gen_cond_stgl_parallel(g_cond=g_cond, b_s=self.b_size_per_prob)
                
                # pdb.set_trace()
            ## -------------------------------------------





            
            is_suc = False
            _ = self.env.reset()
            if given_probs is not None: ## set to given value
                raise NotImplementedError
                ## must be np
                # self.env.set_state(qpos=given_starts[i_ep, :2], qvel=given_starts[i_ep, 2:])
                # st_state = given_starts[i_ep, :2]
            else:
                ## 2d, (n_probs, 4) --> 1d, (4,) qpos, qvel
                st_state = self.problems_dict['start_state'][i_ep,]
                gl_pos = self.problems_dict['goal_pos'][i_ep]


            self.env.set_state(qpos=st_state[:2], qvel=st_state[2:])
            assert  ( st_state[2:] ==  np.array([0.,0.]) ).all(), 'zero start speed'
            assert  gl_pos.shape == (2,), 'just for maze2d env, not sure for antmaze'

            
            # pdb.set_trace() ## check current state
            
            ## <d4rl.pointmaze.maze_model.MazeEnv>
            self.env.set_target(target_location=gl_pos) ## used to change the target





            
            target = self.env._target # tuple large: (7,9)
            ep_targets.append(target)
            
            # pdb.set_trace() ## check target
            
            assert len(target) == len(self.obs_select_dim)

            ## we set up the start and goal in the env above
            ## now interact with the env
            pick_tj_ep = pick_traj_acc[i_ep % self.n_batch_acc_probs]
            
            # pdb.set_trace()
            is_suc, total_reward, score, rollout = self.env_interact_1_ep(
                pick_traj=pick_tj_ep, start_state=st_state, target=gl_pos)
            
            # pdb.set_trace()
            

            ## ---------------------------------------------------
            ## ------------ Finished one eval episode ------------

            ep_pred_obss.append( pick_tj_ep ) # shoule be unnormed, (tot_hzn,2)
            ep_rollouts.append(rollout)
            ep_titles_obs.append( f'PredObs: {i_ep}_score{int(score)}' )
            ep_titles_act.append( f'Act: {i_ep}_score{int(score)}' )

            ep_is_suc.append(is_suc)
            # pdb.set_trace() ## TODO: From Here Oct 24 16:45
           
            ep_scores.append(score)
            ep_total_rewards.append( total_reward )

            ## TODO: ensure the last tail parts of results are saved to img
            ## --- save multiple trajs in one large image ---
            if len(ep_pred_obss) % trajs_per_img == 0 or i_ep == num_ep - 1:
                ## the direct obs prediction
                tmp_st_idx = (i_ep // trajs_per_img) * trajs_per_img
                tmp_end_idx = tmp_st_idx + trajs_per_img # not inclusive
                tmp_tgts = np.array(ep_targets[tmp_st_idx:tmp_end_idx])
                tmp_tls_obs = ep_titles_obs[tmp_st_idx:tmp_end_idx]
                tmp_tls_act = ep_titles_act[tmp_st_idx:tmp_end_idx]

                tmp_scs = np.array(ep_scores[tmp_st_idx:tmp_end_idx])
                tmp_avg_sc = int(tmp_scs.mean())
                tmp_num_f = (tmp_scs < 100).sum() # not suc

                # pdb.set_trace()
                get_is_non_keypt = getattr(self.diffusion, 'get_is_non_keypt', None)

                if get_is_non_keypt is not None:
                    raise NotImplementedError
                    # is_non_keypt = get_is_non_keypt(b_size=trajs_per_img, idx_keypt=None, 
                        # n_comp=self.comp_diffusion.eval_num_cp_trajs)
                else:
                    is_non_keypt = None

                # pdb.set_trace()

                img_obs, rows_obs = self.renderer.composite(None, np.array(ep_pred_obss[tmp_st_idx:tmp_end_idx]), 
                                ncol=n_col, goal=tmp_tgts, titles=tmp_tls_obs, return_rows=True, is_non_keypt=is_non_keypt)

                img_act, rows_act = self.renderer.composite(None, np.array(ep_rollouts[tmp_st_idx:tmp_end_idx]), 
                                ncol=n_col, goal=tmp_tgts, titles=tmp_tls_act, return_rows=True, )
                
                f_path_3 = join(self.savepath, f'{tmp_st_idx}_act_obs_nns{tmp_num_f}_sc{tmp_avg_sc}.png')
                n_rows = len(rows_obs)
                img_whole = []
                ## cat (act,obs) pairs
                for i_r in range(n_rows):
                    img_whole.append( np.concatenate([rows_act[i_r], rows_obs[i_r]], axis=0) ) # 2H,W,C
                img_whole = np.concatenate(img_whole)
                utils.save_img(f_path_3, img_whole)


                

        
        ## ----------------------------------------------------------------
        ## ------------------ Finish All Eval Episodes --------------------

        
        utils.print_color(self.env.name,)

        ## Oct 21 New, metrics based on if success 
        ep_is_suc = np.array(ep_is_suc)
        ep_srate = ep_is_suc.mean() * 100
        ep_fail_idxs = np.where(ep_is_suc == False, )[0]
        assert len(ep_is_suc) == num_ep
        # pdb.set_trace()


        avg_ep_scores = np.mean(ep_scores)
        avg_ep_rewards = np.mean(ep_total_rewards)
        # print(f'{ep_scores=}')
        # print(f'{ep_total_rewards=}')
        print(f'{avg_ep_scores=}')
        print(f'{avg_ep_rewards=}')
        ## save result as a json file
        json_path = join(self.savepath, '00_rollout.json')

        sc_low_idxs = np.where(np.array(ep_scores) < self.score_low_limit)[0].tolist() # np -> list
        sc_low_idxs_d=dict(zip(sc_low_idxs, np.round(ep_scores, decimals=2)[sc_low_idxs]))
        print(f'{sc_low_idxs_d=}')


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
            ('p_type', 'plan_once_parallel'),
            ## ----
            ('ep_fail_idxs', ep_fail_idxs.tolist()),
            ('sc_low_idx', sc_low_idxs_d),
            ##
            ('ep_is_suc', ep_is_suc.tolist()),
            ('ep_scores', dict(zip(ep_range, ep_scores)) ),
            ('ep_total_rewards', dict(zip(ep_range, ep_total_rewards)) ),
        ])

        utils.save_json(json_data, json_path)

        # print(f'[save_plan_result]: save to {json_path}')
        new_savepath = f'{self.savepath.rstrip(os.sep)}-sc{int(avg_ep_scores)}/'
        utils.rename_fn(self.savepath, new_savepath)
        new_json_path = json_path.replace(self.savepath, new_savepath)
        utils.print_color(f'new_json_path: {new_json_path} \n',)
        
        return json_data







    def env_interact_1_ep(self, pick_traj, start_state, target):
        """
        Interact for one episode, moved out from larger function
        pick_traj: the traj to follow, unnormed
        """
        ## make sure robot state is aligned with the outer loop
        assert ( start_state == self.env.state_vector() ).all()
        assert ( target == self.env._target ).all()
        # pdb.set_trace() ## check pick_traj

        assert np.isclose(pick_traj[0], start_state[ self.obs_select_dim, ]).all()
        assert np.isclose(pick_traj[-1], target).all()
        ## metrics to return
        is_suc = False
        total_reward = 0
        rollout = [start_state.copy(),]


        n_max_steps = self.env.max_episode_steps

        for t in range(n_max_steps):
            ## maze2d: np (4,)
            cur_state = self.env.state_vector().copy()
            
            assert len(cur_state) == 4, 'only for maze2D, for now.'


            ## ----------- A Simple Controller ------------
            
            ## can use actions or define a simple controller based on state predictions
            if self.act_control == 'pd_ori':
                raise NotImplementedError
            elif self.act_control == 'pred':
                raise NotImplementedError ## can refer to other .py
            elif self.act_control == 'dfu_force': ## from diffusion forcing
                ## pick_traj ## only have obs actually
                ### ----- get the desired vel -----
                if t == 0:
                    plan_vel = pick_traj[t, :2] - cur_state[:2]
                elif t > 0 and t < len(pick_traj):
                    plan_vel = pick_traj[t, :2] - pick_traj[t - 1, :2]
                else:
                    ## large than our traj
                    plan_vel = np.zeros_like(cur_state[2:])
                ### ----- get the desired position -----
                if t < len(pick_traj):
                    plan_pos = pick_traj[t,:2]
                else:
                    plan_pos = pick_traj[-1,:2]
                    assert np.isclose(plan_pos, target, atol=0.09).all()
                
                ## PD controller: hyperparam ?
                action = 12.5 * (plan_pos - cur_state[:2]) + 1.2 * (plan_vel - cur_state[2:])
                action = np.clip(action, a_min=-1, a_max=1)
                # pdb.set_trace() ## the first action should be zero


            ## ---------------------------

            # pdb.set_trace()
            
            ## by default, terminal is False forever
            ## np 1d (4,); float; bool (is finished, always False); 
            obs_next, rew, terminal, _ = self.env.step(action)

            is_suc = rew > 0 or is_suc ## sparse reward

            total_reward += rew
            score = self.env.get_normalized_score(total_reward) * 100

            # if t % 50 == 0:
            if t == n_max_steps - 1:
                print(
                    f't: {t} | r: {rew:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                    f'pos: {obs_next[:2]} | vel: {obs_next[2:]} | action: {action} | '
                    f'Max Steps: {n_max_steps}'
                )

            ## update rollout observations
            rollout.append(obs_next.copy())

            
        

        ## -----------------------------------------------------------
        ## ------------ Finished one env interact episode ------------
        # pdb.set_trace()

        return is_suc, total_reward, score, rollout
    

    ### -------------------------------------------------
    ### ------------------ Ben --------------------------
    ### -------------------------------------------------


    ### TODO: Oct 30 update the major planning code
    def ben_plan_once_parallel(self, pl_seed=None, given_probs=None): ## num_ep
        '''
        ** Ben Env based on Gym Robotics **
        code to launch planning,
        Faster version of plan_one.
        20 * 20 (b_s_per_sample) = 400, parallelly generate trajectories for 20 problems,
        but cannot support replanning during the rollout of one episode
        given_probs: dict to specify start/goal, len is ??
        '''
        assert Is_Gym_Robot_Env
        
        if pl_seed is not None:
            utils.set_seed(pl_seed) ## seed everything
        
        # pdb.set_trace()

        if given_probs is not None: # if given, just evaluate the given states
            num_ep = len(given_probs)
        else:
            num_probs = len(self.problems_dict['start_state'])
            num_ep = num_probs if self.plan_n_ep == -100 else self.plan_n_ep
        
        utils.print_color(f'[ben_plan_once_parallel]: {num_ep=}')

        ep_scores = []
        ep_total_rewards = []
        ep_pred_obss = []
        ep_rollouts = []
        ep_targets = []
        ep_is_suc = [] ## a list of bool
        ep_titles_obs, ep_titles_act = [], []
        trajs_per_img = min(self.vis_trajs_per_img, num_ep)
        n_col = min(5, trajs_per_img)
        # pdb.set_trace()


        ## loop through all eval episodes
        for i_ep in range(num_ep):

            ## ------- generate plans via diffusion -------
            if i_ep % self.n_batch_acc_probs == 0:
                tmp_last_p_idx = min(num_ep, i_ep + self.n_batch_acc_probs)
                ## (n_probs, 4), lg: [7,1] -> [7,4]
                input_st_acc = self.problems_dict['start_state'][i_ep:tmp_last_p_idx, self.obs_select_dim]
                ## (n_probs, 2)
                gl_pos_acc = self.problems_dict['goal_pos'][i_ep:tmp_last_p_idx]


                ## we need to make the cell-idx level state into mujoco coordinate level first
                input_st_acc = utils.ben_luo_rowcol_to_xy(self.dset_type, trajs=input_st_acc)
                gl_pos_acc = utils.ben_luo_rowcol_to_xy(self.dset_type, trajs=gl_pos_acc)
                # pdb.set_trace()
                ## utils.ben_xy_to_luo_rowcol(self.dset_type, trajs=gl_pos_acc) ## just for check


                g_cond = {
                        ## (2, n_probs, dim)
                        'st_gl': np.array( [
                            input_st_acc, 
                            gl_pos_acc,], dtype=np.float32 )
                    }
                ## two lists, of len n_probs
                ## [out1, out2, ...] ; [traj1 2d (h,d), traj2, ...]
                
                m_out_list, pick_traj_acc = self.policy.gen_cond_stgl_parallel(g_cond=g_cond, b_s=self.b_size_per_prob)
                
            ## -------------------------------------------





            
            is_suc = False
            _ = self.env.reset() ## obs_dict, info
            if given_probs is not None: ## set to given value
                raise NotImplementedError
                ## must be np
                # self.env.set_state(qpos=given_starts[i_ep, :2], qvel=given_starts[i_ep, 2:])
                # st_state = given_starts[i_ep, :2]
            else:
                ## 2d, (n_probs, 4) --> 1d, (4,) qpos, qvel
                ## in cell-idx coordinate
                st_state = self.problems_dict['start_state'][i_ep,]
                gl_pos = self.problems_dict['goal_pos'][i_ep]


            # self.env.set_state(qpos=st_state[:2], qvel=st_state[2:])
            assert  ( st_state[2:] ==  np.array([0.,0.]) ).all(), 'zero start speed'
            assert  gl_pos.shape == (2,), 'just for maze2d env, not sure for antmaze'

            
            # pdb.set_trace() ## check current state
            
            ## <gymnasium_robotics.envs.maze.point_maze.PointMazeEnv>
            # self.env.set_target(target_location=gl_pos) ## used to change the target
            ssg = {'reset_cell': st_state[:2], 'goal_cell': gl_pos,}
            obs_gyro, _ = self.env.reset_given(options=ssg)
            st_state_mj = obs_gyro['observation']
            assert (st_state_mj[2:] == 0).all()
            target_mj = self.env.goal ## should be set, but normalized in mujoco xy

            # pdb.set_trace() ## check current state

            ep_targets.append(gl_pos) ## just for vis, so input cell-idx

            
            assert len(target_mj) == len(self.obs_select_dim)

            ## we set up the start and goal in the env above
            ## now interact with the env
            pick_tj_ep = pick_traj_acc[i_ep % self.n_batch_acc_probs]
            
            # pdb.set_trace()
            is_suc, total_reward, score, rollout = self.ben_env_interact_1_ep(
                pick_traj=pick_tj_ep, start_state=st_state_mj, target=target_mj)
            
            ## for vis, we need to transform to cell-idx coordinate
            pick_tj_ep = utils.ben_xy_to_luo_rowcol(self.dset_type, pick_tj_ep)
            rollout = np.array(rollout)
            rollout[:, :2] = utils.ben_xy_to_luo_rowcol(self.dset_type, rollout[:, :2])
            # pdb.set_trace() ## Oct 31
            

            ## ---------------------------------------------------
            ## ------------ Finished one eval episode ------------

            ep_pred_obss.append( pick_tj_ep ) # shoule be unnormed, (tot_hzn,2)
            ep_rollouts.append(rollout)
            ep_titles_obs.append( f'PredObs: {i_ep}_score{int(score)}' )
            ep_titles_act.append( f'Act: {i_ep}_score{int(score)}' )

            ep_is_suc.append(is_suc)
            # pdb.set_trace() ## TODO: From Here Oct 24 16:45
           
            ep_scores.append(score)
            ep_total_rewards.append( total_reward )

            ## TODO: ensure the last tail parts of results are saved to img
            ## --- save multiple trajs in one large image ---
            if len(ep_pred_obss) % trajs_per_img == 0 or i_ep == num_ep - 1:
                ## the direct obs prediction
                tmp_st_idx = (i_ep // trajs_per_img) * trajs_per_img
                tmp_end_idx = tmp_st_idx + trajs_per_img # not inclusive
                tmp_tgts = np.array(ep_targets[tmp_st_idx:tmp_end_idx])
                tmp_tls_obs = ep_titles_obs[tmp_st_idx:tmp_end_idx]
                tmp_tls_act = ep_titles_act[tmp_st_idx:tmp_end_idx]

                tmp_scs = np.array(ep_scores[tmp_st_idx:tmp_end_idx])
                tmp_sr = int(np.array(ep_is_suc[tmp_st_idx:tmp_end_idx]).mean()*100)
                tmp_avg_sc = int(tmp_scs.mean())
                tmp_num_f = (tmp_scs < 100).sum() # not suc

                # pdb.set_trace()
                get_is_non_keypt = getattr(self.diffusion, 'get_is_non_keypt', None)

                if get_is_non_keypt is not None:
                    raise NotImplementedError
                    # is_non_keypt = get_is_non_keypt(b_size=trajs_per_img, idx_keypt=None, 
                        # n_comp=self.comp_diffusion.eval_num_cp_trajs)
                else:
                    is_non_keypt = None

                # pdb.set_trace()

                img_obs, rows_obs = self.renderer.composite(None, np.array(ep_pred_obss[tmp_st_idx:tmp_end_idx]), 
                                ncol=n_col, goal=tmp_tgts, titles=tmp_tls_obs, return_rows=True, is_non_keypt=is_non_keypt)

                img_act, rows_act = self.renderer.composite(None, np.array(ep_rollouts[tmp_st_idx:tmp_end_idx]), 
                                ncol=n_col, goal=tmp_tgts, titles=tmp_tls_act, return_rows=True, )
                
                f_path_3 = join(self.savepath, f'{tmp_st_idx}_act_obs_nns{tmp_num_f}_sr{tmp_sr}.png')
                n_rows = len(rows_obs)
                img_whole = []
                ## cat (act,obs) pairs
                for i_r in range(n_rows):
                    img_whole.append( np.concatenate([rows_act[i_r], rows_obs[i_r]], axis=0) ) # 2H,W,C
                img_whole = np.concatenate(img_whole)
                utils.save_img(f_path_3, img_whole)


                

        
        ## ----------------------------------------------------------------
        ## ------------------ Finish All Eval Episodes --------------------
        ## visualize each traj in a separate single img for paper
        if self.is_vis_single:
            self.ben_save_traj_img_single(ep_pred_obss, 'pred_obs', targets_un=ep_targets)
            self.ben_save_traj_img_single(ep_rollouts, 'rout_act', targets_un=ep_targets)

        
        utils.print_color(self.env.name,)

        ## Oct 21 New, metrics based on if success 
        ep_is_suc = np.array(ep_is_suc)
        ep_srate = ep_is_suc.mean() * 100
        ep_fail_idxs = np.where(ep_is_suc == False, )[0]
        assert len(ep_is_suc) == num_ep
        # pdb.set_trace()


        avg_ep_scores = np.mean(ep_scores)
        avg_ep_rewards = np.mean(ep_total_rewards)
        # print(f'{ep_scores=}')
        # print(f'{ep_total_rewards=}')
        print(f'{avg_ep_scores=}')
        print(f'{avg_ep_rewards=}')
        ## save result as a json file
        json_path = join(self.savepath, '00_rollout.json')

        sc_low_idxs = np.where(np.array(ep_scores) < self.score_low_limit)[0].tolist() # np -> list
        sc_low_idxs_d=dict(zip( sc_low_idxs, np.round(ep_scores, decimals=2)[sc_low_idxs].tolist() ))
        print(f'{sc_low_idxs_d=}')


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
            ('p_type', 'ben_plan_once_parallel'),
            ## ----
            ('ep_fail_idxs', ep_fail_idxs.tolist()),
            ('sc_low_idx', sc_low_idxs_d),
            ##
            ('ep_is_suc', ep_is_suc.tolist()),
            ('ep_scores', dict(zip(ep_range, ep_scores)) ),
            ('ep_total_rewards', dict(zip(ep_range, ep_total_rewards)) ),
        ])
        
        ## --- for debug, can delete
        # for k,v in json_data.items():
            # print(k, type(v))
        ## -------

        utils.save_json(json_data, json_path)

        # print(f'[save_plan_result]: save to {json_path}')
        new_savepath = f'{self.savepath.rstrip(os.sep)}-sr{int(ep_srate)}/'
        utils.rename_fn(self.savepath, new_savepath)
        new_json_path = json_path.replace(self.savepath, new_savepath)
        utils.print_color(f'new_json_path: {new_json_path} \n',)


        
        
        return json_data
    

    def ben_save_traj_img_single(self, trajs_un, tj_type, targets_un):
        """
        save each traj in a separate single img
        """
        ## trajs_un: B,H,2
        for i_ep, tmp_tj in enumerate(trajs_un):
            
            # pdb.set_trace()
            ## to shape: [1,2]
            tmp_img = self.renderer.renders(tmp_tj, goal=targets_un[i_ep][None,], fig_dpi=250,)

            tmp_path = join(self.savepath, f'{tj_type}', f'{i_ep}_{tj_type}.png')
            utils.save_img(save_path=tmp_path, img=tmp_img)
        
        return
        
        




    def ben_env_get_obs(self) -> np.ndarray:
        obs, info = self.env.point_env._get_obs() ## info is empty
        return obs

    
    def ben_env_interact_1_ep(self, pick_traj, start_state, target):
        """
        ** Ben **
        Interact for one episode, Ben's Version
        pick_traj: the traj to follow, unnormed
        """
        ## make sure robot state is aligned with the outer loop
        # self.env: gymnasium_robotics.envs.maze.point_maze.PointMazeEnv
        assert ( start_state == self.ben_env_get_obs() ).all()
        assert ( target == self.env.goal ).all()
        # pdb.set_trace() ## check pick_traj

        ## FIXME: Oct 30 uncomment
        # assert np.isclose(pick_traj[0], start_state[ self.obs_select_dim, ], atol=0.05).all()
        # assert np.isclose(pick_traj[-1], target, atol=0.05).all()

        ## metrics to return
        is_suc = False
        total_reward = 0
        rollout = [start_state.copy(),]


        n_max_steps = self.env.max_episode_steps

        for t in range(n_max_steps):
            ## maze2d: np (4,)
            cur_state = self.ben_env_get_obs().copy()
            
            assert len(cur_state) == 4, 'only for maze2D, for now.'


            ## ----------- A Simple Controller ------------
            
            ## can use actions or define a simple controller based on state predictions
            if self.act_control == 'pd_ori':
                raise NotImplementedError
            elif self.act_control == 'pred':
                raise NotImplementedError ## can refer to other .py
            elif self.act_control == 'dfu_force': ## from diffusion forcing
                ## pick_traj ## only have obs actually
                ### ----- get the desired vel -----
                if t == 0:
                    plan_vel = pick_traj[t, :2] - cur_state[:2]
                elif t > 0 and t < len(pick_traj):
                    plan_vel = pick_traj[t, :2] - pick_traj[t - 1, :2]
                else:
                    ## large than our traj
                    plan_vel = np.zeros_like(cur_state[2:])
                ### ----- get the desired position -----
                if t < len(pick_traj):
                    plan_pos = pick_traj[t,:2]
                else:
                    plan_pos = pick_traj[-1,:2]
                    ## FIXME: Oct 30: uncomment
                    # assert np.isclose(plan_pos, target, atol=0.09).all()
                
                ## PD controller: hyperparam ?
                action = 12.5 * (plan_pos - cur_state[:2]) + 1.2 * (plan_vel - cur_state[2:])
                action = np.clip(action, a_min=-1, a_max=1)
                # pdb.set_trace() ## the first action should be zero


            ## ---------------------------

            ## ------ delete -------
            # pdb.set_trace()
            
            ## by default, terminal is False forever
            ## np 1d (4,); float; bool (is finished, always False); 
            # obs_next, rew, terminal, _ = self.env.step(action)

            ## ------ delete above -------
            
            ## gym robotic, ben's env
            ## obs is dict: []; done is true when arrive goal; trunc is always false
            obs_gyro, rew, done, _, _ = self.env.step(action)
            obs_next = obs_gyro['observation']

            is_suc = rew > 0 or is_suc ## sparse reward

            total_reward += rew
            score = 0 # self.env.get_normalized_score(total_reward) * 100

            # if t % 50 == 0:
            if t == n_max_steps - 1:
                print(
                    f't: {t} | r: {rew:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                    f'pos: {obs_next[:2]} | vel: {obs_next[2:]} | action: {action} | '
                    f'Max Steps: {n_max_steps}'
                )

            ## update rollout observations
            rollout.append(obs_next.copy())

            
        

        ## -----------------------------------------------------------
        ## ------------ Finished one env interact episode ------------
        # pdb.set_trace()

        return is_suc, total_reward, score, rollout

        






        

    ## ---------------------------------------
    ## ---------------------------------------
    ## Old Slow version, but can support replanning because we plan one traj each time

    def plan_once(self, pl_seed=None, given_probs=None): ## num_ep
        '''
        code to launch planning
        given_probs: dict to specify start/goal, len is ??
        '''
        if pl_seed is not None:
            utils.set_seed(pl_seed) ## seed everything
        if given_probs is not None: # if given, just evaluate the given states
            num_ep = len(given_probs)
        else:
            num_probs = len(self.problems_dict['start_state'])
            num_ep = num_probs if self.plan_n_ep == -100 else self.plan_n_ep
        
        utils.print_color(f'[plan_once]: {num_ep=}')

        ep_scores = []
        ep_total_rewards = []
        ep_pred_obss = []
        ep_rollouts = []
        ep_targets = []
        ep_is_suc = [] ## a list of bool
        ep_titles_obs, ep_titles_act = [], []
        trajs_per_img = min(self.vis_trajs_per_img, num_ep)
        n_col = min(5, trajs_per_img)
        # pdb.set_trace()


        for i_ep in range(num_ep):
            
            is_suc = False
            _ = self.env.reset()
            if given_probs is not None: ## set to given value
                raise NotImplementedError
                ## must be np
                # self.env.set_state(qpos=given_starts[i_ep, :2], qvel=given_starts[i_ep, 2:])
                # st_state = given_starts[i_ep, :2]
            else:
                ## 2d, (n_probs, 4) --> 1d, (4,) qpos, qvel
                st_state = self.problems_dict['start_state'][i_ep,]
                gl_pos = self.problems_dict['goal_pos'][i_ep]


            self.env.set_state(qpos=st_state[:2], qvel=st_state[2:])
            assert  ( st_state[2:] ==  np.array([0.,0.]) ).all(), 'zero start speed'
            assert  gl_pos.shape == (2,)

            obs_cur = self.env.state_vector().copy()
            # pdb.set_trace() ## check current state
            
            ## <d4rl.pointmaze.maze_model.MazeEnv>
            self.env.set_target(target_location=gl_pos) ## used to change the target
            ## check target NOTE:
            # pdb.set_trace()




            rollout = [obs_cur.copy()]
            # start_state_list.append(obs_cur)
            target = self.env._target # tuple large: (7,9)
            ep_targets.append(target)
            
            # pdb.set_trace() ## check target
            
            assert len(target) == len(self.obs_select_dim)

            ## loop through timesteps
            total_reward = 0 ## accumulate reward of one episode
            # pdb.set_trace() ## large: 800, we also use 800?

            for t in range(self.env.max_episode_steps):
                ## redundant, same as obs_cur, for the janner's controller
                cur_state = self.env.state_vector().copy()
                assert (obs_cur == cur_state).all()
                assert len(cur_state) == 4, 'only for maze2D, for now.'

                if t == 0:
                    ## run diffusion model
                    ## set conditioning xy position to be the goal
                    input_st = cur_state[ self.obs_select_dim, ]

                    g_cond = {
                        ## (2, dim)
                        'st_gl': np.array( [input_st[None,], target[None,]], dtype=np.float32 )
                    }
                    # pdb.set_trace()
                    m_out = self.policy.gen_cond_stgl(g_cond=g_cond, b_s=self.b_size_per_prob)

                    pick_traj = m_out.pick_traj
                    # pdb.set_trace()

                    # actions = samples.actions[0] # B,H,dim(2)
                    # sequence = samples.observations[0] # B,H,dim(4)
                
                # position = obs_cur[0:2]
                # velocity = obs_cur[2:4]

                ## ----------- A Simple Controller ------------
                
                ## TODO: Oct 23 19:04, the above seems good, but too slow
                ## can use actions or define a simple controller based on state predictions
                if self.act_control == 'pd_ori':
                    raise NotImplementedError
                elif self.act_control == 'pred':
                    raise NotImplementedError ## can refer to other .py
                elif self.act_control == 'dfu_force': ## from diffusion forcing
                    ## pick_traj ## only have obs actually
                    ### ----- get the desired vel -----
                    if t == 0:
                        plan_vel = pick_traj[t, :2] - cur_state[:2]
                    elif t > 0 and t < len(pick_traj):
                        plan_vel = pick_traj[t, :2] - pick_traj[t - 1, :2]
                    else:
                        ## large than our traj
                        plan_vel = np.zeros_like(cur_state[2:])
                    ### ----- get the desired position -----
                    if t < len(pick_traj):
                        plan_pos = pick_traj[t,:2]
                    else:
                        plan_pos = pick_traj[-1,:2]
                        assert np.isclose(plan_pos, target, atol=0.09).all()
                    
                    ## PD controller: hyperparam ?
                    action = 12.5 * (plan_pos - cur_state[:2]) + 1.2 * (plan_vel - cur_state[2:])
                    action = np.clip(action, a_min=-1, a_max=1)
                    # pdb.set_trace() ## the first action should be zero


                ## ---------------------------

                # pdb.set_trace()
                
                ## by default, terminal is False forever
                ## np 1d (4,); float; bool (is finished, always False); 
                obs_next, rew, terminal, _ = self.env.step(action)

                is_suc = rew > 0 or is_suc ## sparse reward

                total_reward += rew
                score = self.env.get_normalized_score(total_reward) * 100

                if t % 50 == 0:
                    print(
                        f't: {t} | r: {rew:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                        f'pos: {obs_next[:2]} | vel: {obs_next[2:]} | action: {action} | '
                        f'Max Steps: {self.env.max_episode_steps}'
                    )

                ## update rollout observations
                rollout.append(obs_next.copy())

                obs_cur = obs_next
            

            ## ---------------------------------------------------
            ## ------------ Finished one eval episode ------------

            ep_pred_obss.append( pick_traj ) # shoule be unnormed, (384,4,)
            ep_rollouts.append(rollout)
            ep_titles_obs.append( f'PredObs: {i_ep}_score{int(score)}' )
            ep_titles_act.append( f'Act: {i_ep}_score{int(score)}' )

            ep_is_suc.append(is_suc)
           
            ep_scores.append(score)
            ep_total_rewards.append( total_reward )

            ## --- save multiple trajs in one large image ---
            if len(ep_pred_obss) % trajs_per_img == 0 or i_ep == num_ep - 1:
                ## the direct obs prediction
                tmp_st_idx = (i_ep // trajs_per_img) * trajs_per_img
                tmp_end_idx = tmp_st_idx + trajs_per_img # not inclusive
                tmp_tgts = np.array(ep_targets[tmp_st_idx:tmp_end_idx])
                tmp_tls_obs = ep_titles_obs[tmp_st_idx:tmp_end_idx]
                tmp_tls_act = ep_titles_act[tmp_st_idx:tmp_end_idx]

                tmp_scs = np.array(ep_scores[tmp_st_idx:tmp_end_idx])
                tmp_avg_sc = int(tmp_scs.mean())
                tmp_num_f = (tmp_scs < 100).sum() # not suc

                # pdb.set_trace()
                get_is_non_keypt = getattr(self.diffusion, 'get_is_non_keypt', None)

                if get_is_non_keypt is not None:
                    raise NotImplementedError
                    # is_non_keypt = get_is_non_keypt(b_size=trajs_per_img, idx_keypt=None, 
                        # n_comp=self.comp_diffusion.eval_num_cp_trajs)
                else:
                    is_non_keypt = None

                # pdb.set_trace()

                img_obs, rows_obs = self.renderer.composite(None, np.array(ep_pred_obss[tmp_st_idx:tmp_end_idx]), 
                                ncol=n_col, goal=tmp_tgts, titles=tmp_tls_obs, return_rows=True, is_non_keypt=is_non_keypt)

                img_act, rows_act = self.renderer.composite(None, np.array(ep_rollouts[tmp_st_idx:tmp_end_idx]), 
                                ncol=n_col, goal=tmp_tgts, titles=tmp_tls_act, return_rows=True, )
                
                f_path_3 = join(self.savepath, f'{tmp_st_idx}_act_obs_nns{tmp_num_f}_sc{tmp_avg_sc}.png')
                n_rows = len(rows_obs)
                img_whole = []
                ## cat (act,obs) pairs
                for i_r in range(n_rows):
                    img_whole.append( np.concatenate([rows_act[i_r], rows_obs[i_r]], axis=0) ) # 2H,W,C
                img_whole = np.concatenate(img_whole)
                utils.save_img(f_path_3, img_whole)


                

        
        ## ----------------------------------------------------------------
        ## ------------------ Finish All Eval Episodes --------------------

        
        utils.print_color(self.env.name,)

        ## Oct 21 New, metrics based on if success 
        ep_is_suc = np.array(ep_is_suc)
        ep_srate = ep_is_suc.mean() * 100
        ep_fail_idxs = np.where(ep_is_suc == False, )[0]
        assert len(ep_is_suc) == num_ep
        # pdb.set_trace()


        avg_ep_scores = np.mean(ep_scores)
        avg_ep_rewards = np.mean(ep_total_rewards)
        # print(f'{ep_scores=}')
        # print(f'{ep_total_rewards=}')
        print(f'{avg_ep_scores=}')
        print(f'{avg_ep_rewards=}')
        ## save result as a json file
        json_path = join(self.savepath, '00_rollout.json')

        sc_low_idxs = np.where(np.array(ep_scores) < self.score_low_limit)[0].tolist() # np -> list
        sc_low_idxs_d=dict(zip(sc_low_idxs, np.round(ep_scores, decimals=2)[sc_low_idxs]))
        print(f'{sc_low_idxs_d=}')

        # ep_range = range(1, len(ep_scores)+1)
        ep_range = range(len(ep_scores)) ## from 0
        json_data = OrderedDict([
            ('num_ep', num_ep),
            ('ep_srate', ep_srate), ## success rate
            ('avg_ep_scores', avg_ep_scores),
            ('avg_ep_rewards', avg_ep_rewards),
            ('pl_seed', pl_seed),
            # ('', ),
            ('epoch_diffusion', self.epoch,),
            ## ---- New Oct 23
            ('p_h5path', self.problems_h5path),
            ('var_temp', self.diffusion.var_temp),
            ('b_size_per_prob', self.b_size_per_prob),
            ('pol_config', self.pol_config),
            ('tj_blder_config', self.tj_blder_config),
            ('n_batch_acc_probs', self.n_batch_acc_probs),
            ('p_type', 'plan_once'),
            ('max_episode_steps', self.env.max_episode_steps),
            ('act_control', self.act_control),
            ## ----
            ('ep_fail_idxs', ep_fail_idxs.tolist()),
            ('sc_low_idx', sc_low_idxs_d),
            ##
            ('ep_is_suc', ep_is_suc.tolist()),
            ('ep_scores', dict(zip(ep_range, ep_scores)) ),
            ('ep_total_rewards', dict(zip(ep_range, ep_total_rewards)) ),
        ])
        # with open(json_path, 'w') as ff:
            # json.dump(json_data, ff, indent=2,) # sort_keys=True
        utils.save_json(json_data, json_path)

        # print(f'[save_plan_result]: save to {json_path}')
        new_savepath = f'{self.savepath.rstrip(os.sep)}-sc{int(avg_ep_scores)}/'
        utils.rename_fn(self.savepath, new_savepath)
        new_json_path = json_path.replace(self.savepath, new_savepath)
        utils.print_color(f'new_json_path: {new_json_path} \n',)
        
        return json_data
    



    


    
    def update_j_data(self, json_data: OrderedDict):
        """update the result data dict to be saved to a json"""
        json_data.update([
            ('epoch_diffusion', self.epoch,),
            ('cond_w', self.diffusion.condition_guidance_w,),
            ('use_ddim', self.diffusion.use_ddim),
            ('ddim_eta', self.diffusion.ddim_eta),
            ## ---- New Oct 23
            ('p_h5path', self.problems_h5path),
            ('var_temp', self.diffusion.var_temp),
            ('b_size_per_prob', self.b_size_per_prob),
            ('pol_config', self.pol_config),
            ('tj_blder_config', self.tj_blder_config),
            ('n_batch_acc_probs', self.n_batch_acc_probs),
            # ('p_type', 'plan_once_parallel'),
            ('max_episode_steps', self.env.max_episode_steps),
            ('act_control', self.act_control),
        ])
        return json_data