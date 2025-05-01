# from diffuser.guides.policies import Policy
from diffuser.guides.policies_invdyn import Policy_InvDyn
import diffuser.datasets as datasets
import diffuser.utils as utils
import numpy as np
from os.path import join
import pdb, json, imageio, os
from collections import OrderedDict
'''partially adapted from d4rl: scripts/reference_scores/maze2d_controller.py'''

class Maze2DEnvPlanner:
    def __init__(self, ) -> None:
        self.batch_size = 1
        # self.num_episodes = num_ep
        # self.pl_seed = None
        self.vis_trajs_per_img = 10
        self.score_low_limit = 100
        np.set_printoptions(precision=3, suppress=True)
        self.act_control = 'pd_ori' # 'pred'


    def setup_load(self, args, args_train):
        """
        used in a separate launch evaluation, where model should be loaded from file
        """
        self.env = datasets.load_environment(args.dataset)
        self.env.seed(0) ## default
        
        ## loading the dataset is too slow...
        diffusion_experiment = utils.load_diffusion(args.logbase, args_train.dataset, 
                    args_train.exp_name, epoch=args.diffusion_epoch)

        self.diffusion = diffusion_experiment.ema ## should an ema model
        self.dataset = diffusion_experiment.dataset
        self.renderer = diffusion_experiment.renderer
        if self.diffusion.is_inv_dyn_dfu:
            self.policy = Policy_InvDyn(self.diffusion, self.dataset.normalizer)
        else:
            self.policy = Policy(self.diffusion, self.dataset.normalizer)
        self.savepath = args.savepath
        self.epoch = diffusion_experiment.epoch
        self.pl_hzn = self.diffusion.horizon

        self.setup_general()

    def setup_given(self, env):
        """
        used when everything is directly prepared and given
        """
        self.env = env
        self.env.seed(0) ## default
        ## TODO:
        

        self.setup_general()

    def setup_general(self):
        '''general stuff for both types of setup'''
        utils.mkdir(self.savepath)
        self.savepath_root = self.savepath

    def plan_multi_run(self, seeds, **kwargs):
        if seeds in (None, []) :
            seeds = [None,]
        results_sd = []
        for i_s, sd in enumerate(seeds):
            ## important: set up env seed again
            self.env.seed(0)

            self.savepath = f'{self.savepath_root}/run-{i_s}/'
            utils.mkdir(self.savepath)
            out = self.plan_once(pl_seed=sd, **kwargs)
            results_sd.append(out)

        ## compute the avg results
        sc_all = []
        sc_pairs = []
        for i_s, out in enumerate(results_sd):
            sc_all.append( out['avg_ep_scores'] )
            sc_pairs.append( (seeds[i_s], out['avg_ep_scores']) )
        sc_all = np.array(sc_all)
        sc_mean = sc_all.mean()
        ep_str = f'{int(self.epoch // 10000)}w'
        json_path = join(self.savepath_root, f'00_runs_summary_sc{int(sc_mean)}_ep{ep_str}.json')
        json_data = OrderedDict([
            ('seeds', seeds),
            ('scores_mean', sc_all.mean().round(decimals=4) ),
            ('scores_std', sc_all.std().round(decimals=4) ),
            ('sc_pairs', sc_pairs),
            ('epoch', self.epoch)
        ])
        print(json_data)
        ## save mean and std
        utils.save_json(json_data, json_path)
        utils.rename_fn(self.savepath_root, f'{self.savepath_root.rstrip(os.sep)}-sc{int(sc_mean)}')
        

        return json_data
        

    
    def plan_once(self, num_ep, pl_seed=None, given_starts=None):
        '''
        code to launch planning
        given_starts: np (B, 4) the start location is directly given
        '''
        if pl_seed is not None:
            ## seed everything
            utils.set_seed(pl_seed)
        if given_starts is not None: # if given, just evaluate the given states
            num_ep = len(given_starts)
        ep_scores = []
        ep_total_rewards = []
        ep_pred_obss = []
        ep_rollouts = []
        ep_targets = []
        ep_titles_obs, ep_titles_act = [], []
        trajs_per_img = min(self.vis_trajs_per_img, num_ep)
        n_col = min(5, trajs_per_img)
        # pdb.set_trace()


        for i_ep in range(num_ep):
            
            obs_cur = self.env.reset()
            if given_starts is not None: ## set to given value
                ## args must be np
                self.env.set_state(qpos=given_starts[i_ep, :2], qvel=given_starts[i_ep, 2:])
            obs_cur = self.env.state_vector().copy()

            rollout = [obs_cur.copy()]
            # start_state_list.append(obs_cur)
            # env.set_target() ## used to change the target
            target = self.env._target # tuple large: (7,9)
            ep_targets.append(target)
            # pdb.set_trace()
            cond = {
                ## TODO: dimension is 1D only? But looks can run
                ## goal position+vel
                self.pl_hzn - 1: np.array([*target, 0, 0]),
            }
            self.policy.pl_hzn = self.pl_hzn
            
            utils.print_color(f'{self.policy.pl_hzn=}')

            ## loop through timesteps
            total_reward = 0 ## accumulate reward of one episode
            # pdb.set_trace() ## large: 800
            for t in range(self.env.max_episode_steps):
                ## same as obs_cur?
                state = self.env.state_vector().copy()

                if t == 0:
                    ## run diffusion model
                    ## set conditioning xy position to be the goal
                    cond[0] = obs_cur
                    action, samples = self.policy(cond, batch_size=self.batch_size)

                    actions = samples.actions[0] # B,H,dim(2)
                    sequence = samples.observations[0] # B,H,dim(4)
                
                # position = obs_cur[0:2]
                # velocity = obs_cur[2:4]

                ## ----------- A Simple Controller ------------
                if t < len(sequence) - 1:
                    next_waypoint = sequence[t+1]
                else:
                    next_waypoint = sequence[-1].copy()
                    ## (goal_x, goal_y, 0, 0)
                    next_waypoint[2:] = 0
                    # pdb.set_trace()

                ## --- d4rl ori code ----
                # act, done = controller.get_action(position, velocity, env.get_target())
                ## ----------------------
                
                ## can use actions or define a simple controller based on state predictions
                if self.act_control == 'pd_ori':
                    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                elif self.act_control == 'pred':
                    if t < len(actions):
                        action = actions[t]
                    else: ## back to pd
                        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])


                ## ---------------------------
                
                ## by default, terminal is False forever
                ## np 1d (4,); float; bool (is finished, always False); 
                obs_next, rew, terminal, _ = self.env.step(action)

                total_reward += rew
                score = self.env.get_normalized_score(total_reward) * 100

                if t % 50 == 0:
                    print(
                        f't: {t} | r: {rew:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                        f'pos: {obs_next[:2]} | vel: {obs_next[2:]} | action: {action}'
                    )

                ## update rollout observations
                rollout.append(obs_next.copy())

                obs_cur = obs_next
            

            ## ---------------------------------------------------
            ## ------------ Finished one eval episode ------------

            ep_pred_obss.append(samples.observations[0]) # 1,384,4
            ep_rollouts.append(rollout)
            ep_titles_obs.append( f'PredObs: {i_ep}_score{int(score)}' )
            ep_titles_act.append( f'Act: {i_ep}_score{int(score)}' )

           
            ep_scores.append(score)
            ep_total_rewards.append( total_reward )

            ## --- save multiple trajs in one large image ---
            if len(ep_pred_obss) % trajs_per_img == 0:
                ## the direct obs prediction
                tmp_st_idx = (i_ep // trajs_per_img) * trajs_per_img
                tmp_end_idx = tmp_st_idx + trajs_per_img # not inclusive
                tmp_tgts = np.array(ep_targets[tmp_st_idx:tmp_end_idx])
                tmp_tls_obs = ep_titles_obs[tmp_st_idx:tmp_end_idx]
                tmp_tls_act = ep_titles_act[tmp_st_idx:tmp_end_idx]

                tmp_scs = np.array(ep_scores[tmp_st_idx:tmp_end_idx])
                tmp_avg_sc = int(tmp_scs.mean())
                tmp_num_f = (tmp_scs < 100).sum() # not suc

                
                get_is_non_keypt = getattr(self.diffusion, 'get_is_non_keypt', None)
                if get_is_non_keypt is not None:
                    is_non_keypt = get_is_non_keypt(b_size=trajs_per_img, idx_keypt=None)
                else:
                    is_non_keypt = None

                
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
            ('avg_ep_scores', avg_ep_scores),
            ('avg_ep_rewards', avg_ep_rewards),
            ('pl_seed', pl_seed),
            # ('', ),
            ('epoch_diffusion', self.epoch,),
            ('sc_low_idx', sc_low_idxs_d),
            ##
            ('ep_scores', dict(zip(ep_range, ep_scores)) ),
            ('ep_total_rewards', dict(zip(ep_range, ep_total_rewards)) ),
        ])
        # with open(json_path, 'w') as ff:
            # json.dump(json_data, ff, indent=2,) # sort_keys=True
        utils.save_json(json_data, json_path)

        # print(f'[save_plan_result]: save to {json_path}')
        utils.rename_fn(self.savepath, f'{self.savepath.rstrip(os.sep)}-sc{int(avg_ep_scores)}')
        
        return json_data


