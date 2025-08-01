import sys, os; sys.path.append('./')
os.environ['PYOPENGL_PLATFORM'] = 'egl' ## enable GPU rendering in mujoco
os.environ['MUJOCO_GL'] = 'egl'
import pdb, torch, copy, pdb, json
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)
##
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from datetime import datetime
import os.path as osp
import diffuser.utils as utils
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import OgB_Stgl_Sml_MazeEnvPlanner_V1




class Parser(utils.Parser):
    dataset: str = None
    config: str = None
    ## should not put any existing variable in config file here
    pl_seeds: str = '-1' # no seed
    plan_n_ep: int = -100 ## all if -100, auto parse to int

def main(args_train, args):
    
    #---------------------------------- setup ----------------------------------#

    ld_config = dict()

    ogmz_planner = OgB_Stgl_Sml_MazeEnvPlanner_V1(args_train, args=args)
    ogmz_planner.setup_load( ld_config=ld_config )

    #---------------------------- start planning -----------------------------#

    pl_seeds = args.pl_seeds

    from diffuser.datasets.d4rl import Is_OgB_Robot_Env
    if len(pl_seeds) == 1:
        ## plan_n_ep
        if Is_OgB_Robot_Env:
            if pl_seeds[0] == -1: ## no seed
                avg_result_dict = ogmz_planner.ogb_plan_once(pl_seed=None,)
            else:
                avg_result_dict = ogmz_planner.ogb_plan_once(pl_seed=pl_seeds[0])
        
    else:
        utils.print_color(f'{args.pl_seeds=}')
        raise NotImplementedError ## can impl plan_multi_run

    ## might prevent the final exception before the program finishes
    ogmz_planner.env.close()
    del ogmz_planner.env
    del ogmz_planner.renderer.env
    
    return avg_result_dict


if __name__ == '__main__':
    ## training args
    args_train = Parser().parse_args('diffusion')
    args = Parser().parse_args('plan')
    ## 1. get epoch to eval on, by default all
    loadpath = args.logbase, args.dataset, args_train.exp_name

    args.pl_seeds = utils.parse_seeds_str(args.pl_seeds) ## a list of int
    args.n_batch_acc_probs = 4 ##
    
    ### --- Hyper-parameters Setup ---
    from diffuser.datasets.d4rl import Is_OgB_Robot_Env
    assert Is_OgB_Robot_Env
    
    

    ## Default
    args.is_replan = None ## placeholder, should be replaced in the if code blcok below
    args.n_act_per_waypnt = 2
    args.is_save_pkl = False
    args.is_rd_agv = False

    ## the state dimension used in the diffusion models
    dfu_ndim = len(args_train.dataset_config['obs_select_dim'])


    ## ---------------------------------------
    ## ----------- Ant Maze Stitch -----------
    if 'antmaze' in args.dataset.lower() and 'stitch' in  args.dataset.lower():
        if 'giant' in args.dataset:

            ## Ant Maze Giant
            if dfu_ndim == 2:
                ## NOTE: Set the eval start idx, by default starts from 0
                args.ep_st_idx = 0
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ## number of actions per waypoint
                # args.ev_cp_infer_t_type = 'same_t' ## ours, faster
                # args.ev_cp_infer_t_type = 'gsc' ## baselines
                
                args.ev_cp_infer_t_type = 'interleave' ## ours
                args.rd_resol = 300 ## website default: 1000
                # args.is_save_pkl = True ## save the plan/rollout trajs to a pkl file

                args.ev_n_comp = 9

                ## high resolution, etc., other eval/render hyperparameters
                # args.rd_resol = 1600
                # args.ep_st_idx = 20
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = False
                # args.vid_fps = 60
                ## --------------------------------

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## 0: no replan
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ##
                    n_max_steps=2000, ## 
                )
                args.inv_epoch = int(8e5)
            
            ## Ant Maze Giant Higher Dim
            elif dfu_ndim == 15:
                args.ev_n_comp = 9
                repl_wp_cfg = {}

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.ev_cp_infer_t_type = 'interleave'

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=15,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150, ##
                    n_max_steps=2000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
            ## Ant Maze Giant
            elif dfu_ndim == 29:
                args.ev_n_comp = 9 # 8
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ## important:
                args.ev_cp_infer_t_type = 'interleave'

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=15, ##
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150, ## 100?
                    n_max_steps=2000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
        
        ## Ant Maze Large Stitch
        elif 'large' in args.dataset:
            if dfu_ndim == 29:
                
                args.ev_cp_infer_t_type = 'interleave' ## gsc / same_t (parallel)
                args.rd_resol = 300 # 1000
                # args.is_save_pkl = True


                repl_wp_cfg = {}
                args.ev_n_comp = 5 ## 6,7 is also fine
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4, ## 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 100?
                    n_max_steps=1000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)


            ## Ant Maze Large Stitch
            elif dfu_ndim == 15:
                ## ----------------
                args.ev_cp_infer_t_type = 'interleave'

                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4, 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 100?
                    n_max_steps=1000, ## 
                    used_idxs=(0,1),
                )
                # pdb.set_trace()
                args.inv_epoch = int(8e5)

            ## Ant Maze Large Stitch
            elif dfu_ndim == 2:
                repl_wp_cfg = {}
                
                args.ev_n_comp = 6 ##
                args.ev_cp_infer_t_type = 'interleave'

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4, 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 
                    n_max_steps=1000, ## 
                )

                args.inv_epoch = int(8e5)


        ## Ant Maze Medium
        elif 'medium' in args.dataset:
            if dfu_ndim == 29:

                ## ---------------------
                repl_wp_cfg = {}
                args.ev_n_comp = 3 ##
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                # pdb.set_trace()
                args.inv_epoch = int(8e5)

            ## Ant Maze Medium
            elif dfu_ndim == 15:
                repl_wp_cfg = {}
                args.ev_n_comp = 3 ## 4
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                ## -- failure analysis vis --
                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.is_rd_agv = True
                args.is_use_subgoal_marker = True
                args.vid_fps = 60
                ## --------------------------


                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)

            
            ## Ant Maze Medium
            elif dfu_ndim == 2:

                args.ev_n_comp = 3

                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = True ## False
                # args.vid_fps = 60

                # args.ep_st_idx = 40
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)

    
    ## ----------------------------------------------------------------
    ## ---------------------- AntMaze Explore -------------------------
    ## ----------------------------------------------------------------
    
    ## Ant Maze Explore
    elif 'explore' in  args.dataset.lower() and 'antmaze' in args.dataset.lower():
        ## Explore Large
        if 'large' in args.dataset:
            if dfu_ndim in [29, 15]:
                args.ev_cp_infer_t_type = 'interleave'
                repl_wp_cfg = {}
                args.ev_n_comp = 10
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=2,
                    ada_dist_minus_n_wp=0,
                    type='m_2',
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)

            ## Explore Large
            elif dfu_ndim == 2:
                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc', 'same_t_p'
                # args.rd_resol = 400
                # args.is_save_pkl = True

                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0,
                    thres=2, ##
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)

            else:
                raise NotImplementedError
        
        ## Explore Medium
        elif 'medium' in args.dataset:
            if dfu_ndim in [29,15]:
                repl_wp_cfg = {} 
                args.ev_n_comp = 5 ## 4

                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 1000
                args.is_save_pkl = True

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=2, ##
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000, ## 2000
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
            
            ## Explore Medium
            elif dfu_ndim == 2:
                repl_wp_cfg = {}
                args.ev_n_comp = 5 ## Used

                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc', 'same_t_p'
                args.rd_resol = 1000
                args.is_save_pkl = True

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## replan
                    thres=2,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)


    ## ----------------------------------------------------------------
    ## ---------------------- Humanoid Stitch -------------------------
    ## ----------------------------------------------------------------

    ## Only Implemented 2D For Now
    elif 'humanoid' in args.dataset.lower():
        if 'giant' in args.dataset:
            args.ev_n_comp = 11
            args.inv_epoch = int(16e5)

            args.ev_cp_infer_t_type = 'interleave'
            args.rd_resol = 1000
            args.is_save_pkl = True
            args.ep_st_idx = 80

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1 ## important:
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                thres=10,
                type='m_2',
                ada_dist_minus_n_wp=300,
                # ada_dist_minus_n_wp=400, ## TODO:
                cond_2_extra=150, ## 100?
                n_max_steps=8000, ## NEW Jan 10
            )

        ## Humanoid
        elif 'large' in args.dataset:
            if dfu_ndim == 23:
                raise NotImplementedError
            elif dfu_ndim == 2:
                ## Humanoid Large
                args.ev_n_comp = 6
                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 400 ## or 1000
                # args.is_save_pkl = True
                # args.ep_st_idx = 80

                ## --------------------
                # args.ev_n_comp = 5 # 5
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = False
                # args.vid_fps = 120 ## humanoid
                # args.ep_st_idx = 60
                ## -------------------

                args.is_replan = 'ada_dist'
                repl_wp_cfg = {}
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=10,
                    type='m_2',
                    ada_dist_minus_n_wp=300,
                    cond_2_extra=150,
                    n_max_steps=5000, ## default for eval
                )
                args.inv_epoch = int(8e5)
        
        ## Humanoid Medium
        elif 'medium' in args.dataset:
            if dfu_ndim == 23:
                raise NotImplementedError
            elif dfu_ndim == 2:
                ## Humanoid Medium
                args.ev_n_comp = 4

                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.ep_st_idx = 80

                args.is_replan = 'ada_dist'
                repl_wp_cfg = {}
                
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=10,
                    type='m_2',
                    ada_dist_minus_n_wp=300,
                    cond_2_extra=150,
                    n_max_steps=5000,
                )
            args.inv_epoch = int(8e5)
        else:
            raise NotImplementedError

    elif 'antsoccer' in args.dataset:
        ## Soccer
        if 'arena' in args.dataset:
            ## a 4D diffusion planner: ant x-y, ball x-y 
            if dfu_ndim == 4:
                repl_wp_cfg = {}
                args.ev_n_comp = 5

                args.ev_cp_infer_t_type = 'interleave'
                args.is_use_subgoal_marker = False
                args.ep_st_idx = 60
                ## teaser animation vis
                # args.rd_resol = 1200
                # args.is_rd_agv = True ## render agent view video
                # args.vid_fps = 60

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ##
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50, ## 
                    n_max_steps=5000, ##
                    used_idxs=(0,1),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

            elif dfu_ndim == 17:
                repl_wp_cfg = {}
                args.ev_n_comp = 5

                args.ev_cp_infer_t_type = 'interleave'
                # args.is_use_subgoal_marker = False
                
                args.ep_st_idx = 60
                ## teaser animation vis
                # args.rd_resol = 1200
                # args.is_rd_agv = True ## render agent view video
                # args.vid_fps = 60

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50, ## 
                    n_max_steps=5000, ## ori default
                    used_idxs=(0,1),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

        ## Soccer
        elif 'medium' in args.dataset:
            if dfu_ndim == 17:
                
                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.ep_st_idx = 0 ## 20
                args.is_replan = 'ada_dist'
                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 600
                args.is_save_pkl = True
                args.is_use_subgoal_marker = False


                args.n_act_per_waypnt = 2 ## 1 vis

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=6,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=10,
                    n_max_steps=5000,
                    used_idxs=(0,1,),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

            ## ---------------------------------
            ## Soccer Medium
            elif dfu_ndim == 4:
                
                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 1000 # 600 # 1000
                args.is_save_pkl = True
                args.is_use_subgoal_marker = False
                ## for website
                args.ep_st_idx = 20
                args.is_rd_agv = True


                repl_wp_cfg = {}
                ## Jan 10 New Replan Method
                args.ev_n_comp = 8
                args.ev_n_comp = 5
                
                args.ev_n_comp = 6 ## Jan 17
                args.ev_n_comp = 7
                args.ev_n_comp = 8

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 2
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50, ## ?
                    cond_2_extra=50, ## 100?
                    n_max_steps=5000,
                    used_idxs=(0,1,),
                )

                # args.inv_epoch = int(12e5)
                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

    ## OGBench Point Maze Env
    elif 'pointmaze' in args.dataset.lower():
        
        ## Point Maze Giant
        if 'giant' in args.dataset:
            # args.ev_n_comp = 9
            args.ev_n_comp = 8
            
            # args.ev_cp_infer_t_type = 'same_t' ## parallel
            args.ev_cp_infer_t_type = 'interleave' ## default
            # args.ev_cp_infer_t_type = 'gsc' ## gsc baseline
            # args.ev_cp_infer_t_type = 'same_t_p'
            # args.ev_cp_infer_t_type = 'ar_back' ## backward autoregressive

            # args.rd_resol = 1000
            # args.is_save_pkl = True ## save rollout stats
            
            # args.ep_st_idx = 40 ## starting eval problem idx, default is 0

            ## 
            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                # max_n_repl=0, ## no repl abl
                thres=1,
                type='m_2',
                ada_dist_minus_n_wp=10,
                cond_2_extra=150, ##
                n_max_steps=1000, ## 
            )
            args.inv_epoch = int(8e5)

        elif 'large' in args.dataset:
            args.ev_n_comp = 5 ##
            args.ev_n_comp = 6 ## either is fine

            args.ev_cp_infer_t_type = 'interleave'

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no replan
                thres=1,
                type='m_2',
                ada_dist_minus_n_wp=0,
                cond_2_extra=150, ##
                n_max_steps=1000, ## 
            )
            args.inv_epoch = int(8e5)
            

        elif 'medium' in args.dataset:
            args.ev_n_comp = 3 ## or 4

            args.ev_cp_infer_t_type = 'interleave' ## or 'same_t_p'

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no replan ablation
                thres=1, ## 4
                type='m_2',
                ada_dist_minus_n_wp=0,
                cond_2_extra=150,
                n_max_steps=1000,
            )
            args.inv_epoch = int(8e5)

    else: 
        raise NotImplementedError
    
    ## 
    args.b_size_per_prob = 40
    args.ev_top_n = 5
    args.ev_pick_type = 'first'
    args.tjb_blend_type = 'exp'
    args.tjb_exp_beta = 2


    ### Diffusion Sampling Hyper-param
    args.var_temp = 1.0 ## or 0.5
    args.cond_w = 2.0
    args.use_ddim = True
    args.ddim_eta = 1.0
    args.ddim_steps = 50
    
    args.repl_wp_cfg = repl_wp_cfg

    latest_e = utils.get_latest_epoch(loadpath)
    # n_e = round(latest_e // 1e5) + 1 # all
    # start_e = 5e5; # 2e5 end_e = 
    # depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()
    
    depoch_list = [latest_e,]
    ## depoch_list = [800000,] # 1M
    
    if args.is_replan == 'ada_dist':
        args.env_n_max_steps = args.repl_ada_dist_cfg['n_max_steps']
    else:
        args.env_n_max_steps = None ## use ogb default ??


    sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S-%f")[:-3]}' + \
                        f"-nm{int(args.plan_n_ep)}-ems{args.env_n_max_steps//1000}k" + \
                        f"-ncp{args.ev_n_comp}" + f"-{args.ev_cp_infer_t_type}"\
                        f"-evSd{','.join( [str(sd) for sd in args.pl_seeds] )}"
    
    # pdb.set_trace()
    ## f'-vt{args.var_temp}'
    if args.is_save_pkl:
        sub_dir += '-pkl'
    if hasattr(args, 'ep_st_idx'):
        sub_dir += f'-st{args.ep_st_idx}'
    if args.is_rd_agv:
        sub_dir += '-agv'

    args.savepath = osp.join(args.savepath, sub_dir)

    result_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main( copy.deepcopy(args_train),  copy.deepcopy(args) )
        
        result_list.append(tmp)
    

