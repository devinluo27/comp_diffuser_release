import sys, os; sys.path.append('./')
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
import pdb, torch, copy, pdb, json
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.benchmark = False ## Jan 12
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
    ## should not put any existing var in config here
    pl_seeds: str = '-1' # no seed
    # ev_n_comp: int = 2
    plan_n_ep: int = -100 ## all if -100, auto parse to int
    # ev_n_mcmc: int = 10
    # var_temp = 0.5

def main(args_train, args):
    
    #---------------------------------- setup ----------------------------------#
    # TODO:
    ld_config = dict()

    ogmz_planner = OgB_Stgl_Sml_MazeEnvPlanner_V1(args_train, args=args)
    ogmz_planner.setup_load( ld_config=ld_config )

    # pdb.set_trace()

    #---------------------------- start planning -----------------------------#

    # seeds = None if args.pl_seeds == -1 else list(range(args.pl_seeds))
    # avg_result_dict = m2d_planner.plan_multi_run(seeds, num_ep=args.plan_n_ep)
    ##---
    # given_starts = np.array([[5, 6], [5, 6.5],
    #                          [5, 7], [5, 7.5], 
    #                          [5, 8], [5, 8.5]], dtype=np.float32)
    ##---
    pl_seeds = args.pl_seeds
    ## Oct 30
    from diffuser.datasets.d4rl import Is_OgB_Robot_Env
    if len(pl_seeds) == 1:
        ## plan_n_ep
        if Is_OgB_Robot_Env:
            if pl_seeds[0] == -1: ## no seed
                # avg_result_dict = ogmz_planner.ogb_plan_once_parallel(pl_seed=None,)
                avg_result_dict = ogmz_planner.ogb_plan_once(pl_seed=None,)
            else:
                # avg_result_dict = ogmz_planner.ogb_plan_once_parallel(pl_seed=pl_seeds[0])
                avg_result_dict = ogmz_planner.ogb_plan_once(pl_seed=pl_seeds[0])
        
    else:
        utils.print_color(f'{args.pl_seeds=}')
        raise NotImplementedError ## can impl plan_multi_run

    ## TODO: Check, prevent the final exception, seems useless
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
    
    # pdb.set_trace()
    ### --- Hyper-parameters Setup ---
    from diffuser.datasets.d4rl import Is_OgB_Robot_Env
    assert Is_OgB_Robot_Env
    
    

    ## Default
    args.is_replan = 'at_given_t' ## True
    args.n_act_per_waypnt = 2
    args.is_save_pkl = False

    
    dfu_ndim = len(args_train.dataset_config['obs_select_dim'])

    
    if 'pointmaze' in args.dataset.lower():
        
        ## Point Maze Giant
        if 'giant' in args.dataset:
            repl_wp_cfg = {}
            
            args.ev_cp_infer_t_type = 'dd'
            # pdb.set_trace()
            # args.rd_resol = 1000
            # args.is_save_pkl = True
            args.ev_pl_hzn = 888 ## ncp=8
            args.ev_pl_hzn = 992 ## ncp=9
            

            # args.ep_st_idx = 40
            ## Jan 10 New Replan Method
            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1 ## important:
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no repl abl
                # thres=4,
                thres=1,
                type='m_2',
                # ada_dist_minus_n_wp=0,
                # ada_dist_minus_n_wp=50,
                ada_dist_minus_n_wp=10,
                cond_2_extra=150, ## 100?
                n_max_steps=1000, ## 
                # n_max_steps=2000, ## 
            )
            args.inv_epoch = int(8e5)
            
            # pdb.set_trace()

        elif 'large' in args.dataset:
            repl_wp_cfg = {}

            args.ev_cp_infer_t_type = 'dd'
            args.ev_pl_hzn = 576 ## ncp=5
            args.ev_pl_hzn = 680 ## ncp=6

            # args.rd_resol = 1000
            # args.is_save_pkl = True


            ## Jan 10 New Replan Method
            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1 ## important:
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no replan ablation ## TODO: check back Jan 24, do we actually run this?
                thres=4,
                # thres=1,
                type='m_2',
                ada_dist_minus_n_wp=0,
                # ada_dist_minus_n_wp=50,
                # ada_dist_minus_n_wp=10,
                cond_2_extra=150, ## 100?
                n_max_steps=1000, ## 
                # n_max_steps=2000, ## 
            )
            args.inv_epoch = int(8e5)
            
            # pdb.set_trace()

        elif 'medium' in args.dataset:
            repl_wp_cfg = {}

            # args.rd_resol = 1000
            # args.is_save_pkl = True

            args.ev_cp_infer_t_type = 'dd' ## Feb 15
            args.ev_pl_hzn = 472 ## ncp=4
            args.ev_pl_hzn = 368 ## ncp=3

            ## Jan 10 New Replan Method
            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1 ## important:
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no replan ablation
                thres=4,
                # thres=1,
                type='m_2',
                ada_dist_minus_n_wp=0,
                # ada_dist_minus_n_wp=50,
                # ada_dist_minus_n_wp=10,
                cond_2_extra=150, ## 100?
                n_max_steps=1000, ## 
                # n_max_steps=2000, ## 
            )
            args.inv_epoch = int(8e5)

    elif 'antmaze' in args.dataset.lower():
        ## Ant Maze Giant
        if 'giant' in args.dataset:
            repl_wp_cfg = {}
            
            args.ev_cp_infer_t_type = 'dd'
            # pdb.set_trace()
            # args.rd_resol = 1000
            # args.is_save_pkl = True
            args.ev_pl_hzn = 888 ## ncp=8
            args.ev_pl_hzn = 992 ## ncp=9
            

            # args.ep_st_idx = 40
            ## Jan 10 New Replan Method
            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1 ## important:
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no repl abl
                # thres=4,
                thres=1,
                type='m_2',
                # ada_dist_minus_n_wp=0,
                # ada_dist_minus_n_wp=50,
                ada_dist_minus_n_wp=10,
                cond_2_extra=150, ## 100?
                n_max_steps=1000, ## 
                # n_max_steps=2000, ## 
            )
            args.inv_epoch = int(8e5)

    else: 
        raise NotImplementedError
    
    
    
    ## TODO: check eval hyper-param
    args.ev_n_comp = 1
    args.b_size_per_prob = 40 # 40 # 20


    ### Update
    args.var_temp = 1.0 # 0.5
    args.cond_w = 2.0
    args.use_ddim = True

    args.ddim_eta = 1.0 ## or 0.0
    # args.ddim_steps = 10 ## for debug
    args.ddim_steps = 50
    ## env rollout dynamics

    ## 160 - 56 = 104
    
    args.repl_wp_cfg = repl_wp_cfg


    latest_e = utils.get_latest_epoch(loadpath)
    # n_e = round(latest_e // 1e5) + 1 # all
    # start_e = 5e5; # 2e5 end_e = 
    # depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()
    
    depoch_list = [latest_e,] ## which checkpoint to load
    ## the number here is not accurate
    # depoch_list = [800000,] # load the checkpoint trained with acutally 1200000 iterations
    
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

    args.savepath = osp.join(args.savepath, sub_dir)

    result_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main( copy.deepcopy(args_train),  copy.deepcopy(args) )
        
        result_list.append(tmp)
    

