import sys; sys.path.append('./')
import pdb, torch, copy, pdb, json
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)
##
import numpy as np
from os.path import join
from datetime import datetime
import os.path as osp
import diffuser.utils as utils
from diffuser.models.cd_stgl_sml_dfu.stgl_sml_planner_v1 import Stgl_Sml_Maze2DEnvPlanner_V1



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
    ld_config = dict(
        # dfu_model_loadpath="logs/maze2d-large-v1/diffusion/m2d_lg_cpV1_Trv2_bs32_Px0_T256/",
    )

    m2d_planner = Stgl_Sml_Maze2DEnvPlanner_V1(args_train, args=args)
    m2d_planner.setup_load( ld_config=ld_config )

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
    from diffuser.datasets.d4rl import Is_Gym_Robot_Env
    if len(pl_seeds) == 1:
        ## plan_n_ep
        if Is_Gym_Robot_Env:
            if pl_seeds[0] == -1: ## no seed
                avg_result_dict = m2d_planner.ben_plan_once_parallel(pl_seed=None,)
            else:
                avg_result_dict = m2d_planner.ben_plan_once_parallel(pl_seed=pl_seeds[0])
        else:
            if pl_seeds[0] == -1: ## no seed
                # avg_result_dict = m2d_planner.plan_once(pl_seed=None,)
                avg_result_dict = m2d_planner.plan_once_parallel(pl_seed=None,)
            else:
                # avg_result_dict = m2d_planner.plan_once(pl_seed=pl_seeds[0])
                avg_result_dict = m2d_planner.plan_once_parallel(pl_seed=pl_seeds[0])
    else:
        utils.print_color(f'{args.pl_seeds=}')
        raise NotImplementedError
        avg_result_dict = m2d_planner.plan_multi_run(seeds, num_ep=args.plan_n_ep, 
                                given_starts=given_starts)




    
    return avg_result_dict


if __name__ == '__main__':
    ## training args
    args_train = Parser().parse_args('diffusion')
    args = Parser().parse_args('plan')
    ## 1. get epoch to eval on, by default all
    loadpath = args.logbase, args.dataset, args_train.exp_name

    args.pl_seeds = utils.parse_seeds_str(args.pl_seeds) ## a list of int
    args.n_batch_acc_probs = 10 ## A5000: 20=2.23it/s, 10=4.10it/s
    
    # pdb.set_trace()
    args.use_ddim = False
    # args.use_ddim = True

    ### --- Hyper-parameters Setup ---
    from diffuser.datasets.d4rl import Is_Gym_Robot_Env
    if Is_Gym_Robot_Env: ## Ben
        if '-large-' in args_train.dataset:
            # args.ev_n_comp = 5 # ben large
            args.ev_pl_hzn = 736  ## aka ncp=5: 192 + (192 - 56) * 4
            args.env_n_max_steps = 1000 ## ben large
        elif '-medium-' in args_train.dataset:
            ## ncp: 5 or 6
            args.ev_pl_hzn = 528 ## aka ncp=5: 144 + (144-48) * 4
            args.env_n_max_steps = 1000 ## ben
        elif '-umaze-' in args_train.dataset:
            # args.ev_n_comp = 5 # umaze h is only 40
            ## TODO: From Here Jan 19: 1:00 AM
            # args.ddim_eta = 1.0
            # args.ddim_eta = 0.0
            args.ev_pl_hzn = 136 ## aka ncp=5
            args.ev_pl_hzn = 160 ## aka ncp=6
            # args.ev_pl_hzn = 40 ## luotest
            args.env_n_max_steps = 1000 #
        

    else:
        assert False
        args.ev_n_comp = 4
        args.env_n_max_steps = 600
    
    
    args.b_size_per_prob = 1
    # args.b_size_per_prob = 40
    # args.ev_top_n = 5
    # args.ev_pick_type = 'first'
    # args.tjb_blend_type = 'exp'
    # args.tjb_exp_beta = 2

    args.var_temp = 0.5
    # args.var_temp = 1.0
    args.cond_w = 2.0

    latest_e = utils.get_latest_epoch(loadpath)
    # n_e = round(latest_e // 1e5) + 1 # all
    # start_e = 5e5; # 2e5 end_e = 
    # depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()
    
    depoch_list = [latest_e,]
    # depoch_list = [800000,] # 1M


    # sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S")}-nm{int(args.plan_n_ep)}'
    sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S-%f")[:-3]}' + \
                        f"-nm{int(args.plan_n_ep)}-phzn{args.ev_pl_hzn}" + \
                        f"-ems{args.env_n_max_steps}" + \
                        f"-evSd{','.join( [str(sd) for sd in args.pl_seeds] )}"
    # pdb.set_trace()
    ## f'-vt{args.var_temp}'
    ## TODO:
    args.is_vis_single = True
    if args.is_vis_single:
        sub_dir += '-vis'

    args.savepath = osp.join(args.savepath, sub_dir)

    result_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main( copy.deepcopy(args_train),  copy.deepcopy(args) )
        
        result_list.append(tmp)
    

