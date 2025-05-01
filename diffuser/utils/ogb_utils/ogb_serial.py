import os, h5py, pdb
from diffuser.utils.serialization import load_config, \
    RandomNumberDataset, get_latest_epoch, DiffusionExperiment
import numpy as np
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer
from diffuser.datasets.ogb_dset import *


def load_ogb_maze_datasetNormalizer(args_train, obs_dim_idxs=None):
    '''
    Directly create a normalizer with given value, so no need to load the dataset, which is slow.
    Returns:
        normalizer: a class
    '''
    assert type(args_train.normalizer) == str
    normalizer = eval(args_train.normalizer)
    
    # ------------------- load from abc, can be extracted -------------------
    data_dict = {}
    dset_type = args_train.dataset_config['dset_type']
    assert dset_type == 'ogb'
    dset_name = args_train.dataset

    # pdb.set_trace()

    if dset_name == "antmaze-giant-stitch-v0":
        ## ant: [2, 29]; [2, 8]
        d_obs_mm = OgB_AntMaze_Giant_Stitch_Obs__Min_Max
        d_act_mm = OgB_AntMaze_Giant_Act__Min_Max
    elif dset_name == "antmaze-large-stitch-v0":
        d_obs_mm = OgB_AntMaze_Large_Stitch_Obs__Min_Max
        d_act_mm = OgB_AntMaze_Giant_Act__Min_Max
    
    elif dset_name == "antmaze-medium-stitch-v0":
        d_obs_mm = OgB_AntMaze_Medium_Stitch_Obs__Min_Max
        d_act_mm = OgB_AntMaze_Giant_Act__Min_Max
        
    ## Add normalizer for other mazes
    elif dset_name == "antmaze-large-explore-v0":
        d_obs_mm = OgB_AntMaze_Large_Explore_Obs__Min_Max
        d_act_mm = OgB_AntMaze_Giant_Act__Min_Max
    elif dset_name == "antmaze-medium-explore-v0":
        d_obs_mm = OgB_AntMaze_Medium_Explore_Obs__Min_Max
        d_act_mm = OgB_AntMaze_Giant_Act__Min_Max
    elif dset_name == "humanoidmaze-giant-stitch-v0":
        d_obs_mm = OgB_HumanoidMaze_Giant_Stitch_Obs__Min_Max
        d_act_mm = OgB_HumanoidMaze_Giant_Act__Min_Max
    elif dset_name == "humanoidmaze-large-stitch-v0":
        d_obs_mm = OgB_HumanoidMaze_Large_Stitch_Obs__Min_Max
        d_act_mm = OgB_HumanoidMaze_Giant_Act__Min_Max
    elif dset_name == "humanoidmaze-medium-stitch-v0":
        d_obs_mm = OgB_HumanoidMaze_Medium_Stitch_Obs__Min_Max
        d_act_mm = OgB_HumanoidMaze_Giant_Act__Min_Max

    ## Soccer
    elif dset_name == "antsoccer-arena-stitch-v0":
        d_obs_mm = OgB_AntSoccer_Arena_Stitch_Obs__Min_Max
        d_act_mm = OgB_AntMaze_Giant_Act__Min_Max
    elif dset_name == "antsoccer-medium-stitch-v0":
        d_obs_mm = OgB_AntSoccer_Medium_Stitch_Obs__Min_Max
        d_act_mm = OgB_AntMaze_Giant_Act__Min_Max

    ## Point Maze
    elif dset_name == 'pointmaze-giant-stitch-v0':
        d_obs_mm = OgB_PointMaze_Giant_Stitch_Obs__Min_Max
        d_act_mm = OgB_PointMaze_Giant_Act__Min_Max

    elif dset_name == 'pointmaze-large-stitch-v0':
        d_obs_mm = OgB_PointMaze_Large_Stitch_Obs__Min_Max
        d_act_mm = OgB_PointMaze_Giant_Act__Min_Max

    elif dset_name == 'pointmaze-medium-stitch-v0':
        d_obs_mm = OgB_PointMaze_Medium_Stitch_Obs__Min_Max
        d_act_mm = OgB_PointMaze_Giant_Act__Min_Max

    ## -------- Navigation Dataset ---------
    elif dset_name == 'antmaze-giant-navigate-v0':
        d_obs_mm = OgB_AntMaze_Giant_Navigate_Obs__Min_Max
        d_act_mm = OgB_AntMaze_Giant_Act__Min_Max
    else:
        assert False, 'to be implemented'

    if obs_dim_idxs is None:
        obs_select_dim = args_train.dataset_config['obs_select_dim']
    elif obs_dim_idxs == 'full':
        obs_select_dim = tuple(range(d_obs_mm.shape[-1]))
    else:
        raise NotImplementedError

    data_dict['observations'] = d_obs_mm[:, obs_select_dim]
    data_dict['actions']  = d_act_mm[:, :]

    # pdb.set_trace()

    norm_const_dict = args_train.dataset_config.get('norm_const_dict', False)
    if norm_const_dict:
        assert np.isclose(data_dict['actions'], np.array(norm_const_dict['actions'], dtype=np.float32)).all()
        data_dict['observations'] = np.array(norm_const_dict['observations'], dtype=np.float32)
        pdb.set_trace() ## should be good, just stop for checking
    # -------------------
    else:
        print('args_train: no norm_const_dict.')

    d_norm = DatasetNormalizer(data_dict, normalizer, eval_solo=True, path_lengths=None)

    # pdb.set_trace()

    return d_norm




def get_ogb_maze_ev_probs_fname(env_name):
    """
    Get the corresponding file name that saves the ev probs
    To load the evaluation problems out
    """
    root_dir = './'
    hdf5_path = None
    if env_name == "antmaze-giant-stitch-v0":
        hdf5_path = "data/ogb_maze/ev_probs/ogb_antM_Gi_ev_prob_numEp20_eSdSt0.hdf5"
    
    elif env_name == "antmaze-large-stitch-v0":
        hdf5_path = "data/ogb_maze/ev_probs//ogb_antM_Lg_ev_prob_numEp20_eSdSt0.hdf5"
    
    elif env_name == "antmaze-medium-stitch-v0":
        hdf5_path = "data/ogb_maze/ev_probs//ogb_antM_Me_ev_prob_numEp20_eSdSt0.hdf5"
    
    ## Explore
    elif env_name == "antmaze-large-explore-v0":
        hdf5_path = "data/ogb_maze/ev_probs//ogb_antM_LgExpl_ev_prob_numEp20_eSdSt0.hdf5"

    elif env_name == "antmaze-medium-explore-v0":
        hdf5_path = "data/ogb_maze/ev_probs//ogb_antM_MeExpl_ev_prob_numEp20_eSdSt0.hdf5"


    ## ------ Huamnoid ------
    elif env_name == "humanoidmaze-giant-stitch-v0":
        hdf5_path = "data/ogb_maze/ev_probs/ogb_HumM_Gi_ev_prob_numEp20_eSdSt0_full_jnt.hdf5"
    elif env_name == "humanoidmaze-large-stitch-v0":
        hdf5_path = "data/ogb_maze/ev_probs/ogb_HumM_Lg_ev_prob_numEp20_eSdSt0_full_jnt.hdf5"

    elif env_name == "humanoidmaze-medium-stitch-v0":
        hdf5_path = "data/ogb_maze/ev_probs/ogb_HumM_Me_ev_prob_numEp20_eSdSt0_full_jnt.hdf5"


    ## ------ Ant Soccer ------
    elif env_name == 'antsoccer-arena-stitch-v0':
        hdf5_path = 'data/ogb_maze/ev_probs//ogb_antSoc_Ar_ev_prob_numEp20_eSdSt0.hdf5' 
    elif env_name == 'antsoccer-medium-stitch-v0':
        hdf5_path = 'data/ogb_maze/ev_probs//ogb_antSoc_Me_ev_prob_numEp20_eSdSt0.hdf5' 

    ## ------ Point Maze ------
    elif env_name == 'pointmaze-giant-stitch-v0':
        hdf5_path = 'data/ogb_maze/ev_probs/ogb_pointM_Gi_ev_prob_numEp20_eSdSt0.hdf5'
    elif env_name == 'pointmaze-large-stitch-v0':
        hdf5_path = 'data/ogb_maze/ev_probs/ogb_pointM_Lg_ev_prob_numEp20_eSdSt0.hdf5'
    elif env_name == 'pointmaze-medium-stitch-v0':
        hdf5_path = 'data/ogb_maze/ev_probs/ogb_pointM_Me_ev_prob_numEp20_eSdSt0.hdf5'
    
    elif env_name == 'antmaze-giant-navigate-v0':
        hdf5_path = 'data/ogb_maze/ev_probs//ogb_antM_Gi_Navi_ev_prob_numEp20_eSdSt0_preAct5.hdf5'
    else:
        raise NotImplementedError

    

    hdf5_path = root_dir + hdf5_path

    return hdf5_path


## ---------------------------------------------------------
## -----------------  For Inv Dyn Model  -------------------
## ---------------------------------------------------------

def ogb_get_inv_model_path(env_name, gl_dim, inv_hzn=None):

    ## -------- All 5 Ant Maze --------
    if env_name == "antmaze-giant-stitch-v0":
        if gl_dim == 2:
            out_path = 'logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_o29d_g2d_invdyn_h12'
        elif gl_dim == 15:
            out_path = 'logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_o29d_g15d_invdyn_h12'
        elif gl_dim == 29:
            out_path = "logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_o29d_g29d_invdyn_h12_dm5"
    

    elif env_name == "antmaze-large-stitch-v0":
        if gl_dim == 2:
            out_path = "logs/antmaze-large-stitch-v0/diffusion/og_antM_Lg_o29d_g2d_invdyn_h12"
        elif gl_dim == 15:
            out_path = "logs/antmaze-large-stitch-v0/diffusion/og_antM_Lg_o29d_g15d_invdyn_h12"
        elif gl_dim == 29:
            out_path = "logs/antmaze-large-stitch-v0/diffusion/og_antM_Lg_o29d_g29d_invdyn_h12"


    elif env_name == "antmaze-medium-stitch-v0":
        if gl_dim == 2:
            out_path = "logs/antmaze-medium-stitch-v0/diffusion/og_antM_Me_o29d_g2d_invdyn_h12"
        elif gl_dim == 15:
            out_path = "logs/antmaze-medium-stitch-v0/diffusion/og_antM_Me_o29d_g15d_invdyn_h12"
        elif gl_dim == 29:
            out_path = "logs/antmaze-medium-stitch-v0/diffusion/og_antM_Me_o29d_g29d_invdyn_h12_dm5"
    
    
    
    elif env_name == "antmaze-large-explore-v0":
        if gl_dim == 2:
            out_path = "logs/antmaze-large-explore-v0/diffusion/og_antMexpl_Lg_o29d_g2d_invdyn_h12"
        elif gl_dim == 15:
            out_path = "logs/antmaze-large-explore-v0/diffusion/og_antMexpl_Lg_o29d_g15d_invdyn_h12"
        elif gl_dim == 29:
            out_path = "logs/antmaze-large-explore-v0/diffusion/og_antMexpl_Lg_o29d_g29d_invdyn_h12"
    
    elif env_name == "antmaze-medium-explore-v0":
        if gl_dim == 2:
            out_path = "logs/antmaze-medium-explore-v0/diffusion/og_antMexpl_Me_o29d_g2d_invdyn_h12"
        elif gl_dim == 15:
            out_path = "logs/antmaze-medium-explore-v0/diffusion/og_antMexpl_Me_o29d_g15d_invdyn_h12"
        elif gl_dim == 29:
            out_path = "logs/antmaze-medium-explore-v0/diffusion/og_antMexpl_Me_o29d_g29d_invdyn_h12"

        # if gl_dim == 2:
        #     out_path = ""
        # elif gl_dim == 15:
        #     out_path = ""
        # elif gl_dim == 29:
        #     out_path = ""


    ## ------ All three Huamnoid Maze ------
    elif env_name == "humanoidmaze-giant-stitch-v0":
        if gl_dim == 2:
            out_path = 'logs/humanoidmaze-giant-stitch-v0/diffusion/og_humM_Gi_o69d_g2d_invdyn_h80_dm5_dout02'

    elif env_name == "humanoidmaze-large-stitch-v0":
        if gl_dim == 2:
            out_path = "logs/humanoidmaze-large-stitch-v0/diffusion/og_humM_Lg_o69d_g2d_invdyn_h80_dm5_dout02"

    elif env_name == "humanoidmaze-medium-stitch-v0":
        if gl_dim == 2:
            out_path = "logs/humanoidmaze-medium-stitch-v0/diffusion/og_humM_Me_o69d_g2d_invdyn_h80_dm5_dout02"


    #### --------- Ant Soccer ---------
    #### --------- Ant Soccer ---------

    elif env_name == 'antsoccer-arena-stitch-v0':
        if gl_dim == 4:
            out_path = "logs/antsoccer-arena-stitch-v0/diffusion/og_antSoc_Ar_o42d_g4d_invdyn_h120_dout02"
        elif gl_dim == 17:
            out_path = "logs/antsoccer-arena-stitch-v0/diffusion/og_antSoc_Ar_o42d_g17d_invdyn_h120_dout02"

    elif env_name == 'antsoccer-medium-stitch-v0':
        if gl_dim == 4:
            out_path = "logs/antsoccer-medium-stitch-v0/diffusion/og_antSoc_Me_o42d_g4d_invdyn_h120_dm4_dout02"
        elif gl_dim == 17:
            out_path = "logs/antsoccer-medium-stitch-v0/diffusion/og_antSoc_Me_o42d_g17d_invdyn_h100_dm4_dout02"
    
    elif 'pointmaze-' in env_name:
        out_path = None

    ## TODO: temporary
    elif env_name == "antmaze-giant-navigate-v0":
        if gl_dim == 2:
            # out_path = 'logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_o29d_g2d_invdyn_h12'
            pass

    ## if you want to add new env, please follow the format above
    ## out_path should be the path to the folder of the inv dyn model
    else:
        raise NotImplementedError
    
    assert out_path != ""
    if 'antsoccer-' not in env_name and 'pointmaze-' not in env_name:
        assert env_name in out_path and f'g{str(gl_dim)}d' in out_path

    return out_path
