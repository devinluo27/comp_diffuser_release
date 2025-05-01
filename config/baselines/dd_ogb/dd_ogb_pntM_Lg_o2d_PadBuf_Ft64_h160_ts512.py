import socket
import os.path as osp

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args
config_fn = osp.splitext(osp.basename(__file__))[0]

diffusion_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

sm_horizon = 160
tot_horizon = sm_horizon
time_dim = 96


base = {
    'dataset': "pointmaze-large-stitch-v0",

    'diffusion': {
        'config_fn': '',

        'sm_horizon': sm_horizon,
        'tot_horizon': tot_horizon,


        ##
        ## cnn model
        'model': 'baselines.dd_maze.dd_maze_temporal_v1.Unet1D_DD_Maze_V1',
        'base_dim': 128,
        'dim_mults': (1, 2, 4, 8),
        'time_dim': time_dim,
        'network_config': dict(t_seq_encoder_type='mlp',
                                cat_t_w=True, 
                                resblock_ksize=5,
                                energy_mode=False,
                                time_mlp_config=3,
                                ###
                                inpaint_token_dim=32,
                                inpaint_token_type='const',
                               ),
        
        ## sm dfu model
        'dfu_model': 'baselines.dd_maze.dd_maze_diffusion_v1.DD_Maze_GauDiffusion_InvDyn_V1',

        'n_diffusion_steps': 512,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False, ##
        'diff_config': dict(
                            infer_deno_type='same',
                            obs_manual_loss_weights={},
                            w_loss_type='all',
                            is_direct_train=True,
                            ##
                            ),
        
        
        # 'trainer_cls': 'ogb_task.ogb_maze_v1.OgB_Stgl_Sml_Trainer_v1',
        'trainer_dict': dict(),


        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset
        'loader': 'datasets.ogb_dset.OgB_SeqDataset_V2',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 300,
        # 'max_path_length': 2200,
        'dataset_config': dict(
            obs_select_dim=(0,1), ####
            dset_type='ogb',
            ###
            pad_option_2='buf',
            pad_type='first',
            extra_pad=64, ##
        ),

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2_inv_v3',
        'n_train_steps': 2e6,

        'batch_size': 128,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 4000,
        'sample_freq': 8000,
        'n_saves': 5,

        'n_reference': 40,
        'n_samples': 10,

        'device': 'cuda',
    },

    'plan': {
        'config_fn': '',

        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': tot_horizon,
        'n_diffusion_steps': 512,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}

