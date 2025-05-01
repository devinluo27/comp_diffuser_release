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


tot_horizon = 12
obs_dim = 29


base = {
    'dataset': "antmaze-medium-stitch-v0",
    

    'diffusion': {
        'config_fn': '',

        'tot_horizon': tot_horizon,
        'goal_sel_idxs': tuple(range(2)),

        ##
        ## inv model
        'model': 'ogb_task.og_inv_dyn.MLP_InvDyn_OgB_V3',
        'inv_m_config': dict(
            hidden_dims=[512, 512, 512],
            final_fc_init_scale=1e-2,
            is_out_dist=False,
        ),
        'act_net_config': dict(
            act_f='gelu',
            use_dpout=False,
        ),
        
        ###
        'trainer_dict': dict(),



        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset
        'loader': 'datasets.ogb_dset.OgB_InvDyn_SeqDataset_V1',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 220,
        'max_n_episodes': 10000, ####
        'dataset_config': dict(
            obs_select_dim=tuple(range(obs_dim)), ####
            dset_type='ogb',
        ),

        
        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),


        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2_inv_v3',
        'n_train_steps': 2e6,

        'ema_decay': 0.995,
        'batch_size': 1024,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 1,

        'sample_freq': 4000,
        'save_freq': 4000,
        'n_saves': 5,

        'n_reference': 20, ##
        'n_samples': 8, ##

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

