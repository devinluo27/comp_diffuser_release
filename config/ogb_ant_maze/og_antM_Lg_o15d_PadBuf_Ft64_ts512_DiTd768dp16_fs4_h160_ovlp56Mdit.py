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
len_ovlap = 56
tot_horizon = sm_horizon
f_s = 4

ovlp_o_dim = 384
ovlp_model_config = dict(
    c_traj_hzn=len_ovlap,
    in_dim=15,
    out_dim=ovlp_o_dim,
    hidden_size=ovlp_o_dim,
    depth=8, num_heads=6, mlp_ratio=4.0,
    tjti_enc_config=dict(
        frame_stack=f_s, w_init_type='no',),
)

di_tr_model_config=dict(
    hidden_size=768, depth=16, num_heads=12, mlp_ratio=4.0,
    ovlp_model_type='dit_enc',
    st_ovlp_model_config=ovlp_model_config, end_ovlp_model_config=ovlp_model_config,
    inpaint_token_dim=48, ##
    inpaint_token_type='const',
    # inpaint_token_type='learn_if_inpt',
    t_cond_type='add',
    frame_stack=f_s,
    )


base = {
    'dataset': "antmaze-large-stitch-v0",

    'diffusion': {
        'config_fn': '',

        'sm_horizon': sm_horizon,
        'tot_horizon': tot_horizon,


        ##
        'model': 'ogb_task.og_models.stgl_sml_dit_1d.DiT1D_TjTi_Stgl_Cond_V1',
        'network_config': di_tr_model_config,


        ## sm dfu model
        'dfu_model': 'models.cd_stgl_sml_dfu.stgl_sml_diffusion_v1.Stgl_Sml_GauDiffusion_InvDyn_V1',
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
                            len_ovlp_cd=len_ovlap,
                            tr_1side_drop_prob=0.20,
                            ## --- NEW ---
                            tr_inpat_prob=0.5,
                            tr_ovlp_prob=0.5,
                            tr_no_ovlp_none=False,
                            # tr_no_ovlp_none=True,
                            # non_repla_inpat_prob=0.5,
                            ),
        
        'trainer_cls': 'ogb_task.ogb_maze_v1.OgB_Stgl_Sml_Trainer_v1',
        'step_start_ema': 0,
        'update_ema_every': 1,
        'trainer_dict': dict(
            optim_type='adamw',
            weight_decay=0,
            ##
            # do_train_resume=True,
            # path_resume='', ## set it to a path of a checkpoint
        ),


        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset
        'loader': 'datasets.ogb_dset.OgB_SeqDataset_V2',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 300,
        'dataset_config': dict(
            obs_select_dim=tuple(range(15)), ####
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
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.9999,
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

