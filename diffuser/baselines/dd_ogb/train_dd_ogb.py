import os,sys,pdb,socket; sys.path.append('./')
if socket.gethostname() == 'bishop':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['MUJOCO_GL'] = 'egl'
else:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['MUJOCO_GL'] = 'osmesa'

import diffuser.utils as utils
import torch, wandb, pdb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
### ---------
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=4, sci_mode=False)
import numpy as np; np.set_printoptions(precision=3, suppress=True)


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = None
    config: str


args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# pdb.set_trace()

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.tot_horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    max_path_length=args.max_path_length,
    ###
    max_n_episodes=getattr(args, 'max_n_episodes', 10000), 
    ###
    termination_penalty=args.termination_penalty,
    use_padding=args.use_padding,
    ## put a linnk to a smaller dataset for debugging purpose
    dset_h5path=getattr(args, 'dset_h5path', None),
    dataset_config=args.dataset_config,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    ## we can pass the env from dataset to avoid double initialization,
    ## which might cause pure black rendering image
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim ## 2
action_dim = dataset.action_dim ## 2

## dataset.normalizer.normalizers['observations'].mins
## torch.tensor(dataset.normalizer.normalizers['observations'].mins)
# for _ in range(20):
    # test_sample = dataset[0]
if 'getNorm' in args.config:
    torch.set_printoptions(precision=10, sci_mode=False)
    # np.set_printoptions(precision=10, suppress=True)
    pdb.set_trace() ## check horizon

# pdb.set_trace() ## check horizon

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

## initialize a DiT1D, Dec 23
if args.model == 'ogb_task.og_models.stgl_sml_dit_1d.DiT1D_TjTi_Stgl_Cond_V1':
    assert False, 'not implemented for DD baseline'
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        device=args.device,
        ##
        horizon=args.sm_horizon,
        transition_dim=observation_dim,
        hidden_size=args.network_config['hidden_size'],
        depth=args.network_config['depth'],
        num_heads=args.network_config['num_heads'],
        mlp_ratio=args.network_config['mlp_ratio'],
        learn_sigma=False,
        ##
        network_config=args.network_config,
    )
    model = model_config()
    ## --- If Used, Inference will be slower, but training will be 3-5%? faster ---
    ## **Not Use, because our init weight will be overwrite**
    if args.network_config.get('use_xf_attn', False):
        assert False
        from diffuser.ogb_task.og_models.dit_1d_utils import replace_attn_with_xformers_one
        model = replace_attn_with_xformers_one(model, att_mask=None)

else:
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        ##
        device=args.device,
        ##
        horizon=args.sm_horizon,
        transition_dim=observation_dim,
        base_dim=args.base_dim, ## new
        dim_mults=args.dim_mults,
        time_dim=args.time_dim, ## new
        network_config=args.network_config, ## new
    )

    model = model_config()



# pdb.set_trace()


## model to be input
dfu_model_config = utils.Config(
    args.dfu_model,
    savepath=(args.savepath, 'dfu_model.pkl'),
    device=args.device,
    ##
    horizon=args.sm_horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_discount=args.loss_discount,
    loss_weights=args.loss_weights,
    ## ----- Luo -----
    diff_config=args.diff_config,
)



from diffuser.baselines.dd_maze.dd_maze_training_v1 import DD_Maze_Trainer_v1
trainer_cls = getattr(args, 'trainer_cls', DD_Maze_Trainer_v1)

## add our ogb trainer, align the hyper-param as DiT



small_trainer_config = utils.Config(
    trainer_cls,
    savepath=(args.savepath, 'small_trainer_config.pkl'),
    ##
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    step_start_ema=getattr(args, 'step_start_ema', 2000),
    update_ema_every=getattr(args, 'update_ema_every', 10),
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),

    results_folder=args.savepath,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    trainer_dict=args.trainer_dict,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#



dfu_model = dfu_model_config(model=model)

trainer = small_trainer_config(diffusion_model=dfu_model, 
                              dataset=dataset, 
                              renderer=renderer,
                              device=args.device,
                              )



if args.trainer_dict.get('do_train_resume', False): # for a sample resume, should be good
    tmp_path = args.trainer_dict['path_resume']
    # pdb.set_trace()
    tmp_cfg = args.config.split('/')[-1]
    tmp_fd_idx = tmp_cfg.find('_resume')
    tmp_cfg = tmp_cfg[:tmp_fd_idx] if tmp_fd_idx >=0 else tmp_cfg
    assert tmp_cfg in tmp_path
    utils.print_color(f'Resume From: {tmp_path}', c='c')
    trainer.load4resume( tmp_path )

# pdb.set_trace()
#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0]) # [1,380,2]
# batch = utils.batchify_seq( [dataset[0], dataset[1]] )
batch = utils.batch_copy(batch, 4)
obs_trajs, act_trajs, stgl_cond = batch

# pdb.set_trace() ## TODO: check a bit the dataloader, Seems fine?

##---- can be delete, for debug
# for k,v in dict(dfu_model.named_parameters()).items():
#     if 'dfu_model' not in k:
#         print(k)
##----
loss, _ = dfu_model.loss(x_clean=obs_trajs, cond_st_gl=stgl_cond) ## just uncond now
loss.backward()
# pdb.set_trace()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- save config ---------------------------------#
#-----------------------------------------------------------------------------#


all_configs = dict(dataset_config=dataset_config._dict, 
                render_config=render_config._dict,
                model_config=model_config._dict,
                dfu_model_config=dfu_model_config._dict,
                small_trainer_config=small_trainer_config._dict
                )

# print(args)
ckp_path = args.savepath
wandb.init(
    project="comp_diffuser_release",
    name=args.logger_name,
    id=args.logger_id,
    dir=ckp_path,
    config=all_configs, ## need to be a dict
    # resume="must",
    mode='online' if dataset_config.dset_h5path is None else 'disabled'
)
# pdb.set_trace()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
# if False: ## ori
if True: ## ori
    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    if args.trainer_dict.get('do_train_resume', False): ## for resume
        n_epochs = int( (args.n_train_steps - trainer.step)  // args.n_steps_per_epoch)

    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch)





