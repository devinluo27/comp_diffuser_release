"""
Reference From https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py
"""
import torch, pdb, einops
import torch.nn as nn
import numpy as np
import diffuser.utils as utils
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from diffuser.ogb_task.og_models.dit_1d_utils import TimestepEmbedder, DiTBlock, LabelEmbedder, FinalLayer, get_1d_sincos_pos_embed_from_grid
from diffuser.models.cond_cp_dfu.sml_helpers import Traj_Time_Encoder
from diffuser.ogb_task.og_models.dit_1d_traj_encoder import DiT1D_Traj_Time_Encoder


class DiT1D_TjTi_Stgl_Cond_V1(nn.Module):
    """
    Diffusion model (CompDiffuser) with a Transformer backbone.
    """
    def __init__(
        self,
        horizon,
        transition_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        network_config={},
    ):
        super().__init__()
        self.learn_sigma = learn_sigma

        self.frame_stack = network_config.get('frame_stack', 1)

        self.horizon = horizon
        transition_dim = transition_dim * self.frame_stack
        self.transition_dim = transition_dim
        self.hidden_size = hidden_size
        
        self.out_channels = transition_dim * 2 if learn_sigma else transition_dim
        
        self.num_heads = num_heads

        self.network_config = network_config


        

        

        ## ------------------------------------------
        ## ------ initialize ovlp part encoder ------
        ## ------------------------------------------

        ovlp_model_type = network_config['ovlp_model_type']
        self.st_ovlp_model_config = network_config['st_ovlp_model_config']
        self.end_ovlp_model_config = network_config['end_ovlp_model_config']
        ###
        if ovlp_model_type == 'unet':
            ## The initilization is customized By Luo, not sure if it is good
            self.st_ovlp_model = Traj_Time_Encoder(**self.st_ovlp_model_config)
            self.end_ovlp_model = Traj_Time_Encoder(**self.end_ovlp_model_config)
            
            # pdb.set_trace() ## check init, just use default?

        elif ovlp_model_type == 'dit_enc':
            self.st_ovlp_model = DiT1D_Traj_Time_Encoder(**self.st_ovlp_model_config)
            self.end_ovlp_model = DiT1D_Traj_Time_Encoder(**self.end_ovlp_model_config)

        ## ---------------------------------------------
        ## ------ For inpainting start and goal --------
        self.create_inpat_nets()

        ## ---------------------------------------------
            

        ### ------------------------------------------------------------------------
        ## -- Define How Condition Signal are fused with the Transformer Denoiser --
        
        ovlp_2_cond_dim = self.st_ovlp_model.out_dim + self.end_ovlp_model.out_dim
        self.t_cond_type = network_config['t_cond_type']
        
        if self.t_cond_type == 'add':
            ## the cat cond [st_ovlp,end_ovlp,st_token,end_token] then + t_feat
            tot_cond_dim = ovlp_2_cond_dim + 2 * self.inpaint_token_dim
            self.time_dim = tot_cond_dim
            self.t_embedder = TimestepEmbedder(self.time_dim)
        else:
            raise NotImplementedError

        # pdb.set_trace()
        ### --------------------------------------------


        ## ---------- Init the DiT 1D Backbone -------------

        ## PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        ## similar to diffusion policy, just one Linear, might upgrade to an MLP
        self.x_embedder = nn.Linear(in_features=transition_dim, out_features=hidden_size)

        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        ## self.x_embedder.num_patches
        self.num_patches = self.horizon // self.frame_stack
        assert self.horizon % self.frame_stack == 0
        
        # Will use fixed sin-cos embedding: (will be init later)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                     cond_dim=tot_cond_dim) for _ in range(depth)
        ])
        
        # pdb.set_trace()


        self.final_layer = FinalLayer(hidden_size, self.out_channels, cond_dim=tot_cond_dim,)

        



        ## -----------
        self.initialize_weights()

        # pdb.set_trace()
        utils.print_color(f'[DiT1D_TjTi_Stgl_Cond_V1] {hidden_size=} {depth=}, {tot_cond_dim=},')
        self.input_t_type = '1d'

        




    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))

        assert self.num_patches * self.frame_stack == self.horizon, 'for now'
        tmp_pos_arr = np.arange(self.num_patches, dtype=np.int32 )
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.hidden_size, pos=tmp_pos_arr)
        
        # pdb.set_trace()
        ## pos_embed will still be float32
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        ## w = self.x_embedder.proj.weight.data # ori
        ## nn.init.xavier_uniform_(w.view([w.shape[0], -1])) # ori
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w) ## no need to reshape, since already a Linear
        nn.init.constant_(self.x_embedder.bias, 0)

        ## NOTE: be careful of the initialization for the cond networks
        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # pdb.set_trace() ## TODO: check our cond model initialization

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    

    def forward(self, x, time,  # y):
                tj_cond: dict,
                force_dropout=False, half_fd=False,):
        """
        Forward pass of DiT.
        x: (B, H, Dim) a batch of trajs
        time: (B,) tensor of diffusion timesteps
        
        
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        time: (B,) tensor of diffusion timesteps
        
        ## not Used y: (N,) tensor of class labels
        """

        is_st_inpat = tj_cond['is_st_inpat'] ## torch tensor gpu
        is_end_inpat = tj_cond['is_end_inpat']
        ## sanity check
        b_size = x.shape[0]
        assert is_st_inpat.shape[0] == b_size and is_st_inpat.ndim == 1 \
            and is_st_inpat.dtype == torch.bool
        assert is_end_inpat.shape[0] == b_size and is_end_inpat.ndim == 1 \
            and is_end_inpat.dtype == torch.bool

        if self.frame_stack > 1:
            x = einops.rearrange(x, "b (t fs) dim -> b t (fs dim)", fs=self.frame_stack)
            # pdb.set_trace()

        ## (B, H, D), e.g., [4, 160, 384]
        x_input_emb = self.x_embedder(x)
        # pdb.set_trace()
        ## (B, H, D)
        x_input_emb = x_input_emb + self.pos_embed


        # pdb.set_trace()


        ## ------------------------------------------------------
        ## ------------- obtain Condition feature ---------------

        st_ovlp_is_drop = tj_cond['st_ovlp_is_drop']
        end_ovlp_is_drop = tj_cond['end_ovlp_is_drop']

        if st_ovlp_is_drop is not None: ##
            st_ovlp_feat = self.st_ovlp_model(tj_cond['st_ovlp_traj'], 
                                    time=tj_cond['st_ovlp_t'])
            assert len(st_ovlp_is_drop) == len(st_ovlp_feat)
            assert st_ovlp_is_drop.dtype == torch.bool ## a numpy array
            st_ovlp_feat[ st_ovlp_is_drop ] = 0.
            
            assert not torch.logical_and(~st_ovlp_is_drop, is_st_inpat).any() ## must be false
        else:
            ## no cond if None
            st_ovlp_feat = torch.zeros( (x.shape[0], self.st_ovlp_model.out_dim), device=x.device)

        
        if tj_cond['end_ovlp_is_drop'] is not None:
            end_ovlp_feat = self.end_ovlp_model(tj_cond['end_ovlp_traj'],
                                                time=tj_cond['end_ovlp_t'])
            end_ovlp_feat[ tj_cond['end_ovlp_is_drop'] ] = 0.
            assert end_ovlp_is_drop.dtype == torch.bool
            assert not torch.logical_and(~end_ovlp_is_drop, is_end_inpat).any()
        else:
            ## no cond if None
            end_ovlp_feat = torch.zeros( (x.shape[0], self.end_ovlp_model.out_dim), device=x.device)


        ## -------------- For Inpainting --------------
        ## TODO: Dec 23 16:32 Check How ViT copy the Cls token; Ans: just repeat is fine
        ## Here we create corresponding condition feature to let the model know if we actually overwrite!
        if self.inpaint_token_type == 'const':
            # (B,token_dim)
            st_token = torch.zeros(size=(b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device)
            num_st_inpt = torch.sum(is_st_inpat).item()
            ## assign value
            st_token[is_st_inpat] = self.st_use_inpaint_token.repeat( (num_st_inpt, 1) )
            st_token[~is_st_inpat] = self.st_no_inpaint_token.repeat( (b_size - num_st_inpt, 1) )

            end_token = torch.zeros(size=(b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device)
            num_end_inpt = torch.sum(is_end_inpat).item()
            end_token[is_end_inpat] = self.end_use_inpaint_token.repeat( (num_end_inpt, 1) )
            end_token[~is_end_inpat] = self.end_no_inpaint_token.repeat( (b_size - num_end_inpt, 1) )

            st_token = self.st_inpaint_model(st_token)
            end_token = self.end_inpaint_model(end_token)
        elif self.inpaint_token_type == 'learn_if_inpt':
            ## repeat to full b_size to init
            st_token = self.st_use_inpaint_token.repeat( (b_size, 1) )
            num_st_inpt = torch.sum(is_st_inpat).item()
            ## make them invalid if no inpat
            st_token[~is_st_inpat] *= 0. 
            st_token[~is_st_inpat] = self.st_no_inpaint_token.repeat( (b_size - num_st_inpt, 1) )


            end_token = self.end_use_inpaint_token.repeat( (b_size, 1) )
            num_end_inpt = torch.sum(is_end_inpat).item()
            end_token[~is_end_inpat] *= 0. 
            end_token[~is_end_inpat] = self.end_no_inpaint_token.repeat( (b_size - num_end_inpt, 1) )

            st_token = self.st_inpaint_model(st_token)
            end_token = self.end_inpaint_model(end_token)


        else:
            raise NotImplementedError
        
        ## NOTE: we can only either do inpainting or en_ovlp conditioning
        # pdb.set_trace()

        ## --------------------------------------------

        if force_dropout:
            # pdb.set_trace() ## important: do not drop the st_token?
            assert not self.training
            if half_fd:
                b_s = len(st_ovlp_feat)
                # drop the second half
                assert b_s % 2 == 0
                st_ovlp_feat[int(b_s//2):] = 0. # * st_ovlp_feat[int(b_s//2):] 
                end_ovlp_feat[int(b_s//2):] = 0. # * end_ovlp_feat[int(b_s//2):] 
            else:
                assert False

        ## ------------ create cond_feat for denoiser --------------

        if self.t_cond_type == 'add':
            ## our cond feat, e.g., B, 256+256+32+32
            y_feat_cat = torch.cat([ st_ovlp_feat, end_ovlp_feat, st_token, end_token ], dim=-1)

            # pdb.set_trace() ## check dim of each one

            ## (B, tot_cond_dim)
            t_feat = self.t_embedder(time)
            c_feat_all = t_feat + y_feat_cat

        else:
            raise NotImplementedError

        

        # pdb.set_trace()

        ## ------------ Denoising BackBone ----------------

        x = x_input_emb ## prev: x.shape=B,H,obs_dim

        for block in self.blocks:
            x = block(x, c_feat_all)  # (B, T, hid_D)
        x = self.final_layer(x, c_feat_all)  # (B, T, obs_dim)
        
        # pdb.set_trace() ## TODO: Dec 23 Check Back after XFormer
        if self.frame_stack > 1:
            x = einops.rearrange(x, "b t (fs dim) -> b (t fs) dim", fs=self.frame_stack)
            # pdb.set_trace()

        return x
    





    def create_inpat_nets(self):
        """
        Create the Needed Module for Inpainting
        """
        
        self.st_inpaint_model = nn.Identity()
        self.end_inpaint_model = nn.Identity()
        self.inpaint_token_dim = self.network_config['inpaint_token_dim'] ## e.g., 32
        self.inpaint_token_type = self.network_config['inpaint_token_type'] ## e.g., const
        if self.inpaint_token_type == 'const':
            self.st_use_inpaint_token: torch.Tensor
            self.register_buffer( 'st_use_inpaint_token', \
                                 torch.full(size=(1,self.inpaint_token_dim), fill_value=1., dtype=torch.float32) )

            self.st_no_inpaint_token: torch.Tensor
            self.register_buffer('st_no_inpaint_token', \
                             torch.full(size=(1,self.inpaint_token_dim), fill_value=0., dtype=torch.float32) )

            self.end_use_inpaint_token: torch.Tensor
            self.register_buffer( 'end_use_inpaint_token', 
                                 torch.full(size=(1,self.inpaint_token_dim), fill_value=1., dtype=torch.float32) )
            
            self.end_no_inpaint_token: torch.Tensor
            self.register_buffer( 'end_no_inpaint_token', 
                             torch.full(size=(1,self.inpaint_token_dim), fill_value=0., dtype=torch.float32) )

        elif self.inpaint_token_type == 'learn_if_inpt': ## added Dec 23
            ## 0 if not inpat, learned vector if do inpat
            self.st_use_inpaint_token = nn.Parameter(
                torch.zeros(1, self.inpaint_token_dim), requires_grad=True)
            
            nn.init.normal_(self.st_use_inpaint_token, std=0.02)

            self.st_no_inpaint_token: torch.Tensor
            self.register_buffer('st_no_inpaint_token', \
                             torch.full(size=(1,self.inpaint_token_dim), fill_value=0., dtype=torch.float32) )

            self.end_use_inpaint_token = nn.Parameter(
                torch.zeros(1, self.inpaint_token_dim), requires_grad=True)
            
            nn.init.normal_(self.end_use_inpaint_token, std=0.02)

            self.end_no_inpaint_token: torch.Tensor
            self.register_buffer( 'end_no_inpaint_token', 
                             torch.full(size=(1,self.inpaint_token_dim), fill_value=0., dtype=torch.float32) )
            
        else:
            raise NotImplementedError
        
        # pdb.set_trace() ## check init



    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    




    








