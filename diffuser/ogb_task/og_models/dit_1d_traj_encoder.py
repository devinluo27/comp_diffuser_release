"""
Adapted based on https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py
"""
if __name__ == '__main__':
    import sys; sys.path.append('./')
import torch, pdb, einops
import torch.nn as nn
import numpy as np
import diffuser.utils as utils
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from diffuser.ogb_task.og_models.dit_1d_utils import TimestepEmbedder, DiTBlock, LabelEmbedder, FinalLayer, get_1d_sincos_pos_embed_from_grid
from diffuser.models.cond_cp_dfu.sml_helpers import Traj_Time_Encoder


class DiT1D_Traj_Time_Encoder(nn.Module):
    """
    A DiT Based Traj Encoder for the Ovlp Part, 
    We just need the Cls-Token
    """
    def __init__(
        self,
        c_traj_hzn,
        in_dim,
        out_dim,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        tjti_enc_config={},
    ):
        super().__init__()

        self.frame_stack = tjti_enc_config.get('frame_stack', 1)

        self.c_traj_hzn = c_traj_hzn
        self.out_dim = out_dim
        self.transition_dim = in_dim * self.frame_stack
        self.hidden_size = hidden_size
        
        self.out_channels = self.transition_dim
        
        self.num_heads = num_heads

        self.tjti_enc_config = tjti_enc_config

        ### ------------------------------------------------------------------------
        ## -- Define How Condition Signal are fused with the Transformer Denoiser --

        self.time_dim = hidden_size
        self.t_embedder = TimestepEmbedder(self.time_dim)

        ## ---------- Init the DiT 1D Backbone -------------

        ## PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        ## similar to diffusion policy, just one Linear, might upgrade to an MLP
        self.x_embedder = nn.Linear(in_features=self.transition_dim, out_features=hidden_size)

        ## plus 1 for the cls_token
        self.num_patches = self.c_traj_hzn // self.frame_stack + 1
        assert self.c_traj_hzn % self.frame_stack == 0
        
        # Will use fixed sin-cos embedding: (will be init later)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        ## Will be init later
        self.cls_token = nn.Parameter(data=torch.randn(1, 1, hidden_size,))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                     cond_dim=hidden_size) for _ in range(depth)
        ])
        
        # pdb.set_trace()
        ## final_layer of the all tokens
        self.final_layer = FinalLayer(hidden_size, 
                                out_channels=hidden_size, cond_dim=hidden_size,)

        assert out_dim == hidden_size, 'for now'
        ## we only care about the final cls-token
        # self.mlp_head = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            # nn.Linear(hidden_size, out_dim)
        # )

        ## -----------
        # pdb.set_trace()
        w_init_type = tjti_enc_config['w_init_type'] # 'dit1d') 
        if w_init_type == 'dit1d':
            self.initialize_weights()
        elif w_init_type == 'no':
            pass
        else:
            raise NotImplementedError

        # pdb.set_trace()
        utils.print_color(f'[DiT1D_Traj_Time_Encoder] {self.num_patches=}, '
                          f'{hidden_size=}, {depth=}, {out_dim=},', c='c')
        self.num_params = utils.report_parameters(self, topk=0)



    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        assert (self.num_patches - 1) * self.frame_stack == self.c_traj_hzn, 'for now'
        tmp_pos_arr = np.arange(self.num_patches, dtype=np.int32 )
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.hidden_size, pos=tmp_pos_arr)
        
        ## pos_embed will still be float32
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w) ## no need to reshape, since already a Linear
        nn.init.constant_(self.x_embedder.bias, 0)

        # pdb.set_trace()
        # Initialize the cls-token
        nn.init.normal_(self.cls_token.data, std=0.02)

        # pdb.set_trace()

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




    

    

    def forward(self, x, time,):
        """
        Forward pass of DiT-based Trajectory Encoder
        x: (B, H, Dim) a batch of trajs
        time: (B,) tensor of diffusion timesteps
        """

        if self.frame_stack > 1:
            x = einops.rearrange(x, "b (t fs) dim -> b t (fs dim)", fs=self.frame_stack)

        ## (B, H, D), e.g., [4, 160, 384]
        x_input_emb = self.x_embedder(x)
        b_s, _, _ = x.shape
        cls_tokens = einops.repeat(self.cls_token, '1 1 d -> b 1 d', b = b_s)


        # pdb.set_trace()
        x_input_emb = torch.cat( (cls_tokens, x_input_emb), dim=1 )
        ## (B, H, D)
        x_input_emb = x_input_emb + self.pos_embed

        # pdb.set_trace()


        ## ------------------------------------------------------
        ## ------------- obtain Condition feature ---------------

        

        ## ------------ create cond_feat for denoiser --------------

        if True:
            ## (B, tot_cond_dim)
            t_feat = self.t_embedder(time)
            c_feat_all = t_feat
        else:
            raise NotImplementedError

        # pdb.set_trace()

        ## ------------ Denoising BackBone ----------------

        x = x_input_emb ## prev: x.shape=B,H,obs_dim

        for block in self.blocks:
            x = block(x, c_feat_all)  # (B, T, hid_D)
        
        x = self.final_layer(x, c_feat_all)  # (B, T, obs_dim)
        
        # if self.frame_stack > 1:
            # x = einops.rearrange(x, "b t (fs dim) -> b (t fs) dim", fs=self.frame_stack)
        
        ## (B, out_dim)
        cls_latents = x[:, 0, :]
        # pdb.set_trace() ## check shape

        return cls_latents
    




if __name__ == '__main__':

    ## 7.45M Parameters
    model = DiT1D_Traj_Time_Encoder(
        c_traj_hzn=56,
        in_dim=15,
        out_dim=256,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        tjti_enc_config=dict(frame_stack=4),
    )

    ## 9.82M Parameters
    model = DiT1D_Traj_Time_Encoder(
        c_traj_hzn=56,
        in_dim=15,
        out_dim=256,
        hidden_size=256,
        depth=8,
        num_heads=4,
        mlp_ratio=4.0,
        tjti_enc_config=dict(frame_stack=4),
    )

    utils.report_parameters(model)

    model
    print('End')