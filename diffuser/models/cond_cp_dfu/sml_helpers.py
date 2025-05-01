import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import diffuser.utils as utils
import pdb

from diffuser.models.helpers import (
    SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock,
)
from diffuser.models.hi_helpers import Hi_ResidualTemporalBlock


class Traj_Time_Encoder(nn.Module):

    def __init__(
        self,
        c_traj_hzn,
        in_dim, ## same as transition_dim
        base_dim, # 32, ## for cnn1d base
        dim_mults, # (1, 2, 4, 8),
        time_dim, #=32, ## time embedding
        out_dim, # 256,
        tjti_enc_config, # {}
    ):
        """
        This Neural Network is used to encode the noisy trajectory chunk to a latent.
        The resulting latent will be passed as a condition to the denoiser model.
        Params:
            - in_dim: input dim, e.g., for maze2d, should be 2
            - base_dim: cnn base dim
            - dim_mults: multiplications to increase the feature dim in CNN
            - time_dim: dim for the time emb
            - out_dim: dim for the final output latent
            - tjti_enc_config: hyperparam for the model
        """
        super().__init__()

        dims = [in_dim, *map(lambda m: round(base_dim * m), dim_mults)] ## dec 22, add round
        in_out = list(zip(dims[:-1], dims[1:]))
        utils.print_color(f'[ models/Traj_Time_Encoder ] Channel dimensions: {in_out}')

        self.tjti_enc_config = tjti_enc_config
        self.c_traj_hzn = c_traj_hzn
        horizon = c_traj_hzn

        ### --- originally time is a 1D (B,) tensor, but now 2D (B, H) ----
        self.t_seq_encoder_type = tjti_enc_config['t_seq_encoder_type']
        self.cnn_out_dim = tjti_enc_config['cnn_out_dim']
        self.f_conv_ks = tjti_enc_config['f_conv_ks']
        self.final_mlp_dims = tjti_enc_config['final_mlp_dims']
        self.mid_conv_ks = tjti_enc_config.get('mid_conv_ks', 5)
        # pdb.set_trace()
        
        utils.print_color(f'TjTi Enc: {self.cnn_out_dim=}, {self.final_mlp_dims=}', c='y')
        
        if self.t_seq_encoder_type == 'mlp':
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.Mish(),
                nn.Linear(time_dim * 4, time_dim),
            )
        elif self.t_seq_encoder_type == 'vit1d':
            raise NotImplementedError()
        else: 
            raise NotImplementedError()
        
        ### ----------------------------------------------------------------

        ## class type of resiudal block
        res_block_type = Hi_ResidualTemporalBlock

        self.downs = nn.ModuleList([])

        num_resolutions = len(in_out)

        utils.print_color('TjTi Enc: unet: ', in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                res_block_type(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                res_block_type(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2
            # cur_dim_out = dim_out

        ## 32,64,96,128
        ## TODO Attention Layer Maybe, July 20
        mid_dim = dims[-1]
        self.mid_block1 = res_block_type(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, kernel_size=self.mid_conv_ks)
        self.mid_block2 = res_block_type(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, kernel_size=self.mid_conv_ks)

        ## A. one separate cnn
        self.final_conv = nn.Sequential(
            Conv1dBlock(mid_dim, mid_dim, kernel_size=self.f_conv_ks),
            nn.Conv1d(mid_dim, self.cnn_out_dim, 1),
        )

        self.last_hzn = horizon
        self.f_mlp_in_dim = self.last_hzn * self.cnn_out_dim
        utils.print_color(f'TjTi Enc: {mid_dim=}, {self.last_hzn=}, {self.f_mlp_in_dim=}', c='y')
        
        # pdb.set_trace()
        ## for older code, just ignore this assert
        # assert self.cnn_out_dim == mid_dim, 'added on Dec 22, maynot be compatible to all prev code'

        assert self.f_mlp_in_dim < self.final_mlp_dims[0] / 0.95, 'ensure mlp is large enough'
        self.final_mlp_dims = [self.f_mlp_in_dim, *self.final_mlp_dims]
        self.out_dim = out_dim ## final output of the encoder model
        assert self.out_dim == self.final_mlp_dims[-1]
        
        f_mlp_in_outs = list(zip(self.final_mlp_dims[:-1], self.final_mlp_dims[1:]))
        f_mlp_n_layer = len(f_mlp_in_outs)
        
        utils.print_color(f'TjTi Enc: {self.f_mlp_in_dim=}, {f_mlp_in_outs=}', c='c')

        self.f_mlp_blocks = []
        for ind, (dim_in, dim_out) in enumerate(f_mlp_in_outs):
            self.f_mlp_blocks.append(nn.Linear( dim_in, dim_out ))
            is_last = ind >= ( f_mlp_n_layer - 1)
            if not is_last:
                self.f_mlp_blocks.append( nn.Mish() )


        ## B. one separate mlp to reduce dimension
        ## back to a 1d vector as condition signal
        self.final_mlp =  nn.Sequential(
            *self.f_mlp_blocks
        )

        # pdb.set_trace()
        if tjti_enc_config.get('w_init_type', None) == 'dit1d':
            self.init_for_dit1d()
            pdb.set_trace() ## maybe not do this

    def forward(self, x, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        ## to (B, dim, hzn), e.g., B 6 384
        x = einops.rearrange(x, 'b h t -> b t h')

        ## ------- encode diffusion time --------
        # pdb.set_trace() ## check time, should be (B,H,)
        ## B, tim_dim
        t_feat = self.time_mlp(time)
        ## --------------------------------------


        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t_feat)
            x = resnet2(x, t_feat)
            h.append(x)
            x = downsample(x)

        # pdb.set_trace()

        x = self.mid_block1(x, t_feat)
        x = self.mid_block2(x, t_feat)

        ## B, dim=128, 6
        x = self.final_conv(x)

        # pdb.set_trace()
        ## B,6,128
        x = einops.rearrange(x, 'b t h -> b h t')

        ## B, h, self.cnn_out_dim
        assert x.shape[1:] == (self.last_hzn, self.cnn_out_dim)
        ## B,768
        x_flat = torch.flatten(x, start_dim=1, end_dim=2)
        # pdb.set_trace()
        x_out = self.final_mlp(x_flat)

        # pdb.set_trace()

        return x_out


    def init_for_dit1d(self):
        # Initialize the layer with a smaller value;
        # Added by Luo, not sure if it is helpful, but seems like we need a smaller init value
        
        ## NOTE: if std=0.02, for mid_block sum, ori: 1600, dit_init: 1300
        ## torch.abs(self.downs[0][0].blocks[0].block[0].weight).sum()
        ## torch.abs(self.mid_block1.blocks[0].block[0].weight).sum()
        pdb.set_trace() ## self.mid_block1.blocks[0].block[0].weight

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=self.tjti_enc_config['init_std']) ## in dit: 0.02?
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
            elif isinstance(module, nn.Conv1d):
                torch.nn.init.normal_(module.weight, std=self.tjti_enc_config['init_std'])
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pdb.set_trace()


