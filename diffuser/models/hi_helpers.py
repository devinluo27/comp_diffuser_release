import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
import math

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class Hi_ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        ## important: Compared to Ori, add one layer and increase layer size
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), ## luo added, July 19
            nn.Mish(),
            nn.Linear(embed_dim * 2, out_channels), ## updated,  July 20
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)



class SinusoidalPosEmb_2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert x.ndim == 2
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        ## emb: (half_dim,)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        ## (B, Hzn, 1) * (1, 1, half_dim) --> (B, Hzn, half_dim)
        emb = x[:, :, None] * emb[None, None, :]
        ## (B, half_dim+half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# class MLP_InvDyn(nn.Module):
#     def __init__(self, ) -> None:
#         super().__init__(*args, **kwargs)


class MLP_InvDyn(nn.Module): # encoder
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        hidden_dim (list): [512, 256, 128] or [in, 256, 256, out]
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        module_list = []
        # (in, 512, 256, 128, out) ? out of date
        layer_dim = [self.input_dim,] + hidden_dim + [output_dim,]
        num_layer = len(layer_dim) - 1

        for i_l in range(num_layer):
            module_list.append(nn.Linear(layer_dim[i_l], layer_dim[i_l+1]))
            module_list.append(nn.ReLU())
        del module_list[-1] # no relu at last
        
        self.encoder = nn.Sequential(*module_list)
        
        from diffuser.utils import print_color
        print_color(f'[MLP_InvDyn]  {num_layer=}, {layer_dim=}')
        # pdb.set_trace()

            
    def forward(self, x):
        x = self.encoder(x)
        return x
    


import torch.nn.functional as F

class MLP_InvDyn_V2(nn.Module): # encoder
    def __init__(self, input_dim, output_dim, inv_m_config):
        '''
        hidden_dim (list): [512, 256, 128] or [in, 256, 256, out]
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        module_list = []
        
        hidden_dim = inv_m_config['inv_hid_dims']

        # (in, 512, 256, 128, out) ? out of date
        layer_dim = [self.input_dim,] + hidden_dim + [output_dim,]
        num_layer = len(layer_dim) - 1

        for i_l in range(num_layer):
            module_list.append(nn.Linear(layer_dim[i_l], layer_dim[i_l+1]))


            if inv_m_config['act_f'] == 'relu':
                act_fn_2 = nn.ReLU
            elif inv_m_config['act_f'] == 'Prelu':
                act_fn_2 = nn.PReLU
            else:
                assert False
            module_list.append( act_fn_2() )


            if inv_m_config['use_dpout'] and i_l < num_layer - 2:
                module_list.append( nn.Dropout(p=inv_m_config['prob_dpout']), )

        # pdb.set_trace()

        del module_list[-1] # no relu at last
        
        self.encoder = nn.Sequential(*module_list)
        
        from diffuser.utils import print_color
        print_color(f'[MLP_InvDyn]  {num_layer=}, {layer_dim=}')
        
        # pdb.set_trace()

            
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    def loss(self, x_t, x_t_1, a_t):

        # print(x_t.shape, x_t_1.shape, a_t.shape)
        
        # pdb.set_trace()

        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        pred_a_t = self.forward(x_comb_t)

        inv_loss = F.mse_loss(pred_a_t, a_t)

        # pdb.set_trace()

        return inv_loss, {}
        