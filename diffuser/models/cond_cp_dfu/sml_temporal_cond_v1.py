import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
import diffuser.utils as utils
from diffuser.models.hi_helpers import Hi_ResidualTemporalBlock
# PositionalEncoding2D,
# Conv1dBlock_dd,
from diffuser.models.helpers import (
    SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock,
)
import numpy as np
from .sml_temporal_dd_v1 import ResidualTemporalBlock_dd
from .sml_helpers import Traj_Time_Encoder
# --------------------------------------------------------

## WCond: Condtioned on Walls
class Unet1D_TjTiCond_V1(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        base_dim=32, # may use 64
        dim_mults=(1, 2, 4, 8),
        time_dim=32,
        network_config={},
    ):
        """
        NOTE energy_mode might be implemented only when cat_t_w=True,
        e.g. some zero init is not impled in residual block.
        wall_embed_dim: embed dim of all walls

        """
        super().__init__()

        ## dim=64 [2,64*1,64*4,64*8]
        dims = [transition_dim, *map(lambda m: base_dim * m, dim_mults)]
        ## [(64,128), (128,256), (256,512)]
        in_out = list(zip(dims[:-1], dims[1:]))
        utils.print_color(f'[ models/Unet1D_TjTiCond_V1 ] Channel dimensions: {in_out}', c='c')

        ## --------- init MLP for time / wall ---------
        ## cat the vector embedding of time and wall before feeding to the MLP
        self.cat_t_w = network_config['cat_t_w'] ## True
        self.resblock_ksize = network_config.get('resblock_ksize', 5) # kernel size for residual block
        self.use_downup_sample = network_config.get('use_downup_sample', True)


        self.st_ovlp_model_config = network_config['st_ovlp_model_config']
        self.end_ovlp_model_config = network_config['end_ovlp_model_config']
        # self.st_ovlp_model_config['in_dim'] = transition_dim
        
        
        self.ext_cond_dim = network_config['ext_cond_dim']
        self.st_ovlp_model = Traj_Time_Encoder(**self.st_ovlp_model_config)
        self.end_ovlp_model = Traj_Time_Encoder(**self.end_ovlp_model_config)
        ##
        wall_embed_dim = self.st_ovlp_model.out_dim + self.end_ovlp_model.out_dim
        # pdb.set_trace()
        assert wall_embed_dim == self.ext_cond_dim

        assert self.use_downup_sample and self.resblock_ksize == 5, 'the default settings'
        
        if self.cat_t_w:
            tot_cond_dim = time_dim + wall_embed_dim
        else:
            raise NotImplementedError
            time_dim = dim

        ## set param used in ebm
        self.energy_mode = network_config['energy_mode']
        if self.energy_mode:
            raise NotImplementedError
            mish = False
            act_fn = nn.SiLU()

            self.energy_param_type = network_config['energy_param_type'] ## should use this line
            if 'L2' in self.energy_param_type:
                self.conv_zero_init = network_config.get('conv_zero_init', False)
               
            else: 
                raise NotImplementedError()
            
            print(f'[ models/temporal_cond ] conv_zero_init {self.energy_param_type} {self.conv_zero_init}')
        else:
            mish = True
            act_fn = nn.Mish()
            self.conv_zero_init = False

        ## TODO: From Here Apr 19

        # self.time_mlp = nn.Sequential(
        #     SinusoidalPosEmb(time_dim),
        #     nn.Linear(time_dim, time_dim * 4),
        #     act_fn,
        #     nn.Linear(time_dim * 4, time_dim),
        # )
        ## Luo: just make it deeper
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            act_fn,
            nn.Linear(time_dim * 2, time_dim * 2),
            act_fn,
            nn.Linear(time_dim * 2, time_dim),
        )
        # pdb.set_trace() ## check dim

        self.network_config = network_config


        ## default no dropout
        # self.concept_drop_prob = network_config['concept_drop_prob'] # -1.0
        self.last_conv_ksize = network_config.get('last_conv_ksize', 1) # 1 is more stable than 5
        self.force_residual_conv = network_config.get('force_residual_conv', False)
        self.time_mlp_config = network_config.get('time_mlp_config', False)
        resblock_config = dict(force_residual_conv=self.force_residual_conv,
                               time_mlp_config=self.time_mlp_config)
        
        assert not self.force_residual_conv, 'must be False'
        assert self.last_conv_ksize == 1, '1 is from diffuser'


        # print(f'[TemporalUnet_WCond] concept_drop_prob: {self.concept_drop_prob}')
        utils.print_color(f'[TemporalUnet_WCond] {time_dim=}, {tot_cond_dim=}, {self.ext_cond_dim=}')
        # pdb.set_trace()
        self.input_t_type = '1d'
        
        


        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        ## num_resolutions is the number of layer in UNet?
        print('[TemporalUnet_WCond]: in_out: ', in_out,)

        res_block_type = ResidualTemporalBlock_dd if self.cat_t_w else None

        

        self.down_times = network_config.get('down_times', 1e5)
        utils.print_color(f'[Unet down_times] {self.down_times}', c='c')
        ## default in_out: [(64,128), (128,256), (256,512)]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind >= self.down_times

            self.downs.append(nn.ModuleList([
                res_block_type(dim_in, dim_out, embed_dim=tot_cond_dim, horizon=horizon,wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init,resblock_config=resblock_config, kernel_size=self.resblock_ksize), # ks should be 5 by default
                res_block_type(dim_out, dim_out, embed_dim=tot_cond_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init,resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                Downsample1d(dim_out) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = res_block_type(mid_dim, mid_dim, embed_dim=tot_cond_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize)
        self.mid_block2 = res_block_type(mid_dim, mid_dim, embed_dim=tot_cond_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind < ( num_resolutions - self.down_times - 1)

            ##? Eg. dim_out:4, dim_in:8, dim_out*2 because we concat residual 
            self.ups.append(nn.ModuleList([
                res_block_type(dim_out * 2, dim_in, embed_dim=tot_cond_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                res_block_type(dim_in, dim_in, embed_dim=tot_cond_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                Upsample1d(dim_in) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2
        
        ## -- Ordinary Diffusion Setup --
        if not self.energy_mode:
            self.final_conv = nn.Sequential(
                Conv1dBlock(base_dim, base_dim, kernel_size=self.resblock_ksize), # 5
                nn.Conv1d(base_dim, transition_dim, 1),
            )
        ## -- Energy Diffusion Parameterization Setup --
        elif self.energy_param_type == 'L2':
            raise NotImplementedError
            self.final_conv = nn.Sequential(
                Conv1dBlock_dd(dim, dim, kernel_size=5, mish=mish, conv_zero_init=False),
                nn.Conv1d(dim, transition_dim, 1),
            )
        else:
            raise NotImplementedError()



    def forward(self, x, time,
                tj_cond: dict,
                # traj_st_ovlp,
                # traj_end_ovlp,
                #  walls_loc, 
                # use_dropout=True, 
                force_dropout=False, half_fd=False,):
        '''
            x : [ batch x horizon x transition ]
            time: [batch,]
            walls_loc: [batch, 6], 2D
            half_fd: drop the conditions for the second half in the input batch 
        '''
        if self.energy_mode:
            x.requires_grad_(True)
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')

        t_feat = self.time_mlp(time) ## e.g., (B,64) ## TODO:
        
        # pdb.set_trace()

        ## Encode Wall Locations to a feature vector w
        # w = self.wallLoc_encoder(walls_loc)
        ## what we want is like [B, dim], use the cls_token if vit1d

        ## obtain feature ## TODO: From Here
        st_ovlp_is_drop = tj_cond['st_ovlp_is_drop']
        if st_ovlp_is_drop is not None: ##
            st_ovlp_feat = self.st_ovlp_model(tj_cond['st_ovlp_traj'], 
                                    time=tj_cond['st_ovlp_t'])
            assert len(st_ovlp_is_drop) == len(st_ovlp_feat)
            assert st_ovlp_is_drop.dtype == bool ## a numpy array
            st_ovlp_feat[ st_ovlp_is_drop ] = 0.
        else:
            ## no cond if None
            # st_ovlp_feat = torch.zeros_like(st_ovlp_feat)
            st_ovlp_feat = torch.zeros( (x.shape[0], self.st_ovlp_model.out_dim), device=x.device)

        
        if tj_cond['end_ovlp_is_drop'] is not None:
            end_ovlp_feat = self.end_ovlp_model(tj_cond['end_ovlp_traj'],
                                                time=tj_cond['end_ovlp_t'])
            end_ovlp_feat[ tj_cond['end_ovlp_is_drop'] ] = 0.
        else:
            # end_ovlp_feat = torch.zeros_like(end_ovlp_feat)
            end_ovlp_feat = torch.zeros( (x.shape[0], self.end_ovlp_model.out_dim), device=x.device)


        # pdb.set_trace()
        






        ## drop concept only when training, rand uniform [0, 1)
        # if use_dropout:
            # assert self.training
            # b = w.shape[0]
            # w[np.random.rand(b,) < self.concept_drop_prob] = 0.


        if force_dropout:
            # assert False
            assert not self.training
            if half_fd:
                b_s = len(st_ovlp_feat)
                # drop the second half
                assert b_s % 2 == 0
                st_ovlp_feat[int(b_s//2):] = 0. * st_ovlp_feat[int(b_s//2):] 
                ## FIXED: Oct 9 Probably a important:
                end_ovlp_feat[int(b_s//2):] = 0. * end_ovlp_feat[int(b_s//2):] 
            else:
                assert False
                w = 0. * w
            # pdb.set_trace()

        if self.cat_t_w:
            ## e.g., B, time_dim+128+128
            t_feat = torch.cat([t_feat, st_ovlp_feat, end_ovlp_feat], dim=-1)
        
        h = []

        # pdb.set_trace()

        for resnet, resnet2, downsample in self.downs:

            x = resnet(x, t_feat)
            x = resnet2(x, t_feat)
            h.append(x)
            x = downsample(x)

        # print(f'after downs: {x.shape}')

        x = self.mid_block1(x, t_feat)
        x = self.mid_block2(x, t_feat)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t_feat)
            x = resnet2(x, t_feat)
            x = upsample(x)

        # print(f'after ups: {x.shape}')

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        ## energy_mode will return inside
        if self.energy_mode:
            assert False, 'not used'
            unet_out = x # B, horizon, dim
            if self.energy_param_type in ['L2',]:
                energy_norm = 0.5 * unet_out * unet_out # should not have neg sign
                energy_norm = energy_norm.sum(dim=(1,2))
            else: 
                raise NotImplementedError()

            
            if not self.training:
                eps = torch.autograd.grad([energy_norm.sum()],[x_inp],create_graph=False)[0]
                # print('energy_norm.sum()', energy_norm.sum())
            else:
                engy_batch = energy_norm.sum()
                eps = torch.autograd.grad([engy_batch,],[x_inp],create_graph=True)[0]
                return eps, engy_batch.detach()

                 
            return eps

        ## final output: B H dim
        # print(f'final output: {x.shape}')

        return x
