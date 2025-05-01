import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        ## emb: (half_dim,)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        ## (B, 1) * (1, half_dim) --> (B, half_dim)
        emb = x[:, None] * emb[None, :]
        ## (B, half_dim+half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def zero_module(module, do_zero):
    """
    Used for energy parameterization
    Zero out the parameters of a module and return it.
    """
    if do_zero:
        for p in module.parameters():
            p.data.fill_(0)

    return module

class Conv1dBlock_dd(nn.Module):
    '''
        Conv1d --> GroupNorm (8 /32) --> (Mish / SiLU)
        ## checkparam groupnorm, n_groups
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8, conv_zero_init=False):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            zero_module( nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2), conv_zero_init ),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)



#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    """
    usually return (B, 1, 1)
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_2d(a, t, x_shape):
    """
    extract to t, to two dimension, e.g., return (B, H, 1)
    """
    assert a.ndim == 1 and t.ndim == 2
    b, h, *_ = t.shape
    ## NOTE: when t is also tensor, will create a new tensor
    out = a[t]
    # pdb.set_trace()
    out = out.reshape(b, h, *((1,) * (len(x_shape) - 2)))
    ## out: B, H, 1
    # pdb.set_trace()

    return out

Two_Power_20 = 2**20
def tensor_randint(low: torch.Tensor, high: torch.Tensor, size):
    '''
    this method allows generating randint tensor where range is defined by a tensor
    '''
    assert low.shape == high.shape == size
    assert (low < high).all()
    assert (high < Two_Power_20).all()
    ## [0 to (high - low - 1)] + low
    return torch.randint( Two_Power_20, size=size, dtype=torch.long) % (high - low) + low


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        ## loss just for the 0th action
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


class WeightedLoss_L2_V2(nn.Module):
    '''
    Added by luo, July 25 2024,
    ** support inputing a training time loss weight **
    '''
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ, ext_loss_w=1.):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
            ext_loss_w: tensor [B,H,1] or 1.
        '''

        loss = ext_loss_w * self._loss(pred, targ)
        # pdb.set_trace()

        weighted_loss = (loss * self.weights).mean()
        ## loss just for the 0th action
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')



class WeightedLoss_L2_InvDyn_V3(nn.Module):
    '''
    Added by luo, July 28 2024,
    ** support inputing a training time loss weight **
    '''
    def __init__(self, weights,):
        super().__init__()
        self.register_buffer('weights', weights)


    def forward(self, pred, targ, ext_loss_w=1.):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
            ext_loss_w: tensor [B,H,1] or 1.
        '''
        ## B,H,1 * B,H,D
        loss = ext_loss_w * self._loss(pred, targ)
        # pdb.set_trace()
        ## can auto boardcast, weights: (B,H,dim) * (H,dim) -> (B,H,dim)
        weighted_loss = (loss * self.weights).mean()

        return weighted_loss, {} # {'a0_loss': 0.}

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')





## ----------- Decision Diffuser Baseline --------------

class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}

class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


## -----------------------------------------------------







Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'l2_v2': WeightedLoss_L2_V2,
    'l2_inv_v3': WeightedLoss_L2_InvDyn_V3,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
    ## dd baseline
    'state_l2': WeightedStateL2,
    
}
