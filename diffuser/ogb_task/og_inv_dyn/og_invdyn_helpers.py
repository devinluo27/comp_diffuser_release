import torch.nn as nn
import torch, math
import torch.nn.functional as F

def ogb_get_default_init_unif_a(weight: torch.Tensor, scale=1.0):
    """ Implemented following default_init in networks.py
    Default kernel initializer.
    ###
    remember to init bias to 0
    nn.init.zeros_(tensor)
    """
    fan_out, fan_in = weight.data.shape
    print(f'[ogb_get_default_init_unif_a] {fan_out=}, {fan_in=}')
    avg_fan = (fan_in + fan_out) / 2
    init_unif_a = math.sqrt( scale / avg_fan )
    return -init_unif_a, init_unif_a



class OgB_MLP(nn.Module): # encoder
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list,
                 output_dim, ## int or None 
                 activate_final, 
                 mlp_config):
        '''
        This is just a helper model for the OGBench GC Policy
        Args:
            hidden_dims (list): [512, 512, 512] or [in, 256, 256, out]
            final_fc_init_scale: 1e-2
        '''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        module_list = []
        
        # final_fc_init_scale = mlp_config['final_fc_init_scale']

        if output_dim is None:
            layer_dim = [self.input_dim,] + hidden_dims
            self.output_dim = hidden_dims[-1]
        else:
            # (in, 512, 256, 128, out) ? out of date
            layer_dim = [self.input_dim,] + hidden_dims + [output_dim,]
        
        ## Jan 8 NEW, default: the last 2 layers do not have dropout
        n_dpout_until = mlp_config.get('n_dpout_until', 2) 
        assert n_dpout_until in [2,1,0]

        num_layer = len(layer_dim) - 1

        for i_l in range(num_layer):
            tmp_linear = nn.Linear(layer_dim[i_l], layer_dim[i_l+1])
            nn.init.zeros_(tmp_linear.bias)

            ## --- Customized Init Func, No Need ---
            # tmp_a, tmp_b = ogb_get_default_init_unif_a(
            #     tmp_linear.weight, scale=final_fc_init_scale)
            # nn.init.uniform_(tmp_linear.weight, a=tmp_a, b=tmp_b)
            ## -------------------------------------

            module_list.append( tmp_linear )

            if mlp_config['act_f'] == 'relu':
                assert False
                act_fn_2 = nn.ReLU
            elif mlp_config['act_f'] == 'Prelu':
                assert False
                act_fn_2 = nn.PReLU
            elif mlp_config['act_f'] == 'gelu':
                act_fn_2 = nn.GELU
            else:
                assert False
            module_list.append( act_fn_2() )

            if mlp_config['use_dpout'] and i_l < num_layer - n_dpout_until:
                # assert False, 'bug free, but not used.'
                module_list.append( nn.Dropout(p=mlp_config['prob_dpout']), )

        # pdb.set_trace()
        if not activate_final:
            del module_list[-1] # no relu at last
        
        self.encoder = nn.Sequential(*module_list)
        self.mlp_config = mlp_config
        self.n_dpout_until = n_dpout_until
        
        # from diffuser.utils import print_color
        # print_color(f'[MLP_InvDyn_OgB_V3]  {num_layer=}, {layer_dim=}')
            
    def forward(self, x):
        x = self.encoder(x)
        return x
    
   
   

from diffuser.utils import load_config, get_latest_epoch
import diffuser.utils as utils


def ogb_load_invdyn_maze_v1(loadpath: str, epoch='latest', device='cuda:0', ld_config={}):
    model_config = load_config(loadpath, 'model_config.pkl')

    model = model_config()
    ema_model = model_config()

    # ema_diffusion = diffusion_config(model)
    if epoch == 'latest':
        epoch = get_latest_epoch([loadpath,])
    ckpt_path = f'{loadpath}/state_{epoch}.pt'
    ckpt_data = torch.load(ckpt_path, weights_only=False)

    # trainer.load(epoch)

    utils.print_color(
        f'\n[ utils/serialization ] Loading Ours V1 Diffuser Inv model epoch: {epoch}\n', c='c')
    
    model.load_state_dict( ckpt_data['model'] )
    ema_model.load_state_dict( ckpt_data['ema'] )

    model.eval()
    ema_model.eval()
    utils.freeze_model(model)
    utils.freeze_model(ema_model)

    # pdb.set_trace()
    
    return model, ema_model, epoch


