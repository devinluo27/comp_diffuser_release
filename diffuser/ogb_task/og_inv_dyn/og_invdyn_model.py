import torch.nn as nn
import torch, math, pdb
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution
from diffuser.ogb_task.og_inv_dyn.og_invdyn_helpers import OgB_MLP, ogb_get_default_init_unif_a
import diffuser.utils as utils


class MLP_InvDyn_OgB_V3(nn.Module):
    """
    Like OgB_GCActor
    Goal-conditioned actor in PyTorch mirroring your JAX/Flax code.

    Args:
        input_dim: Dimension of the concatenated observation-goal input.
        hidden_dims: Sequence of hidden layer dimensions.
        action_dim: Dimensionality of the action space.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash samples with tanh.
        state_dependent_std: Use a state-dependent standard deviation network.
        const_std: If True, log_std is always zero (unless state_dependent_std=True).
        final_fc_init_scale: Scaling factor for the final linear layer initialization.
        gc_encoder: Optional module that encodes observation+goal into a single tensor.
    """
    def __init__(
        self,
        input_dim,
        # hidden_dims,
        action_dim,
        obs_dim, ## just for reference
        ####
        act_net_config={},
        inv_m_config={},
    ):
        super().__init__()
        self.obs_dim = obs_dim

        ## --- These Values Probably No Need to Change ---
        ## GCBC: False
        self.tanh_squash = inv_m_config.get('tanh_squash', False)
        ## GCBC: False
        self.state_dependent_std = inv_m_config.get('state_dependent_std', False)
        ## GCBC: True
        self.const_std = inv_m_config.get('const_std', True)

        assert self.tanh_squash == False and self.state_dependent_std == False
        assert self.const_std == True

        self.log_std_min = inv_m_config.get( 'log_std_min', -5.0 )
        self.log_std_max = inv_m_config.get( 'log_std_max', 2.0 )
        ## GCBC: 1e-2
        self.final_fc_init_scale = inv_m_config['final_fc_init_scale']
        
        ## --- Jan 1: to encode the goal state to a latent first ---
        gl_enc_config = inv_m_config.get('gl_enc_config', None)
        if gl_enc_config is not None:
            in_goal_dim = input_dim - obs_dim
            self.gl_encoder = OgB_MLP(
                input_dim=in_goal_dim,
                output_dim=None, 
                activate_final=False, 
                **gl_enc_config
            )
            ## [current state (concat) latent goal]
            input_dim = obs_dim + self.gl_encoder.output_dim
        else:
            self.gl_encoder = None
        

        hidden_dims = inv_m_config['hidden_dims']
        
        # pdb.set_trace()

        self.actor_net = OgB_MLP(
            input_dim, hidden_dims, output_dim=None,
            activate_final=True, mlp_config=act_net_config,)
        

        print(f'[OgB_GCActor] {hidden_dims=}')
        utils.print_color(f'[OgB_GCActor] {self.actor_net=}')

        # pdb.set_trace()


        # -------- Customized Init Func -----------
        # Mean network
        self.mean_net = nn.Linear(hidden_dims[-1], action_dim, bias=True)
        tmp_a, tmp_b = ogb_get_default_init_unif_a(
            self.mean_net.weight, scale=self.final_fc_init_scale)
        nn.init.uniform_(self.mean_net.weight, a=tmp_a, b=tmp_b)
        nn.init.zeros_(self.mean_net.bias)
        # -----------------------------------------

        utils.print_color(f'[OgB_GCActor] {self.mean_net=}')

        ## All False, because const_std is True By default
        # Log-std network (only if state_dependent_std is True)
        if self.state_dependent_std:
            assert False
            self.log_std_net = nn.Linear(hidden_dims[-1], action_dim)
            orthogonal_init_(self.log_std_net, scale=self.final_fc_init_scale)
        else:
            # If it's not state-dependent but also not constant, we have a single trainable param
            if not self.const_std:
                assert False
                self.log_stds = nn.Parameter(torch.zeros(action_dim))
        

        self.is_out_dist = inv_m_config['is_out_dist'] ## Try False
        self.train_temp = inv_m_config.get('train_temp', 1.0)
        self.eval_temp = inv_m_config.get('eval_temp', 0.0)



    def forward(self, observations, goals, temperature=1.0):
        """
        Forward pass to compute a distribution over actions given observations (and possibly goals).
        Returns a torch.distributions distribution object.
        """
        # Encode obs+goal if gc_encoder is defined, else just cat them
        if self.gl_encoder is not None:
            ## just encode the goal to a latent
            gl_feat = self.gl_encoder(goals)
            inputs = torch.cat((observations, gl_feat), dim=-1)
            # pdb.set_trace()

        else:
            if goals is not None:
                inputs = torch.cat((observations, goals), dim=-1)
            else:
                inputs = observations

        # pdb.set_trace() ## check shape, e.g., torch.Size([B, 31=29+2])

        # Pass through the actor MLP
        features = self.actor_net(inputs)

        # Compute the means
        means = self.mean_net(features)

        if self.is_out_dist:
            ## Dec 25: Should Be Usable After Some Check
            # Compute the log_stds
            if self.state_dependent_std:
                assert False
                log_stds = self.log_std_net(features)
            else:
                if self.const_std:
                    # All zeros => std=1
                    log_stds = torch.zeros_like(means)
                else:
                    assert False
                    # A learned parameter vector, broadcast to the same shape as means
                    log_stds = self.log_stds.unsqueeze(0).expand_as(means)

            # Clamp (equivalent to jnp.clip in JAX)
            log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)

            pdb.set_trace() ## check dist

            # Build a diagonal Gaussian distribution
            base_dist = Normal(loc=means, scale=torch.exp(log_stds) * temperature)
            dist = Independent(base_dist, 1)  # Make it a multivariate diagonal

            # If we want tanh-squashed actions, use a Tanh transform. (PyTorch >=1.7)
            if self.tanh_squash:
                raise NotImplementedError

            return dist
        else:
            ## Dec 25 Simplified Version
            return means
    
    def loss(self, x_t, x_t_1, a_t):

        # print(x_t.shape, x_t_1.shape, a_t.shape)
        
        if self.is_out_dist:
            pred_dist = self.forward(x_t, x_t_1)
            log_prob = pred_dist.log_prob( a_t )
            inv_loss = -log_prob.mean()
        else:
            # pdb.set_trace()
            # x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            pred_a_t = self.forward(x_t, x_t_1,)
            inv_loss = F.mse_loss(pred_a_t, a_t)


        # pdb.set_trace()

        return inv_loss, {}
        