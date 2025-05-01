from collections import namedtuple
# import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories_invdyn = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')

class Policy_InvDyn:

    def __init__(self, diffusion_model, 
                 normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):


        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions)
        ## prev_obs, next_obs; ([1, 383, 8])
        obs_comb = torch.cat([sample[:, :-1, :], sample[:, 1:, :]], dim=2)
        obs_flat = einops.rearrange(obs_comb, 'b h d -> (b h) d')
        ## [383, 2]
        actions_flat = self.diffusion_model.inv_model(obs_flat)
        # pdb.set_trace()
        actions = einops.rearrange(actions_flat, '(b h) d -> b h d', b=sample.shape[0])
        sample = utils.to_np(sample)
        actions = utils.to_np(actions)

        # pdb.set_trace()

        ## extract action [ batch_size x horizon x transition_dim ]
        # actions = sample[:, :, :self.action_dim]



        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, 0:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories_invdyn(actions, observations)
        return action, trajectories
        # else:
        #     return action
