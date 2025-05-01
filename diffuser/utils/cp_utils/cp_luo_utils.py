import numpy as np
import torch
from diffuser.utils.arrays import to_np

def extract_ovlp_from_full(x: torch.Tensor, len_ovlp_cd):
    """
    x: either np or tensor, a non-self version, that can be called from outside.
    """
    st_traj = x[:, :len_ovlp_cd, :]
    end_traj = x[:, -len_ovlp_cd:, :]
    if torch.is_tensor(st_traj):
        assert torch.is_tensor(end_traj)
        st_traj = st_traj.detach().clone()
        end_traj = end_traj.detach().clone()
    else:
        assert type(st_traj) == np.ndarray
        assert type(end_traj) == np.ndarray

    return st_traj, end_traj

def compute_ovlp_dist(trajs_list_un, len_ovlp_cd):
    '''
    actually if we do not do thresholding, normed trajs also seems fine, since we only do ranking
    Args:
        - trajs_list_un: a list of len n_comp [ unnorm trajs [B,H,D] ], should be np
    '''
    trajs_list = trajs_list_un
    num_tj = len(trajs_list)
    assert num_tj >= 2
    assert trajs_list[0].ndim == 3

    dist_all = []
    for i_tj in range(num_tj-1):
        traj_1 = trajs_list[i_tj]
        traj_2 = trajs_list[i_tj+1]
        assert traj_1.ndim == 3 and traj_2.ndim == 3
        ## B,H,C
        _, end_traj_1 = extract_ovlp_from_full(traj_1, len_ovlp_cd)
        st_traj_2, _ = extract_ovlp_from_full(traj_2, len_ovlp_cd)
        
        print(f'{end_traj_1.shape=}')

        # tmp_dist = np.linalg.norm( end_traj_1 - st_traj_2 ).item()
        mse_dist = (end_traj_1 - st_traj_2) ** 2
        ## (B,)
        mse_dist = np.mean(mse_dist, axis=(1,2))

        dist_all.append(mse_dist)

    ## (B, n_comp-1)
    dist_all = np.stack(dist_all, axis=1)
    print(f'{dist_all.shape=}')
    # print(dist_all)

    ## (B,) the avg dist of one sample
    dist_per_sam = dist_all.sum(axis=1)
    ## print(dist_per_sam)
    ## s_idxs[0] is the idx with smallest distance
    s_idxs = np.argsort(dist_per_sam)
    
    return s_idxs, dist_per_sam



def pick_top_n_trajs(trajs_list, s_idxs, top_n, ):
    """
    s_idxs: first one is the samllest dist one
    return:
        same structure as trajs_list, but pick out the smallest top_n
        trajs_list_topn: a list of len n_comp, each elem: [B,H,Dim]
    """
    ## '# of traj candidate should be equal to # of sort idxs'
    assert len(s_idxs) == len(trajs_list[0])
    assert top_n <= len(s_idxs)
    n_comp = len(trajs_list)
    trajs_list_topn = []
    for i_c in range(n_comp):
        ## B, hzn,dim
        trajs_list_topn.append(trajs_list[i_c][ s_idxs[:top_n],  ])
    return trajs_list_topn


def get_np_trajs_list(trajs_list, do_unnorm, normalizer):
    num_tj = len(trajs_list)
    ## to numpy
    trajs_list = [ to_np(trajs) for trajs in trajs_list ]
    ## unnormalize
    if do_unnorm:
        for i_tj in range(num_tj):
            trajs_list[i_tj] = normalizer.unnormalize(trajs_list[i_tj], 'observations')
    return trajs_list

def parse_seeds_str(seeds_str: str):
    '''return a list of int'''
    out = [int(sd) for sd in seeds_str.strip().split(',')]
    # print(f'{out=}')
    # if len(out) == 1:
        # out = out[0]
    return out