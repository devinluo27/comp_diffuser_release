import numpy as np
import diffuser.utils as utils
from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.datasets.normalization import DatasetNormalizer

class Traj_Blender:
    def __init__(self, diffusion: Stgl_Sml_GauDiffusion_InvDyn_V1, 
                        normalizer: DatasetNormalizer,
                        blend_type: str,
                        exp_beta=3,
                        ):
        self.exp_beta = exp_beta
        self.diffusion = diffusion
        self.blend_type = blend_type
        self.len_ovlp = self.diffusion.len_ovlp_cd
        self.hzn_step_size = self.diffusion.horizon - self.len_ovlp
        self.hzn = self.diffusion.horizon
        self.gap_len = self.hzn - 2 * self.len_ovlp
        self.normalizer = normalizer
        assert self.gap_len > 0

    def blend_traj_lists(self, trajs_list, do_unnorm):
        """
        trajs_list: list of len n_comp: [ (B,H,D),..., ]
        returns:
            - trajs_out: a list of [tot_hzn, dim]
        """
        ## to np and unnorm
        trajs_list = utils.get_np_trajs_list(trajs_list, do_unnorm=do_unnorm, 
                                             normalizer=self.normalizer)

        n_comp = len(trajs_list)
        b_s,_, dd = trajs_list[0].shape ## b h d

        tot_hzn = n_comp * self.diffusion.horizon - \
                    (n_comp - 1) * self.diffusion.len_ovlp_cd
        
        print(f'{tot_hzn=}')
        # trajs_out = np.zeros( shape=(b_s, tot_hzn, dd) ) ## NOTE: default is float64
        trajs_out = np.zeros( shape=(b_s, tot_hzn, dd), dtype=np.float32 ) ## Dec 2: changed to float32
        cnt_v = np.zeros_like(trajs_out)
        ## copy non-ovlp parts
        for i_c in range(n_comp):
            tjs_p_i = trajs_list[i_c]

            if i_c == 0:
                tmp_idx_1 = 0
                tmp_idx_2 = self.hzn_step_size
                ## B,hstep,dim
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, :self.hzn_step_size, :]
            elif i_c < n_comp - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.gap_len
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, self.len_ovlp:self.len_ovlp+self.gap_len, :]
                
            elif i_c == n_comp - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.hzn_step_size

                assert tmp_idx_2 == tot_hzn
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, self.len_ovlp:, :]

            cnt_v[ :, tmp_idx_1:tmp_idx_2, : ] += 1
            utils.print_color(f'{i_c=} {tmp_idx_1=}, {tmp_idx_2=}, {tot_hzn=}')

        ## handle and merge the ovlp parts
        for i_c in range(n_comp-1):
            tmp_idx_1 = (i_c + 1) * self.hzn_step_size
            tmp_idx_2 = tmp_idx_1 + self.len_ovlp

            ## b,sm_hzn,d
            tjs_p_i = trajs_list[i_c]
            _, end_tjs_i = self.diffusion.extract_ovlp_from_full(tjs_p_i)
            ## b,sm_hzn,d
            tjs_p_i_plus_1 = trajs_list[i_c+1]
            st_tjs_i_plus_1, _ = self.diffusion.extract_ovlp_from_full(tjs_p_i_plus_1)


            ## b,len_o,d
            trajs_blend = blend_2_np_trajs_23d(end_tjs_i, st_tjs_i_plus_1, 
                                               self.blend_type, self.exp_beta)

            trajs_out[:, tmp_idx_1:tmp_idx_2, :] = trajs_blend
            cnt_v[:, tmp_idx_1:tmp_idx_2, :] += 1

            utils.print_color(f'{i_c=} {tmp_idx_1=}, {tmp_idx_2=}')
        assert tmp_idx_2 == (tot_hzn - self.hzn_step_size)


        assert (cnt_v == 1).all()

        

        return trajs_out





def blend_2_np_trajs_23d(traj_1: np.ndarray, traj_2: np.ndarray, blend_type='exponential', beta=5):
    """
    ** Only takes in the ovlp parts **, blend full traj_1 and traj_2
    ** Blend for multiple dim,
    ----[----
         ----]-----

    Parameters:
    - traj_1: np.ndarray, shape (N1, D), first trajectory positions
    - traj_2: np.ndarray, shape (N2, D), second trajectory positions
    - blend_type: str, type of blending function ('exponential', 'cosine', 'linear', 'smoothstep')
    - beta: float, parameter for the exponential blending function (controls sharpness)

    Returns:
    - traj_blend: np.ndarray, blended trajectory positions
    """

    # assert traj_1.ndim == 2 and traj_1.shape[1] == 1
    assert traj_1.ndim in [2,3] and traj_2.ndim in [2,3] 
    assert traj_1.shape and traj_2.shape

    if traj_1.ndim == 2:
        len_tj, _ = traj_1.shape
    else:
        b_s, len_tj, _ = traj_1.shape
    # Overlapping region from t = 8 to t = 10
    t_overlap_start = 0
    t_overlap_end = len_tj - 1
    t_overlap = np.arange(0, len_tj) ## 1D

    ## Blending function selection, Checked, correct formula
    if blend_type in ['exponential', 'exp']:
        # Exponential blending function
        def w(t):
            exponent = -beta * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            return (np.exp(exponent) - np.exp(-beta)) / (1 - np.exp(-beta))
    elif blend_type == 'cosine':
        # Cosine blending function
        def w(t):
            return 0.5 * (1 + np.cos(np.pi * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)))
    elif blend_type == 'linear':
        # Linear blending function
        def w(t):
            return 1 - (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
    elif blend_type == 'smoothstep':
        # Smoothstep blending function
        def w(t):
            x = (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            return 1 - (3 * x**2 - 2 * x**3)
    else:
        raise ValueError("Invalid blending function. Choose 'exponential', 'cosine', 'linear', or 'smoothstep'.")

    ## Compute weights, np 1d, from 0 to 1
    ## weights = w(t_overlap)[:, np.newaxis]  # Column vector for broadcasting
    weights = w(t_overlap)  # Column vector for broadcasting
    if traj_1.ndim == 2:
        weights = weights[:, None] ## (len_tj, 1)
    elif traj_1.ndim == 3:
        weights = weights[None, :, None,] ## (1, len_tj, 1)
    
    # print(f'{weights[(0,-1),]=},{weights.shape=}')

    ## Blend the overlapping region
    traj_blend = weights * traj_1 + (1 - weights) * traj_2

    # print(f'{traj_blend.shape=}')

    return traj_blend
