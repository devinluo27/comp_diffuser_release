import numpy as np
import pdb

def m2d_rand_sample_probs(start_pos_list, gl_pos_list, rs_cfg):
    """
    densely loop through all possibe pairs of start / goal, 
    note that we include the velocity in the start_state
    """
    out_data = dict(start_state=[], goal_pos=[])
    for i_st, base_st_pos in enumerate(start_pos_list):
        for i_gl, base_gl_pos in enumerate(gl_pos_list):
            for i_ep in range(rs_cfg['n_probs_per_pair']):
                p_low, p_high = rs_cfg['r_pos_range'] ## -0.3, +0.3
                tmp_noise_1 = np.random.uniform(size=base_st_pos.shape, low=p_low, high=p_high)
                st_pos = base_st_pos + tmp_noise_1
                
                tmp_noise_2 = np.random.uniform(size=base_gl_pos.shape, low=p_low, high=p_high)
                gl_pos = base_gl_pos + tmp_noise_2
                
                # pdb.set_trace()
                
                st_state = np.concatenate([st_pos, [0.,0.]])
                out_data['start_state'].append( st_state )
                out_data['goal_pos'].append( gl_pos )
    
    # pdb.set_trace()
    return out_data


def merge_prob_dicts(prob_dicts: list[dict]):
    '''
    prob_dicts: list of dict
    '''
    mg_prob_dict = { k: [] for k in prob_dicts[0].keys() }
    ## loop through the list of dict
    for p_d in prob_dicts:
        assert set(p_d.keys()) == {'start_state', 'goal_pos'}
        for k in p_d.keys():
            ## list of np2d
            mg_prob_dict[k].append(p_d[k])
    for kk in mg_prob_dict.keys():
        mg_prob_dict[kk] = np.concatenate(mg_prob_dict[kk], axis=0)
    # pdb.set_trace()

    return mg_prob_dict


    
    



