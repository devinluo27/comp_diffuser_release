import pdb

def load_eval_problems_pb(env, pd_config):
    '''
    loading the evaluation problems for potential based diffusion
    '''

    if pd_config.get('load_unseen_maze', False):
        if pd_config.get('no_check_bit', False):
            npi = str(pd_config['npi'])
            # load no-checked problems
            problems_h5path = env._dataset_url.replace('.hdf5', f'-problems-nochk_{npi}pi.hdf5')
            assert False, "'bugfree, but don't do it now"
        else:
            assert pd_config.get('npi', 0) == 0
            # load checked problems
            problems_h5path = env._dataset_url.replace('.hdf5', '-problems.hdf5')

        problems_dict = env.get_dataset(h5path=problems_h5path, no_check=True) # a dict
    else:
        assert False

    return problems_dict



### --------
def split_trajs_list_by_prob(trajs_list, n_probs, cond_type='gl'):
    '''
    We flatten multiple e.g., 10 problems inside one batch model forward,
    In this function, we unflatten the large batch, and return a list,
    where each element is one trajs_list of len n_comp corresponding to a problem.
    
    Args:
        each probs lies near: p1,p1,p1,p2,p2,...
        trajs_list: list of n_c : (B, h, d)
    '''
    out_list = [] ## a list of trajs_list
    n_comp = len(trajs_list)
    b_s_all, _, _ = trajs_list[0].shape ## b_s*n_p=200,h,d

    ## e.g., 20 = 200 / 10
    n_sam_per_prob = round( b_s_all / n_probs)
    assert n_sam_per_prob  * n_probs == b_s_all
    # pdb.set_trace() ## check len of trajs_list = n_comp = 4

    ## e.g., 0...9 --> [0:20], 20:40, ..., 180:200
    for i_p in range(n_probs):
        tmp_st = i_p * n_sam_per_prob
        tmp_end = (i_p+1) * n_sam_per_prob
        tmp_tj_l = [] ## tj_list of one prob
        for i_c in range(n_comp):
            ## append (n_sam_per_prob,h,d)
            tmp_tj_l.append( trajs_list[i_c][ tmp_st:tmp_end ] )
        
        ## must under the assumption of inpainting
        ## ensure st and gl of the first and last traj in the batch is the same
        assert (tmp_tj_l[0][0, 0, :] == tmp_tj_l[0][-1, 0, :]).all()
        assert cond_type in ['gl', 'ret']
        if cond_type == 'gl':
            assert (tmp_tj_l[-1][0, -1, :] == tmp_tj_l[-1][-1, -1, :]).all()

        ## Oct 24, check tmp_tj_l
        # pdb.set_trace()
        out_list.append( tmp_tj_l )
    
    # pdb.set_trace()
    return out_list


