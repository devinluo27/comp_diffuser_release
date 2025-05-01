import sys, pdb
sys.path.insert(0, './')
import gym, gc, os, torch, random, yaml, h5py
import numpy as np
from importlib import reload
from diffuser.utils import utils
from tqdm import tqdm
import warnings
warnings.simplefilter('always', ResourceWarning)  # Show all resource warnings
from tap import Tap
from diffuser.pl_eval.gen_ev_probs.m2d_pl_const import m2d_get_bottom_top_rows
from diffuser.pl_eval.gen_ev_probs.gen_m2d_probs_utils import m2d_rand_sample_probs, merge_prob_dicts

class Parser(Tap):
    sub_conf: str = 'config.maze2d'

def main():
    '''a helper script to generate dataset of random samples in the Libero env'''

    args = Parser().parse_args()

    file_path = 'diffuser/pl_eval/gen_ev_probs/gen_m2d_our_probs_confs.yaml'
    # Open the YAML file and load its contents
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        rs_cfg = config[args.sub_conf]

    seed_u = rs_cfg['seed_u']
    random.seed(seed_u)
    np.random.seed(seed_u)
    torch.manual_seed(seed_u)

    # pdb.set_trace()
    ## TODO: Oct 21, 18:53, From Here

    el_name = rs_cfg['el_name']
    assert el_name == 'maze2d-large-v1', 'not implement others yet'

    ## do we actually need to init an env?
    # env = gym.make(rs_cfg['el_name'],) #  gen_data=True)
    utils.print_color(f'el_name: {el_name}')
    prob_dicts = []
    if rs_cfg['prob_type'] == 'bottom_top_2way':
        ## np2d (n_valid_cell, 2)
        bottom_row, top_row = m2d_get_bottom_top_rows(el_name)
        ## 1. from bottom to top
        st_p_l_1 = bottom_row
        gl_p_l_1 = top_row
        ### from bottom to top and from top to bottom
        tmp_probs = m2d_rand_sample_probs(st_p_l_1, gl_p_l_1, rs_cfg)
        prob_dicts.append(tmp_probs)

        # pdb.set_trace()

        ## 2. from top to bottom
        st_p_l_2 = top_row
        gl_p_l_2 = bottom_row
        tmp_probs = m2d_rand_sample_probs(st_p_l_2, gl_p_l_2, rs_cfg)
        prob_dicts.append(tmp_probs)

        all_prob_dict = merge_prob_dicts(prob_dicts)

        # pdb.set_trace()

    else:
        raise NotImplementedError

    
    ## check consistency ??

    






    h5_root = '/coc/flash7/yluo470/robot2024/hi_diffuser/data/m2d/ev_probs'
    h5_save_path = f'{h5_root}/{args.sub_conf}.hdf5'
    # pdb.set_trace()
    num_probs = all_prob_dict['start_state'].shape[0]

    ## ----------------------------------
    ## Finished all, save to hdf5
    with h5py.File(h5_save_path, 'w') as file:
        file.create_dataset('start_state', data=all_prob_dict['start_state'])
        file.create_dataset('goal_pos', data=all_prob_dict['goal_pos'])

        file.attrs['env_seed'] = seed_u
        file.attrs['env_name'] = el_name

    ## lock file
    if 'luotest' not in args.sub_conf:
        os.chmod(h5_save_path, 0o444)
    utils.print_color(f'[save to] {h5_save_path=}')
    

if __name__ == '__main__':
    main()

