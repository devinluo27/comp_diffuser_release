import numpy as np

def ogb_get_Maze_stgl_from_tk_idx(env_name, tk_idx):
    """
    Only for Maze Tasks in OGBench
    """
    if '-medium' in env_name:
        if tk_idx == 1:
            st_gl = np.array([[1, 1.], [6, 6.]])
        elif tk_idx == 2:
            st_gl = np.array([[ 6., 1 ], [ 1, 6 ]])
            
        elif tk_idx == 3:
            st_gl = np.array([[ 5, 3. ], [ 4, 2 ]])
            
        elif tk_idx == 4:
            st_gl = np.array([[ 6, 5. ], [  6, 1 ]])
            
        elif tk_idx == 5:
            st_gl = np.array([[ 2., 6 ], [ 1, 1. ]])
            
    
    elif '-large' in env_name:
        if tk_idx == 1:
            st_gl = np.array([[1, 1.], [7,10.]]) ## Large Tk 1
        elif tk_idx == 2:
            st_gl = np.array([ [5, 4.], [7, 1.]]) ## Large TK 2
        elif tk_idx == 3:
            st_gl = np.array([ [7, 4.], [1, 10.]]) ## Seed 1
        elif tk_idx == 4:
            st_gl = np.array([ [3, 8.], [5, 4.]])
        elif tk_idx == 5:
            st_gl = np.array([ [1, 1.], [5, 4.]]) ## Seed 0, tk 5
    
    elif '-giant' in env_name:
        if tk_idx == 1:
            st_gl = np.array([[1, 1.], [10,14.]])
        elif tk_idx == 2:
            st_gl = np.array([[1, 14.], [10,1.]])
        elif tk_idx == 3:
            st_gl = np.array([[8, 14.], [1.,1.]])
        elif tk_idx == 4:
            st_gl = np.array([[8, 3.], [5,12]])
        elif tk_idx == 5:
            st_gl = np.array([[5, 9.], [3, 8]])
    else:
        raise NotImplementedError

    return st_gl


def ogb_get_antSoccer_stgl_from_tk_idx(env_name, tk_idx):
    """
    Only for Maze Tasks in OGBench
    """
    if '-medium' in env_name:
        if tk_idx == 1:
            st_gl = np.array([[1, 1., 3., 4.], [6, 6., 6., 6.,]])
        elif tk_idx == 2:
            st_gl = np.array([[ 6., 1, 6, 5], [ 1, 1., 1., 1.  ]])
            
        elif tk_idx == 3:
            st_gl = np.array([[ 5, 3., 4, 2. ], [ 6, 5, 6, 5. ]])
            
        elif tk_idx == 4:
            st_gl = np.array([[ 6, 5., 1., 1. ], [ 5, 3., 5., 3 ]])
            
        elif tk_idx == 5:
            st_gl = np.array([[ 1., 6, 6, 1. ], [ 1, 6., 1., 6. ]])

    elif '-arena' in env_name:
        if tk_idx == 1:
            st_gl = np.array([[1, 6., 2., 3.], [5, 2., 5., 2.,]])
            
        elif tk_idx == 2:
            st_gl = np.array([[ 1, 1., 5, 5. ], [ 1., 1, 1, 1 ]])
            
        elif tk_idx == 3:
            st_gl = np.array([[ 6, 1., 2, 3. ], [ 6, 6, 6, 6. ]])
            
        elif tk_idx == 4:
            st_gl = np.array([[ 6, 6., 1., 1. ], [ 6., 1., 6, 1. ]])
            
        elif tk_idx == 5:
            st_gl = np.array([[ 4., 6, 6, 2. ], [ 1, 6., 1., 6. ]])

    return st_gl

def ogb_ij_to_xy(ij_trajs):
    maze_unit = 4
    _offset_x, _offset_y = 4, 4
    assert ij_trajs.ndim in [2, 3] and ij_trajs.shape[-1] == 2
    x = ij_trajs[..., 1:2] * maze_unit - _offset_x
    y = ij_trajs[..., 0:1] * maze_unit - _offset_y
    
    out = np.concatenate([x, y], axis=-1)
    return out

def ogb_ij_to_xy_4d(ij_trajs):
    maze_unit = 4
    _offset_x, _offset_y = 4, 4
    assert ij_trajs.ndim in [2, 3] and ij_trajs.shape[-1] in [2, 4]
    x_1 = ij_trajs[..., 1:2] * maze_unit - _offset_x
    y_1 = ij_trajs[..., 0:1] * maze_unit - _offset_y

    x_2 = ij_trajs[..., 3:4] * maze_unit - _offset_x
    y_2 = ij_trajs[..., 2:3] * maze_unit - _offset_y
    
    out = np.concatenate([x_1, y_1, x_2, y_2], axis=-1)
    return out