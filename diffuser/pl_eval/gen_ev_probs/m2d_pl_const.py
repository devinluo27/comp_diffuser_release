import numpy as np

## ours: use the Diffuser default coordinate system
## LH: Long horizon
Maze2D_Large_Bottom_Row_ours_LH = \
    np.array([ [1,1],[2,1],[3,1], [4,1], [5,1], [7,1] ], dtype=np.float32)

Maze2D_Large_Top_Row_ours_LH = \
    np.array([ [1,10],[2,10],[3,10], [4,10], [5,10], [7,10] ], dtype=np.float32)


def m2d_get_bottom_top_rows(env_name):
    '''return two rows'''
    if env_name == 'maze2d-large-v1':
        return Maze2D_Large_Bottom_Row_ours_LH, Maze2D_Large_Top_Row_ours_LH
    elif env_name == 'maze2d-medium-v1':
        assert False
    elif env_name == 'maze2d-umaze-v1':
        assert False
    else:
        assert False