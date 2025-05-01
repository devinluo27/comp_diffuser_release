import numpy as np
import matplotlib.pyplot as plt
from typing import List
import imageio, os


def pad_traj2d(traj: np.ndarray, req_len, pad_type='last'):
    ''' (t, 2) -> (t+res, 2), last pad '''
    residual = req_len - traj.shape[0]
    assert residual >= 0
    if residual > 0:
        pad = traj[-1:, :].repeat(residual, axis=0) # (1, 2) -> (res, 2)
        traj = np.append( traj, pad, axis=0 ) # (t+res, 2)
    return traj

def pad_traj2d_list(env_solutions: List[np.ndarray]):
    '''given a list of traj with different horizon, pad them to the max horizon'''
    # list of np2d
    assert env_solutions[0].ndim == 2
    max_len = max([ len(s) for s in env_solutions ])
    tmp = [] # new list of np
    for i in range(len(env_solutions)):
        tmp.append( pad_traj2d(env_solutions[i], max_len) )
    return tmp

def pad_traj2d_list_v2(env_solutions: List[np.ndarray], req_len):
    '''pad them to the given horizon 
    shape [ (t, 2), ] (given a list of traj with different horizon)'''
    # list of np2d
    assert env_solutions[0].ndim == 2
    tmp = [] # new list of np
    for i in range(len(env_solutions)):
        tmp.append( pad_traj2d(env_solutions[i], req_len) )
    return tmp

def pad_traj2d_list_v3(env_solutions: List[np.ndarray], target:List[np.ndarray]):
    '''pad them to the given horizon 
    shape [ (t, 2), ] (given a list of traj with different horizon)'''
    # list of np2d
    assert env_solutions[0].ndim == 2
    assert target[0].ndim == 2
    tmp = [] # new list of np
    for i in range(len(env_solutions)):
        tmp.append( pad_traj2d(env_solutions[i], len(target[i])) )
    return tmp