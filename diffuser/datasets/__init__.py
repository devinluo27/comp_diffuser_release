from .sequence import *
from .d4rl import load_environment
from .d4rl import load_env_gym_robo

import numpy as np
MAZE_Large_Obs_Min = np.array([0.39643136, 0.44179875], dtype=np.float32)
MAZE_Large_Obs_Max = np.array([ 7.2163844, 10.219488 ], dtype=np.float32)

MAZE_Large_Act_Min = np.array([-1.,  -1.], dtype=np.float32)
MAZE_Large_Act_Max = np.array([ 1., 1.], dtype=np.float32)

## ======= Ben =======
Ben_maze_large_Obs_Min = np.array( [-4.9163504, -3.4131463], dtype=np.float32 )
Ben_maze_large_Obs_Max = np.array( [4.8761816, 3.3245058], dtype=np.float32 )
Ben_maze_large_Act_Min = np.array([-1.,  -1.], dtype=np.float32)
Ben_maze_large_Act_Max = np.array([ 1., 1.], dtype=np.float32)

## ------- Ben Medium ---------

Ben_maze_Medium_Obs_Min = np.array( [  -2.9093022346, -2.9120881557   ], dtype=np.float32 )
Ben_maze_Medium_Obs_Max = np.array( [  2.8583071232, 2.7509465218  ], dtype=np.float32 )


Ben_maze_UMaze_Obs_Min = np.array( [ -1.2499687672, -1.2705398798 ], dtype=np.float32 )
Ben_maze_UMaze_Obs_Max = np.array( [ 1.2770261765, 1.2761843204 ], dtype=np.float32 )


## ===================



## For Diffuser Baseline:
MAZE_Large_ObsVel_Min = np.array([0.39643136, 0.44179875, -5.2262554, -5.2262554], dtype=np.float32)
MAZE_Large_ObsVel_Max = np.array([ 7.2163844, 10.219488, 5.2262554,  5.2262554], dtype=np.float32)