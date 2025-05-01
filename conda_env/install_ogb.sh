#!/bin/bash

##
## Env Install: Step 4
##

source ~/.bashrc
# source activate hi_diffuser_ogbench_v3
source activate compdfu_ogb_release


## Install the customized OGBench env by CompDiffuser
## We do not change the env logics, but simply add some features for better visualization

## ** Please clone the 'ogbench_cpdfu_release' repo ***
## You can also replace '../ogbench_cpdfu_release' by the acutal path in your computer
pip install -e '../ogbench_cpdfu_release'

## this mujoco version might not be compatible to 'gymnasium-robotics', but since we probably won't use gymnasium-robotics, so it should be fine.
pip install mujoco==3.2.6
pip install setuptools==65.5.0