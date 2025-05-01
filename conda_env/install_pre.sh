#!/bin/bash

##
## Env Install: Step 1
##

source ~/.bashrc
# source activate hi_diffuser_ogbench_v3
source activate compdfu_ogb_release

## install gymnasium env, which will also install d4rl etc. 
## d4rl is probably not used in OGBench, but since our code is inherent from diffuser, some files might still import d4rl (though these files probably also can be removed)
pip install -e git+https://github.com/Farama-Foundation/Gymnasium.git@81b87efb9f011e975f3b646bab6b7871c522e15e#egg=gymnasium
pip install -e git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@3e42c3061aa23ce7f29ae8ec150ca6224e13ef09#egg=gymnasium_robotics

## install torch
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121