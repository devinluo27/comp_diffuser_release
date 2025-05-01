#!/bin/bash

source ~/.bashrc
source activate hi_diffuser
# module load patchelf glew ## for compling the mujoco

pip install pip==21 && \
pip install "cython<3" ## ori: <=3 # <3 ?
pip install setuptools==65.5.0 

## 0.40.0
pip install wheel==0.38.0 && \
pip install gym==0.19.0 && \
pip install d4rl==1.1 && \
pip install "shimmy>=0.2.1" && \
pip install stable-baselines3 && \

## For Diffuser Legacy
pip install scikit-image==0.17.2 scikit-video==1.1.11