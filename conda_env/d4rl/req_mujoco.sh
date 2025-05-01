## install mujoco210 first, just simply install it to 
## ~/.mujoco/mujoco210, otherwise might have bugs!

## remember to put this to ~/.bashrc
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/nethome/yluo470/.mujoco/mujoco210/bin"
## actually it is a not existed path
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"


## install rendering dependencies
conda install -y -c conda-forge glew
conda install -y -c conda-forge mesalib
conda install -c menpo glfw3

## oneline version
conda install -y -c conda-forge glew && conda install -y -c conda-forge mesalib && conda install -c menpo glfw3

### Sep 24: use another channel to prevent re-install another version of python
# conda install -c pkgs/main glew


## Then add your conda environment include to CPATH (put this in your .bashrc to make it permanent):
export CPATH=$CONDA_PREFIX/include
## Finally, install patchelf with pip install patchelf

## for libmamba solver
## $ conda install --solver=classic conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive
