#!/bin/bash

#SBATCH --job-name=script-ohi-inv
#SBATCH --output=trash/slurm/train_OG_inv/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude="clippy,voltron,dendrite,claptrap" ## glados,oppy
#SBATCH --qos="debug"
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a40:1
##SBATCH --partition="rl2-lab"
##SBATCH --gres=gpu:rtx_6000:1
##SBATCH --gres=gpu:a5000:1


source ~/.bashrc
source activate compdfu_ogb_release

## optionally 'cd' to 'comp_diffuser_release' folder
# cd $Folder

# echo $(hostname)


## ------------ AntMaze Explore ------------
## AntMaze Medium Explore
# config="config/maze_og/og_inv/og_antMexpl_Me_o29d_g2d_invdyn_h12.py"

## AntMaze Large Explore
# config="config/maze_og/og_inv_Jan1/og_antMexpl_Lg_o29d_g2d_invdyn_h12.py"

## -----------------------------------------


## humanoid
# config="config/maze_og/og_inv_Jan4/og_humM_Gi_o69d_g2d_invdyn_h80_dm5_dout02.py"
# config="config/maze_og/og_inv_Jan4/og_humM_Lg_o69d_g2d_invdyn_h80_dm5_dout02.py"
# config="config/maze_og/og_inv_Jan4/og_humM_Me_o69d_g2d_invdyn_h80_dm5_dout02.py"


## ------------ AntSoccer Arena ------------
# config="config/ogb_invdyn/og_inv_ant_soc/og_antSoc_Ar_o42d_g4d_invdyn_h120_dout02.py"
# config="config/ogb_invdyn/og_inv_ant_soc/og_antSoc_Ar_o42d_g17d_invdyn_h120_dout02.py"

## ------------ AntSoccer Medium ------------
# config="config/ogb_invdyn/og_inv_ant_soc/og_antSoc_Me_o42d_g4d_invdyn_h120_dm4_dout02.py"
# config="config/ogb_invdyn/og_inv_ant_soc/og_antSoc_Me_o42d_g17d_invdyn_h100_dm4_dout02.py"



## --------- Training Inverse Dynamics Models: OGBench AntMaze Stitch 2D Planner ----------

## ** Example: **
## the state space of ant is 29D (so observation is 29D)
## 2D (x-y) diffusion planner (so goal is 2D)
## to set which inverse dynamics model to use, check 'diffuser/utils/ogb_utils/ogb_serial.py'

## ant maze giant stitch
config="config/ogb_invdyn/og_inv_ant/og_antM_Gi_o29d_g2d_invdyn_h12.py" ## for 2D planner
# config="config/ogb_invdyn/og_inv_ant/og_antM_Gi_o29d_g15d_invdyn_h12.py"
# config="config/ogb_invdyn/og_inv_ant/og_antM_Gi_o29d_g29d_invdyn_h12_dm5.py"

## Large
## ant maze large stitch
# config="config/ogb_invdyn/og_inv_ant/og_antM_Lg_o29d_g2d_invdyn_h12.py"
# config="config/ogb_invdyn/og_inv_ant/og_antM_Lg_o29d_g15d_invdyn_h12.py"
# config="config/ogb_invdyn/og_inv_ant/og_antM_Lg_o29d_g29d_invdyn_h12.py"

## Medium
## ant maze medium stitch
# config="config/maze_og/og_inv_Jan1/og_antM_Me_o29d_g2d_invdyn_h12.py"
# config="config/maze_og/og_inv_Jan6/og_antM_Me_o29d_g15d_invdyn_h12.py"
# config="config/maze_og/og_inv_Jan1/og_antM_Me_o29d_g29d_invdyn_h12_dm5.py"



{

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/ogb_task/og_inv_dyn/train_og_invdyn.py --config $config \


exit 0

}



