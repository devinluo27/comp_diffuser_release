#!/bin/bash

#SBATCH --job-name=script-hi
#SBATCH --output=trash/slurm/train_StglSml/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --exclude="clippy,voltron"
###SBATCH --partition="rl2-lab"

#SBATCH --gres=gpu:l40s:1
##SBATCH --gres=gpu:a40:1
###SBATCH --gres=gpu:rtx_6000:1
##
##SBATCH --qos="long"
##SBATCH --time=72:00:00
##
#SBATCH --qos="debug"
#SBATCH --time=48:00:00

source ~/.bashrc
source activate hi_diffuser

## Oct 8
config="config/cp_maze_v1/cd_stgl/m2d_lg_Cd_Stgl_luotest.py"




{

# PYTHONBREAKPOINT=0 \ ## -B
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/models/cd_stgl_sml_dfu/train_stgl_sml.py  --config $config \


exit 0

}