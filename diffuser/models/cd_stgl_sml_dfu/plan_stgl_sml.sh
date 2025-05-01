#!/bin/bash

#SBATCH --job-name=script-ev-hi
#SBATCH --output=trash/slurm/plan_StglSml/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude="clippy,voltron, claptrap"
###SBATCH --partition="rl2-lab"

##SBATCH --gres=gpu:a40:1
#SBATCH --gres=gpu:rtx_6000:1
##
#SBATCH --qos="debug"
#SBATCH --time=5:00:00
##
###SBATCH --qos="debug"
###SBATCH --time=48:00:00

source ~/.bashrc
source activate hi_diffuser

echo $(hostname)
config="config/cp_maze_v1/cd_stgl_ben_Jan14/m2d_Me_BenPadex68FL_Cd_Stgl_h144_o48_ts512_bs128f_drop015.py"

{

# PYTHONBREAKPOINT=0 \ ## -B
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/models/cd_stgl_sml_dfu/plan_stgl_sml.py \
    --config $config \
    --plan_n_ep $2 \
    --pl_seeds ${3:--1} \


exit 0

}