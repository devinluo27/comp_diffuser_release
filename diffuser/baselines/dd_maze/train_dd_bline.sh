#!/bin/bash

#SBATCH --job-name=script-hi-dd-bl
#SBATCH --output=trash/slurm/train_dd_maze_bline/slurm-%j.out
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
cd /coc/flash7/yluo470/robot2024/hi_diffuser/

## Jan 17
config="config/baselines/ben/dd_bline_m2d_Umz_BenPadex40FL_luotest.py"
# config="config/baselines/ben/dd_bline_m2d_Umz_BenPadex40FL.py"

# config="config/baselines/ben/dd_bline_m2d_Me_BenPadex68FL_h144.py"
# config="config/baselines/ben/dd_bline_m2d_Lg_BenPadex60FL_h192.py"

{

# PYTHONBREAKPOINT=0 \ ## -B
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/baselines/dd_maze/train_dd_bline.py --config $config \


exit 0

}