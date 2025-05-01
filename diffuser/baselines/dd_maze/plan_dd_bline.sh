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
cd /coc/flash7/yluo470/robot2024/hi_diffuser/
echo $(hostname)

#### Jan 18
#### Ben
conda activate hi_diffuser_ben
config=""
config="config/baselines/ben/dd_bline_m2d_Umz_BenPadex40FL.py" ## hzn=136

# config="config/baselines/ben/dd_bline_m2d_Me_BenPadex68FL_h144.py"
# config="config/baselines/ben/dd_bline_m2d_Lg_BenPadex60FL_h192.py"


{

# PYTHONBREAKPOINT=0 \ ## -B
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/baselines/dd_maze/plan_dd_bline.py \
    --config $config \
    --plan_n_ep $2 \
    --pl_seeds ${3:--1} \


exit 0

}