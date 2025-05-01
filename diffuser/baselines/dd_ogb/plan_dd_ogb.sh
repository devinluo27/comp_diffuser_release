#!/bin/bash


#SBATCH --job-name=script-ev-hi
#SBATCH --output=trash/slurm/plan_OG_DD_Feb15/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
### brainiac, 2080: alexa,alexa
#SBATCH --exclude="clippy,voltron,claptrap,alexa,bmo,olivaw,oppy"

##SBATCH --partition="rl2-lab"
##SBATCH --gres=gpu:a40:1
#SBATCH --gres=gpu:l40s:1
##SBATCH --qos="short"

##SBATCH --gres=gpu:rtx_6000:1
##SBATCH --gres=gpu:a5000:1

#SBATCH --qos="debug"
#SBATCH --time=16:00:00

echo $(hostname)

source ~/.bashrc
source activate compdfu_ogb_release

## ============================================
## Evaluation Script for Decision Diffuser (DD)
## ============================================


## --------- Planning: OGBench PointMaze Stitch 2D Planner ----------
## Giant
# config="config/baselines/dd_ogb/dd_ogb_pntM_Gi_o2d_PadBuf_Ft64_h160_ts512.py"
## Large
# config="config/baselines/dd_ogb/dd_ogb_pntM_Lg_o2d_PadBuf_Ft64_h160_ts512.py"
## Medium
# config="config/baselines/dd_ogb/dd_ogb_pntM_Me_o2d_PadBuf_Ft64_h160_ts512.py"

## --------- Planning: OGBench AntMaze Navigation 2D Planner ----------
# config="config/baselines/dd_ogb/dd_ogb_antM_Gi_Navi_o2d_noPad_h880_ts512.py"

## --------- Planning: OGBench AntMaze Stitch 2D Planner ----------
config="config/baselines/dd_ogb/dd_ogb_antM_Gi_o2d_PadBuf_Ft64_h160_ts512.py"



{

EGL_ID=0

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
MUJOCO_EGL_DEVICE_ID=$EGL_ID \
python diffuser/baselines/dd_ogb/plan_dd_ogb.py \
    --config $config \
    --plan_n_ep $2 \
    --pl_seeds ${3:--1} \


exit 0

}