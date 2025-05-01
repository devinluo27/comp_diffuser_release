#!/bin/bash

#SBATCH --job-name=script-ohi
#SBATCH --output=trash/slurm/train_OG_StglSml_jan3/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --exclude="clippy,voltron,alexa"
#SBATCH --mem=100G

##SBATCH --gres=gpu:rtx_6000:1
##SBATCH --gres=gpu:a40:1
#SBATCH --gres=gpu:l40s:1
##
##SBATCH --qos="long"
##SBATCH --qos="debug"
#SBATCH --qos="short"
##SBATCH --time=100:00:00
#SBATCH --time=48:00:00

source ~/.bashrc
source activate compdfu_ogb_release

## optionally 'cd' to 'comp_diffuser_release' folder
# cd $Your_Folder_of_This_Repo

config="config/og_antM_Gi_o2d_luotest.py"
# config="config/ogb_ant_maze/og_antM_Lg_o2d_Cd_Stgl_PadBuf_Ft64_ts512_resume.py"
# config="config/ogb_ant_maze/og_antM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512_resume.py"

## --------- Training: OGBench AntMaze Stitch, High Dimension Planner ----------
## Large 15D
config="config/ogb_ant_maze/og_antM_Lg_o15d_PadBuf_Ft64_ts512_DiTd768dp16_fs4_h160_ovlp56Mdit.py"
## Large 29D
# config="config/ogb_ant_maze/og_antM_Lg_o29d_DiTd1024dp12_PadBuf_Ft64_fs4_h160_ovlp56MditD512.py"


## --------- Training: OGBench AntMaze Stitch 2D Planner ----------
## Giant
# config="config/ogb_ant_maze/og_antM_Gi_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
## Large
# config="config/ogb_ant_maze/og_antM_Lg_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
## Medium
# config="config/ogb_ant_maze/og_antM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"

## --------- Training: OGBench PointMaze Stitch 2D Planner ----------
## Giant
# config="config/ogb_pnt_maze/og_pntM_Gi_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
## Large
# config="config/ogb_pnt_maze/og_pntM_Lg_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
## Medium
# config="config/ogb_pnt_maze/og_pntM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"


{

echo $(hostname)

# CUDA_LAUNCH_BLOCKING=1 \
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py  --config $config \


exit 0

}