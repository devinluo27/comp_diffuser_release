#!/bin/bash

#SBATCH --job-name=script-ohi
#SBATCH --output=trash/slurm/train_OG_DD_jan3/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --exclude="clippy,voltron,alexa"
##SBATCH --partition="rl2-lab"
#SBATCH --mem=100G

##SBATCH --gres=gpu:rtx_6000:1
#SBATCH --gres=gpu:a40:1
##SBATCH --gres=gpu:l40s:1
##SBATCH --gres=gpu:a5000:1
##
##SBATCH --qos="long"
##SBATCH --time=100:00:00
##SBATCH --time=160:00:00
##
#SBATCH --qos="debug"
#SBATCH --time=48:00:00

source ~/.bashrc
source activate compdfu_ogb_release

## OGBench AntMaze, to be supported
# config="config/baselines/dd_ogb/dd_ogb_antM_Gi_o2d_PadBuf_Ft64_h160_ts512.py"
config="config/baselines/dd_ogb/dd_ogb_antM_Gi_Navi_o2d_noPad_h880_ts512.py"

## OGBench PointMaze
## giant maze
# config="config/baselines/dd_ogb/dd_ogb_pntM_Gi_o2d_PadBuf_Ft64_h160_ts512.py"
## large maze
# config="config/baselines/dd_ogb/dd_ogb_pntM_Lg_o2d_PadBuf_Ft64_h160_ts512.py"
## medium maze
# config="config/baselines/dd_ogb/dd_ogb_pntM_Me_o2d_PadBuf_Ft64_h160_ts512.py"

{

echo $(hostname)

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/baselines/dd_ogb/train_dd_ogb.py --config $config \


exit 0

}