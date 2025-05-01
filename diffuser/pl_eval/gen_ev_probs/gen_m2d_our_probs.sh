#!/bin/bash

source ~/.bashrc
source activate hi_diffuser

sub_conf='maze2d_lg_ev_prob_luotest'
sub_conf='maze2d_lg_ev_prob_luotest_2'
sub_conf='maze2d_lg_ev_prob_bt2way_nppp3'
sub_conf='maze2d_lg_ev_prob_bt2way_nppp3_rprange02'



{

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/pl_eval/gen_ev_probs/gen_m2d_our_probs.py \
    --sub_conf ${sub_conf}


exit 0

}