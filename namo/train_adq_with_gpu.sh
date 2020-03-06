#!/bin/bash
echo "Using GPU: $3"
echo "python train_scripts/train_algo.py -n_data $1 -n_trial $2 -pi adq_new_reward -Qloss adv -n_score 1 -tau 2 -explr_const 0.5 -d_lr 1e-3 -g_lr 1e-4"
export CUDA_VISIBLE_DEVICES=$3
python train_scripts/train_algo.py -n_data $1 -n_trial $2 -pi adq_new_reward -Qloss adv -n_score 1 -tau 2 -explr_const 0.5 -d_lr 1e-3 -g_lr 1e-4
