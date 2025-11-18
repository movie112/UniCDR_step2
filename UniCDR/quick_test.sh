#!/bin/bash
# Quick test script for FisherTune with fast training

echo "Running FAST FisherTune test..."

python train_rec_fishertune.py \
    --domains sport_cloth \
    --task dual-user-intra \
    --fishertune_mode adaptive \
    --num_epoch 30 \
    --iterations_per_epoch 50 \
    --ft_warmup_epochs 3 \
    --ft_update_freq 10 \
    --ft_num_samples 50 \
    --batch_size 2048 \
    --cuda \
    --experiment_name "fast_test"

echo "Fast test complete!"
