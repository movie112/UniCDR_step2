#!/bin/bash
# Two-stage training: Fast exploration + Fine-tuning

DOMAINS="sport_cloth"
TASK="dual-user-intra"
MODE="adaptive"

echo "========================================="
echo "STAGE 1: Fast Exploration (100 iter/epoch)"
echo "========================================="

python train_rec_fishertune.py \
    --domains $DOMAINS \
    --task $TASK \
    --fishertune_mode $MODE \
    --num_epoch 200 \
    --iterations_per_epoch 100 \
    --batch_size 2048 \
    --save \
    --experiment_name "${MODE}_stage1_fast" \
    --cuda

echo ""
echo "========================================="
echo "STAGE 2: Fine-tuning (500 iter/epoch)"
echo "========================================="

# Continue from best checkpoint with more iterations
python train_rec_fishertune.py \
    --domains $DOMAINS \
    --task $TASK \
    --fishertune_mode $MODE \
    --num_epoch 50 \
    --iterations_per_epoch 500 \
    --batch_size 1024 \
    --lr 0.0001 \
    --save \
    --experiment_name "${MODE}_stage2_finetune" \
    --cuda

echo ""
echo "Two-stage training complete!"
