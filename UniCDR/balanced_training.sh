#!/bin/bash
# Balanced training: Good speed + Good performance

echo "Running BALANCED FisherTune training..."
echo "- 200 iterations/epoch (2x faster than original)"
echo "- 150 epochs total"
echo "- Total updates: 30,000 (vs 50,000 original)"
echo "- Expected time: ~20-25 hours (vs 50 hours)"
echo ""

python train_rec_fishertune.py \
    --domains sport_cloth \
    --task dual-user-intra \
    --fishertune_mode adaptive \
    --num_epoch 150 \
    --iterations_per_epoch 200 \
    --batch_size 1536 \
    --ft_warmup_epochs 10 \
    --ft_update_freq 5 \
    --save \
    --cuda \
    --experiment_name "balanced_training"

echo ""
echo "Balanced training complete!"
