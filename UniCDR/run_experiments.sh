#!/bin/bash

# FisherTune Experiment Runner for UniCDR
# Run various configurations to find optimal settings

# Base directory
BASE_DIR=$(dirname "$0")
cd "$BASE_DIR"

# Create directories
mkdir -p logs
mkdir -p saved_models
mkdir -p results

# Common settings
COMMON_ARGS="--cuda --seed 42 --num_epoch 100 --batch_size 1024"
DOMAIN="sport_cloth"
TASK="dual-user-intra"

echo "=========================================="
echo "FisherTune Experiments for UniCDR"
echo "=========================================="

# Experiment 1: Baseline (no FisherTune)
run_baseline() {
    echo "Running Baseline..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --lr 0.001 \
        --exp_name baseline_${DOMAIN} \
        2>&1 | tee logs/baseline_${DOMAIN}.log
}

# Experiment 2: FisherTune with default settings
run_fishertune_default() {
    echo "Running FisherTune Default..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --shared_lr 0.0001 \
        --specific_lr 0.001 \
        --shared_delta_min 0.5 \
        --shared_delta_max 0.95 \
        --specific_delta_min 0.1 \
        --specific_delta_max 0.7 \
        --exp_name fishertune_default_${DOMAIN} \
        2>&1 | tee logs/fishertune_default_${DOMAIN}.log
}

# Experiment 3: FisherTune with conservative shared params (high delta)
run_fishertune_conservative_shared() {
    echo "Running FisherTune Conservative Shared..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --shared_lr 0.00005 \
        --specific_lr 0.001 \
        --shared_delta_min 0.7 \
        --shared_delta_max 0.99 \
        --specific_delta_min 0.1 \
        --specific_delta_max 0.6 \
        --exp_name fishertune_conservative_shared_${DOMAIN} \
        2>&1 | tee logs/fishertune_conservative_shared_${DOMAIN}.log
}

# Experiment 4: FisherTune with aggressive specific params (low delta)
run_fishertune_aggressive_specific() {
    echo "Running FisherTune Aggressive Specific..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --shared_lr 0.0001 \
        --specific_lr 0.002 \
        --shared_delta_min 0.5 \
        --shared_delta_max 0.95 \
        --specific_delta_min 0.05 \
        --specific_delta_max 0.5 \
        --exp_name fishertune_aggressive_specific_${DOMAIN} \
        2>&1 | tee logs/fishertune_aggressive_specific_${DOMAIN}.log
}

# Experiment 5: FisherTune with Edge Dropout perturbation
run_fishertune_edge_dropout() {
    echo "Running FisherTune Edge Dropout..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --perturbation_type edge_dropout \
        --edge_dropout_rate 0.3 \
        --exp_name fishertune_edge_dropout_${DOMAIN} \
        2>&1 | tee logs/fishertune_edge_dropout_${DOMAIN}.log
}

# Experiment 6: FisherTune with Popularity-based perturbation
run_fishertune_popularity() {
    echo "Running FisherTune Popularity..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --perturbation_type popularity \
        --popularity_alpha 0.7 \
        --exp_name fishertune_popularity_${DOMAIN} \
        2>&1 | tee logs/fishertune_popularity_${DOMAIN}.log
}

# Experiment 7: FisherTune with Natural Gradient
run_fishertune_natural_grad() {
    echo "Running FisherTune Natural Gradient..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --use_natural_gradient \
        --fisher_damping 1e-3 \
        --exp_name fishertune_natural_grad_${DOMAIN} \
        2>&1 | tee logs/fishertune_natural_grad_${DOMAIN}.log
}

# Experiment 8: FisherTune with Fisher Regularization
run_fishertune_fisher_reg() {
    echo "Running FisherTune Fisher Regularization..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --use_fisher_regularization \
        --fisher_reg_strength 0.05 \
        --exp_name fishertune_fisher_reg_${DOMAIN} \
        2>&1 | tee logs/fishertune_fisher_reg_${DOMAIN}.log
}

# Experiment 9: FisherTune with Adaptive VI
run_fishertune_adaptive_vi() {
    echo "Running FisherTune Adaptive VI..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --use_adaptive_vi \
        --variational_tau 0.5 \
        --variational_gamma 0.2 \
        --tau_decay 0.995 \
        --gamma_growth 1.005 \
        --exp_name fishertune_adaptive_vi_${DOMAIN} \
        2>&1 | tee logs/fishertune_adaptive_vi_${DOMAIN}.log
}

# Experiment 10: FisherTune with Adaptive Scheduler
run_fishertune_adaptive_scheduler() {
    echo "Running FisherTune Adaptive Scheduler..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --use_adaptive_scheduler \
        --scheduler_adaptation_rate 0.05 \
        --exp_name fishertune_adaptive_scheduler_${DOMAIN} \
        2>&1 | tee logs/fishertune_adaptive_scheduler_${DOMAIN}.log
}

# Experiment 11: FisherTune with longer warmup
run_fishertune_long_warmup() {
    echo "Running FisherTune Long Warmup..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 20 \
        --exp_name fishertune_long_warmup_${DOMAIN} \
        2>&1 | tee logs/fishertune_long_warmup_${DOMAIN}.log
}

# Experiment 12: FisherTune with frequent FIM updates
run_fishertune_freq_fim() {
    echo "Running FisherTune Frequent FIM Updates..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 10 \
        --fim_update_freq 5 \
        --exp_name fishertune_freq_fim_${DOMAIN} \
        2>&1 | tee logs/fishertune_freq_fim_${DOMAIN}.log
}

# Experiment 13: FisherTune All features combined
run_fishertune_all_features() {
    echo "Running FisherTune All Features..."
    python train_fishertune.py \
        --domains $DOMAIN \
        --task $TASK \
        $COMMON_ARGS \
        --use_fishertune \
        --fishertune_warmup 15 \
        --shared_lr 0.00005 \
        --specific_lr 0.0015 \
        --shared_delta_min 0.6 \
        --shared_delta_max 0.98 \
        --specific_delta_min 0.08 \
        --specific_delta_max 0.6 \
        --perturbation_type combined \
        --edge_dropout_rate 0.25 \
        --use_adaptive_vi \
        --variational_tau 0.7 \
        --variational_gamma 0.15 \
        --use_fisher_regularization \
        --fisher_reg_strength 0.02 \
        --exp_name fishertune_all_features_${DOMAIN} \
        2>&1 | tee logs/fishertune_all_features_${DOMAIN}.log
}

# Parse command line arguments
case "$1" in
    "baseline")
        run_baseline
        ;;
    "default")
        run_fishertune_default
        ;;
    "conservative")
        run_fishertune_conservative_shared
        ;;
    "aggressive")
        run_fishertune_aggressive_specific
        ;;
    "edge_dropout")
        run_fishertune_edge_dropout
        ;;
    "popularity")
        run_fishertune_popularity
        ;;
    "natural_grad")
        run_fishertune_natural_grad
        ;;
    "fisher_reg")
        run_fishertune_fisher_reg
        ;;
    "adaptive_vi")
        run_fishertune_adaptive_vi
        ;;
    "adaptive_scheduler")
        run_fishertune_adaptive_scheduler
        ;;
    "long_warmup")
        run_fishertune_long_warmup
        ;;
    "freq_fim")
        run_fishertune_freq_fim
        ;;
    "all_features")
        run_fishertune_all_features
        ;;
    "all")
        echo "Running all experiments sequentially..."
        run_baseline
        run_fishertune_default
        run_fishertune_conservative_shared
        run_fishertune_aggressive_specific
        run_fishertune_edge_dropout
        run_fishertune_popularity
        run_fishertune_natural_grad
        run_fishertune_fisher_reg
        run_fishertune_adaptive_vi
        run_fishertune_adaptive_scheduler
        run_fishertune_long_warmup
        run_fishertune_freq_fim
        run_fishertune_all_features
        ;;
    "quick")
        echo "Running quick experiments (baseline + default + all_features)..."
        run_baseline
        run_fishertune_default
        run_fishertune_all_features
        ;;
    *)
        echo "Usage: $0 {baseline|default|conservative|aggressive|edge_dropout|popularity|natural_grad|fisher_reg|adaptive_vi|adaptive_scheduler|long_warmup|freq_fim|all_features|all|quick}"
        echo ""
        echo "Options:"
        echo "  baseline           - Run baseline without FisherTune"
        echo "  default            - Run FisherTune with default settings"
        echo "  conservative       - Conservative shared params (highly frozen)"
        echo "  aggressive         - Aggressive specific params (wide tuning)"
        echo "  edge_dropout       - Edge dropout perturbation"
        echo "  popularity         - Popularity-based perturbation"
        echo "  natural_grad       - Natural gradient updates"
        echo "  fisher_reg         - Fisher-based regularization"
        echo "  adaptive_vi        - Adaptive variational inference"
        echo "  adaptive_scheduler - Adaptive threshold scheduler"
        echo "  long_warmup        - Longer warmup period"
        echo "  freq_fim           - More frequent FIM updates"
        echo "  all_features       - All features combined"
        echo "  all                - Run all experiments sequentially"
        echo "  quick              - Run baseline + default + all_features"
        exit 1
        ;;
esac

echo "=========================================="
echo "Experiment completed!"
echo "Results saved in logs/ directory"
echo "=========================================="
