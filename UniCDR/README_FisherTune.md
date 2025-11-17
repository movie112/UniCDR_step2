# FisherTune for UniCDR

This repository extends UniCDR with **FisherTune**, a domain-aware parameter selection method for improved cross-domain recommendation performance.

## Overview

FisherTune applies Fisher Information Matrix (FIM) analysis to identify and selectively tune parameters that are most important for:
1. **Task performance** - Parameters sensitive to recommendation accuracy
2. **Domain generalization** - Parameters sensitive to domain shift

By focusing computational resources on the most important parameters, FisherTune achieves better performance with controlled training overhead.

## Key Features

- **Multiple FisherTune Variants**: FIM-only, DR-FIM, adaptive scheduling
- **Parameter Selection Strategies**: Unified, shared-only, specific-only, adaptive
- **Domain Perturbation Methods**: Edge dropout, popularity weighting, noise injection, cross-domain
- **Efficient Implementation**: Online Fisher updates, gradient caching, block-wise computation
- **Comprehensive Logging**: Training curves, Fisher statistics, performance comparison

## Quick Start

### 1. Basic Training with FisherTune

```bash
# Baseline (no FisherTune)
python train_rec_fishertune.py \
    --domains sport_cloth \
    --task dual-user-intra \
    --fishertune_mode baseline \
    --cuda

# FIM with scheduling
python train_rec_fishertune.py \
    --domains sport_cloth \
    --task dual-user-intra \
    --fishertune_mode fim_only \
    --cuda

# DR-FIM (recommended)
python train_rec_fishertune.py \
    --domains sport_cloth \
    --task dual-user-intra \
    --fishertune_mode dr_fim \
    --cuda

# Adaptive mode (shared frozen, specific tuned)
python train_rec_fishertune.py \
    --domains sport_cloth \
    --task dual-user-intra \
    --fishertune_mode adaptive \
    --cuda
```

### 2. Run Multiple Experiments

```bash
# Run selected experiments
python run_fishertune_experiments.py \
    --domains sport_cloth \
    --experiments baseline fim_only dr_fim adaptive \
    --num_epoch 50 \
    --cuda

# Run all experiments
python run_fishertune_experiments.py \
    --domains sport_cloth \
    --all_experiments \
    --num_epoch 50 \
    --cuda
```

### 3. Analyze Results

```bash
python analyze_fishertune_results.py \
    --log_dir ./experiment_logs/run_YYYYMMDD_HHMMSS
```

## Experiment Configurations

### 1. FIM-Only with Scheduling
Standard Fisher Information Matrix with exponential scheduling. Good baseline for FisherTune.
```python
--fishertune_mode fim_only
```

### 2. DR-FIM (Domain-Related FIM)
Incorporates domain shift sensitivity. Best for improving generalization.
```python
--fishertune_mode dr_fim
```

### 3. Unified Parameter Selection
Treats all parameters equally, no shared/specific distinction.
```python
--fishertune_mode unified
```

### 4. Shared-Only
Only tunes shared parameters (embeddings and shared aggregator).
```python
--fishertune_mode shared_only
```

### 5. Specific-Only
Only tunes domain-specific parameters.
```python
--fishertune_mode specific_only
```

### 6. Adaptive Mode
**Recommended for best performance**
- Shared parameters: High threshold (δ=0.7-0.95) → mostly frozen
- Specific parameters: Low threshold (δ=0.1-0.5) → more tuning
```python
--fishertune_mode adaptive
```

### 7. Perturbation Strategies

```bash
# Edge dropout - simulates sparse domain
--fishertune_mode perturbation_edge_dropout --ft_perturbation_rate 0.2

# Popularity weighting - simulates long-tail shift
--fishertune_mode perturbation_popularity --ft_perturbation_rate 0.3

# Noise injection - simulates noisy domain
--fishertune_mode perturbation_noise --ft_noise_scale 0.05

# Cross-domain - uses actual domain pairs (most realistic)
--fishertune_mode perturbation_cross_domain
```

## Configuration Parameters

### Core FisherTune Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ft_warmup_epochs` | 5 | Epochs before applying FisherTune |
| `--ft_update_freq` | 5 | Frequency of Fisher recomputation |
| `--ft_num_samples` | 100 | Samples for Fisher estimation |
| `--ft_delta_min` | 0.1 | Minimum selection threshold |
| `--ft_delta_max` | 0.9 | Maximum selection threshold |
| `--ft_schedule_T` | 10 | Exponential decay time constant |
| `--ft_perturbation_rate` | 0.1 | Perturbation strength |
| `--ft_noise_scale` | 0.01 | Noise scale for perturbation |

### Scheduling Formula

The threshold decreases exponentially:
```
δ(t) = δ_min + (δ_max - δ_min) × exp(-t/T)
```

- **t=0**: High threshold → few parameters updated (most sensitive)
- **t→∞**: Low threshold → more parameters updated

## UniCDR Parameter Structure

### Shared Parameters
- `share_user_embedding` or `share_item_embedding`
- Shared `BehaviorAggregator` (index: num_domains)
- Cross-domain information transfer

### Domain-Specific Parameters
- `specific_user_emb_list[i]` for each domain
- `specific_item_emb_list[i]` for each domain
- Per-domain `BehaviorAggregator` (index: 0 to num_domains-1)
- Domain-specific discriminators

## Expected Performance

Based on FisherTune methodology:

1. **Adaptive Mode**: Expected ~3-8% improvement in NDCG@10
   - Shared parameters capture universal patterns
   - Specific parameters adapt to domain characteristics

2. **DR-FIM with Cross-Domain Perturbation**: Expected ~5-10% improvement
   - Identifies domain-invariant features
   - Reduces overfitting to source domain

3. **Training Overhead**: < 15% increase
   - Online Fisher updates: ~5%
   - Gradient masking: ~1%
   - DR-FIM computation: ~5-10%

## Output Files

After running experiments:

```
experiment_logs/
├── run_YYYYMMDD_HHMMSS/
│   ├── baseline_results.json         # Individual experiment results
│   ├── fim_only_results.json
│   ├── dr_fim_results.json
│   ├── adaptive_results.json
│   ├── experiment_comparison.csv     # Comparison table
│   ├── experiment_report.txt         # Detailed report
│   ├── all_results.json              # Combined results
│   ├── training_curves.png           # Visualization
│   ├── validation_metrics.png
│   ├── fishertune_analysis.png
│   ├── results_table.tex             # LaTeX table
│   └── results_table.csv
```

## Best Practices

1. **Start with Baseline**: Always run baseline for comparison
2. **Use Adaptive Mode**: Best balance of shared preservation and specific adaptation
3. **Cross-Domain Perturbation**: Most realistic for CDR scenarios
4. **Monitor Fisher Statistics**: Check `avg_shared_fim` vs `avg_specific_fim`
5. **Adjust Warmup**: Start FisherTune after model has initial learning (5-10 epochs)
6. **Balance Efficiency**: Reduce `ft_num_samples` for faster training

## Technical Details

### Fisher Information Matrix

For parameter θ:
```
FIM(θ) = E[(∂L/∂θ)²]
```

Diagonal approximation for efficiency: O(|θ|) instead of O(|θ|²)

### Domain-Related FIM

```
DR-FIM = FIM(x,y) + exp(-ε) × |FIM(x,y) - FIM(x',y)| / min(FIM(x), FIM(x'))
```

Where x' is the perturbed sample simulating domain shift.

### Variational Stabilization

Optional regularization using prior:
```
F_stable = γ(Λ - τ⁻²I)
```

Where Λ is the precision from variational inference.

## Citation

If you use FisherTune with UniCDR, please cite:

```bibtex
@inproceedings{unicdr,
  title={UniCDR: A Unified Cross-Domain Recommendation Framework},
  author={...},
  booktitle={...},
  year={...}
}

@article{fishertune,
  title={FisherTune: Domain-Aware Parameter Selection},
  author={...},
  journal={...},
  year={...}
}
```

## Troubleshooting

1. **Out of Memory**: Reduce `--ft_num_samples` or `--batch_size`
2. **Slow Training**: Increase `--ft_update_freq` to compute Fisher less often
3. **Poor Performance**: Try different perturbation strategies
4. **Unstable Training**: Enable variational stabilization in config

## Future Extensions

- [ ] Multi-task Fisher aggregation
- [ ] Automatic threshold tuning
- [ ] Layer-wise importance analysis
- [ ] Integration with other CDR methods
