#!/usr/bin/env python
"""
Script to run multiple FisherTune experiments and compare results

This script automates running different FisherTune configurations
and generates comparison reports.
"""

import subprocess
import json
import os
import time
import argparse
from pathlib import Path
import pandas as pd


def get_all_experiment_configs():
    """
    Define all experiment configurations to run

    Each configuration represents a different FisherTune variant:
    1. baseline: Standard UniCDR without FisherTune
    2. fim_only: FIM with scheduling
    3. dr_fim: Domain-Related FIM
    4. unified: All parameters, no shared/specific distinction
    5. shared_only: Only shared parameters
    6. specific_only: Only domain-specific parameters
    7. adaptive: Shared frozen, specific tuned
    8. perturbation variants: Different perturbation strategies
    """

    experiments = {
        'baseline': {
            'description': 'Standard UniCDR without FisherTune',
            'args': ['--fishertune_mode', 'baseline']
        },
        'fim_only': {
            'description': 'FIM with scheduling (no DR-FIM)',
            'args': ['--fishertune_mode', 'fim_only']
        },
        'dr_fim': {
            'description': 'Domain-Related FIM',
            'args': ['--fishertune_mode', 'dr_fim']
        },
        'unified': {
            'description': 'Unified parameter selection (ignore shared/specific)',
            'args': ['--fishertune_mode', 'unified']
        },
        'shared_only': {
            'description': 'Only tune shared parameters',
            'args': ['--fishertune_mode', 'shared_only']
        },
        'specific_only': {
            'description': 'Only tune domain-specific parameters',
            'args': ['--fishertune_mode', 'specific_only']
        },
        'adaptive': {
            'description': 'Shared frozen (high delta), Specific tuned (low delta)',
            'args': ['--fishertune_mode', 'adaptive']
        },
        'perturbation_edge_dropout': {
            'description': 'DR-FIM with edge dropout perturbation',
            'args': ['--fishertune_mode', 'perturbation_edge_dropout', '--ft_perturbation_rate', '0.2']
        },
        'perturbation_popularity': {
            'description': 'DR-FIM with popularity-based weighting',
            'args': ['--fishertune_mode', 'perturbation_popularity', '--ft_perturbation_rate', '0.3']
        },
        'perturbation_noise': {
            'description': 'DR-FIM with noise injection',
            'args': ['--fishertune_mode', 'perturbation_noise', '--ft_noise_scale', '0.05']
        },
        'perturbation_cross_domain': {
            'description': 'DR-FIM with actual cross-domain perturbation',
            'args': ['--fishertune_mode', 'perturbation_cross_domain']
        }
    }

    return experiments


def run_single_experiment(exp_name, exp_config, base_args, log_dir):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*80}\n")

    # Construct command
    cmd = ['python', 'train_rec_fishertune.py'] + base_args + exp_config['args']
    cmd += ['--experiment_name', exp_name]
    cmd += ['--log_dir', log_dir]

    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    # Run experiment
    result = subprocess.run(cmd, capture_output=False, text=True)

    elapsed_time = (time.time() - start_time) / 60

    if result.returncode != 0:
        print(f"WARNING: Experiment {exp_name} failed with return code {result.returncode}")
        return None

    print(f"\nExperiment {exp_name} completed in {elapsed_time:.2f} minutes")

    # Load results
    results_file = os.path.join(log_dir, f"{exp_name}_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    else:
        print(f"WARNING: Results file not found: {results_file}")
        return None


def compare_results(all_results, log_dir):
    """Generate comparison report"""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 80)

    # Extract best metrics for comparison
    comparison_data = []

    for exp_name, results in all_results.items():
        if results is None:
            continue

        row = {
            'Experiment': exp_name,
            'Mode': results['config']['fishertune_mode'],
            'Training Time (min)': results['training_time']
        }

        # Get best metrics for each domain
        for domain, metrics in results['best_metrics'].items():
            row[f'{domain}_NDCG10'] = metrics.get('NDCG_10', 0)
            row[f'{domain}_HT10'] = metrics.get('HT_10', 0)

        # Compute average across domains
        ndcg_values = [v for k, v in row.items() if k.endswith('_NDCG10')]
        ht_values = [v for k, v in row.items() if k.endswith('_HT10')]

        if ndcg_values:
            row['Avg_NDCG10'] = sum(ndcg_values) / len(ndcg_values)
        if ht_values:
            row['Avg_HT10'] = sum(ht_values) / len(ht_values)

        comparison_data.append(row)

    if not comparison_data:
        print("No results to compare!")
        return

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Sort by average NDCG@10
    if 'Avg_NDCG10' in df.columns:
        df = df.sort_values('Avg_NDCG10', ascending=False)

    # Print table
    print("\n📊 BEST TEST METRICS:\n")
    print(df.to_string(index=False, float_format='{:.4f}'.format))

    # Compute improvement over baseline
    baseline_ndcg = None
    baseline_ht = None

    if 'baseline' in all_results and all_results['baseline'] is not None:
        baseline_row = df[df['Experiment'] == 'baseline'].iloc[0]
        baseline_ndcg = baseline_row['Avg_NDCG10']
        baseline_ht = baseline_row['Avg_HT10']

        print(f"\n📈 IMPROVEMENTS OVER BASELINE (Avg NDCG@10: {baseline_ndcg:.4f}):\n")

        improvements = []
        for _, row in df.iterrows():
            if row['Experiment'] != 'baseline':
                ndcg_imp = ((row['Avg_NDCG10'] - baseline_ndcg) / baseline_ndcg) * 100
                ht_imp = ((row['Avg_HT10'] - baseline_ht) / baseline_ht) * 100
                improvements.append({
                    'Experiment': row['Experiment'],
                    'NDCG@10 Improvement (%)': ndcg_imp,
                    'HT@10 Improvement (%)': ht_imp,
                    'Training Time (min)': row['Training Time (min)']
                })

        imp_df = pd.DataFrame(improvements)
        imp_df = imp_df.sort_values('NDCG@10 Improvement (%)', ascending=False)
        print(imp_df.to_string(index=False, float_format='{:.2f}'.format))

    # Save comparison results
    comparison_file = os.path.join(log_dir, 'experiment_comparison.csv')
    df.to_csv(comparison_file, index=False)
    print(f"\n💾 Comparison saved to {comparison_file}")

    # Generate detailed report
    report_file = os.path.join(log_dir, 'experiment_report.txt')
    with open(report_file, 'w') as f:
        f.write("FisherTune Experiment Report\n")
        f.write("=" * 80 + "\n\n")

        for exp_name, results in all_results.items():
            if results is None:
                continue

            f.write(f"\nExperiment: {exp_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mode: {results['config']['fishertune_mode']}\n")
            f.write(f"Training Time: {results['training_time']:.2f} minutes\n")

            f.write("\nBest Metrics:\n")
            for domain, metrics in results['best_metrics'].items():
                f.write(f"  {domain}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"    {metric_name}: {value:.4f}\n")

            if 'fishertune_stats' in results and results['fishertune_stats']:
                latest_stats = results['fishertune_stats'][-1]
                f.write("\nFisherTune Statistics:\n")
                for k, v in latest_stats.items():
                    f.write(f"  {k}: {v}\n")

            f.write("\n")

    print(f"📝 Detailed report saved to {report_file}")

    # Highlight best configuration
    best_exp = df.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION: {best_exp['Experiment']}")
    print(f"   Average NDCG@10: {best_exp['Avg_NDCG10']:.4f}")
    print(f"   Average HT@10: {best_exp['Avg_HT10']:.4f}")
    print(f"   Training Time: {best_exp['Training Time (min)']:.2f} minutes")

    if baseline_ndcg:
        imp = ((best_exp['Avg_NDCG10'] - baseline_ndcg) / baseline_ndcg) * 100
        print(f"   Improvement over baseline: {imp:.2f}%")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser('FisherTune Experiment Runner')

    # Base training arguments
    parser.add_argument('--domains', type=str, default='sport_cloth',
                        help='Domain to run experiments on')
    parser.add_argument('--task', type=str, default='dual-user-intra',
                        help='CDR task type')
    parser.add_argument('--num_epoch', type=int, default=50,
                        help='Number of training epochs (reduced for quick comparison)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--aggregator', type=str, default='mean',
                        help='Aggregator type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')

    # Experiment selection
    parser.add_argument('--experiments', type=str, nargs='+',
                        default=['baseline', 'fim_only', 'dr_fim', 'adaptive'],
                        help='List of experiments to run')
    parser.add_argument('--all_experiments', action='store_true',
                        help='Run all experiments')

    # Output
    parser.add_argument('--log_dir', type=str, default='./experiment_logs',
                        help='Directory for experiment logs')

    args = parser.parse_args()

    # Create log directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n🚀 FisherTune Experiment Suite")
    print(f"📁 Log directory: {log_dir}")

    # Base arguments for all experiments
    base_args = [
        '--domains', args.domains,
        '--task', args.task,
        '--num_epoch', str(args.num_epoch),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--aggregator', args.aggregator,
        '--seed', str(args.seed)
    ]

    if args.cuda:
        base_args.append('--cuda')

    # Get experiments to run
    all_configs = get_all_experiment_configs()

    if args.all_experiments:
        experiments_to_run = list(all_configs.keys())
    else:
        experiments_to_run = args.experiments

    print(f"\n📋 Experiments to run: {experiments_to_run}")

    # Validate experiment names
    for exp_name in experiments_to_run:
        if exp_name not in all_configs:
            print(f"ERROR: Unknown experiment '{exp_name}'")
            print(f"Available: {list(all_configs.keys())}")
            return

    # Run experiments
    all_results = {}
    total_start_time = time.time()

    for i, exp_name in enumerate(experiments_to_run, 1):
        print(f"\n[{i}/{len(experiments_to_run)}] ", end="")
        results = run_single_experiment(exp_name, all_configs[exp_name], base_args, log_dir)
        all_results[exp_name] = results

    total_time = (time.time() - total_start_time) / 60

    print(f"\n\n🏁 All experiments completed in {total_time:.2f} minutes")

    # Generate comparison report
    compare_results(all_results, log_dir)

    # Save all results
    all_results_file = os.path.join(log_dir, 'all_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 All results saved to {all_results_file}")


if __name__ == "__main__":
    main()
