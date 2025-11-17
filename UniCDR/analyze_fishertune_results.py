#!/usr/bin/env python
"""
Analysis and visualization of FisherTune experiment results
"""

import json
import os
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_experiment_results(log_dir):
    """Load all experiment results from a directory"""
    results = {}

    # Look for result JSON files
    for file in Path(log_dir).glob("*_results.json"):
        exp_name = file.stem.replace("_results", "")
        with open(file, 'r') as f:
            results[exp_name] = json.load(f)

    return results


def plot_training_curves(results, output_dir):
    """Plot training loss curves for all experiments"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_name, exp_results in results.items():
        if 'train_losses' not in exp_results:
            continue

        epochs = [entry['epoch'] for entry in exp_results['train_losses']]
        losses = [entry['avg_loss'] for entry in exp_results['train_losses']]

        ax.plot(epochs, losses, label=exp_name, marker='o', markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_dir}/training_curves.png")


def plot_validation_metrics(results, output_dir):
    """Plot validation metric progression"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot NDCG@10
    ax = axes[0]
    for exp_name, exp_results in results.items():
        if 'val_metrics' not in exp_results:
            continue

        # Aggregate across domains
        for domain, domain_metrics in exp_results['val_metrics'].items():
            epochs = [entry['epoch'] for entry in domain_metrics]
            ndcg10 = [entry['NDCG_10'] for entry in domain_metrics]
            ax.plot(epochs, ndcg10, label=f"{exp_name}_{domain}", marker='s', markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('NDCG@10')
    ax.set_title('Validation NDCG@10 Progression')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)

    # Plot HT@10
    ax = axes[1]
    for exp_name, exp_results in results.items():
        if 'val_metrics' not in exp_results:
            continue

        for domain, domain_metrics in exp_results['val_metrics'].items():
            epochs = [entry['epoch'] for entry in domain_metrics]
            ht10 = [entry['HT_10'] for entry in domain_metrics]
            ax.plot(epochs, ht10, label=f"{exp_name}_{domain}", marker='s', markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hit Rate@10')
    ax.set_title('Validation Hit Rate@10 Progression')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved validation metrics to {output_dir}/validation_metrics.png")


def plot_fishertune_statistics(results, output_dir):
    """Plot FisherTune-specific statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Filter experiments with FisherTune stats
    ft_results = {k: v for k, v in results.items()
                  if 'fishertune_stats' in v and len(v['fishertune_stats']) > 0}

    if not ft_results:
        print("No FisherTune statistics found")
        return

    # Plot 1: Average Fisher Information (Shared vs Specific)
    ax = axes[0, 0]
    for exp_name, exp_results in ft_results.items():
        stats = exp_results['fishertune_stats']
        epochs = [s.get('epoch', i) for i, s in enumerate(stats)]
        shared_fim = [s.get('avg_shared_fim', 0) for s in stats]
        specific_fim = [s.get('avg_specific_fim', 0) for s in stats]

        ax.plot(epochs, shared_fim, label=f"{exp_name} (shared)", linestyle='-')
        ax.plot(epochs, specific_fim, label=f"{exp_name} (specific)", linestyle='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Fisher Information')
    ax.set_title('Fisher Information: Shared vs Domain-Specific')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Parameter Selection Ratios
    ax = axes[0, 1]
    for exp_name, exp_results in ft_results.items():
        stats = exp_results['fishertune_stats']
        epochs = [s.get('epoch', i) for i, s in enumerate(stats)]
        shared_ratio = [s.get('shared_selection_ratio', 0) * 100 for s in stats]
        specific_ratio = [s.get('specific_selection_ratio', 0) * 100 for s in stats]

        ax.plot(epochs, shared_ratio, label=f"{exp_name} (shared)", linestyle='-')
        ax.plot(epochs, specific_ratio, label=f"{exp_name} (specific)", linestyle='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Selection Ratio (%)')
    ax.set_title('Parameter Selection Ratios Over Time')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)

    # Plot 3: Best Test NDCG@10 Bar Chart
    ax = axes[1, 0]
    exp_names = list(results.keys())
    avg_ndcg = []
    for exp_name in exp_names:
        metrics = results[exp_name].get('best_metrics', {})
        ndcg_values = [m.get('NDCG_10', 0) for m in metrics.values()]
        avg_ndcg.append(np.mean(ndcg_values) if ndcg_values else 0)

    bars = ax.bar(range(len(exp_names)), avg_ndcg, color='steelblue')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylabel('Average NDCG@10')
    ax.set_title('Best Test NDCG@10 by Experiment')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, avg_ndcg):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Plot 4: Training Time Comparison
    ax = axes[1, 1]
    training_times = [results[exp_name].get('training_time', 0) for exp_name in exp_names]

    bars = ax.bar(range(len(exp_names)), training_times, color='coral')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time by Experiment')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, training_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fishertune_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved FisherTune analysis to {output_dir}/fishertune_analysis.png")


def generate_latex_table(results, output_dir):
    """Generate LaTeX table for paper"""
    data = []

    for exp_name, exp_results in results.items():
        row = {'Experiment': exp_name}

        # Get training time
        row['Time (min)'] = f"{exp_results.get('training_time', 0):.1f}"

        # Get best metrics
        for domain, metrics in exp_results.get('best_metrics', {}).items():
            row[f'{domain} NDCG@10'] = f"{metrics.get('NDCG_10', 0):.4f}"
            row[f'{domain} HT@10'] = f"{metrics.get('HT_10', 0):.4f}"

        # Compute average
        ndcg_values = [m.get('NDCG_10', 0) for m in exp_results.get('best_metrics', {}).values()]
        ht_values = [m.get('HT_10', 0) for m in exp_results.get('best_metrics', {}).values()]

        row['Avg NDCG@10'] = f"{np.mean(ndcg_values):.4f}" if ndcg_values else "0.0000"
        row['Avg HT@10'] = f"{np.mean(ht_values):.4f}" if ht_values else "0.0000"

        data.append(row)

    df = pd.DataFrame(data)

    # Sort by average NDCG@10
    df = df.sort_values('Avg NDCG@10', ascending=False)

    # Generate LaTeX
    latex_str = df.to_latex(index=False, escape=False)

    # Save
    latex_file = os.path.join(output_dir, 'results_table.tex')
    with open(latex_file, 'w') as f:
        f.write("% FisherTune Experiment Results\n")
        f.write("% Auto-generated LaTeX table\n\n")
        f.write(latex_str)

    print(f"Saved LaTeX table to {latex_file}")

    # Also save CSV for easy editing
    csv_file = os.path.join(output_dir, 'results_table.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved CSV table to {csv_file}")


def compute_statistical_significance(results, baseline_name='baseline'):
    """
    Compute statistical significance of improvements over baseline

    Note: This is a simple comparison. For proper significance testing,
    you would need multiple runs with different seeds.
    """
    if baseline_name not in results:
        print(f"Baseline experiment '{baseline_name}' not found")
        return

    baseline_metrics = results[baseline_name].get('best_metrics', {})

    print("\n📊 PERFORMANCE IMPROVEMENTS ANALYSIS")
    print("=" * 60)

    for exp_name, exp_results in results.items():
        if exp_name == baseline_name:
            continue

        exp_metrics = exp_results.get('best_metrics', {})

        print(f"\n{exp_name} vs {baseline_name}:")

        total_improvement = 0
        num_domains = 0

        for domain in baseline_metrics:
            if domain not in exp_metrics:
                continue

            baseline_ndcg = baseline_metrics[domain].get('NDCG_10', 0)
            exp_ndcg = exp_metrics[domain].get('NDCG_10', 0)

            if baseline_ndcg > 0:
                improvement = ((exp_ndcg - baseline_ndcg) / baseline_ndcg) * 100
                total_improvement += improvement
                num_domains += 1

                symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
                print(f"  {domain}: {improvement:+.2f}% {symbol}")

        if num_domains > 0:
            avg_improvement = total_improvement / num_domains
            print(f"  Average: {avg_improvement:+.2f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser('FisherTune Results Analyzer')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as log_dir)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.log_dir

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"📂 Loading results from {args.log_dir}")

    # Load results
    results = load_experiment_results(args.log_dir)

    if not results:
        print("No experiment results found!")
        return

    print(f"Found {len(results)} experiments: {list(results.keys())}")

    # Generate analysis
    print("\n🎨 Generating visualizations...")

    plot_training_curves(results, args.output_dir)
    plot_validation_metrics(results, args.output_dir)
    plot_fishertune_statistics(results, args.output_dir)

    print("\n📊 Generating tables...")
    generate_latex_table(results, args.output_dir)

    print("\n📈 Computing improvements...")
    compute_statistical_significance(results)

    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
