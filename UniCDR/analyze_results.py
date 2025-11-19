"""
Analyze and compare FisherTune experiment results.
"""

import json
import os
import glob
import pandas as pd
import argparse
from collections import defaultdict


def load_training_log(log_path):
    """Load training log from JSON file."""
    with open(log_path, 'r') as f:
        return json.load(f)


def extract_best_metrics(log_data):
    """Extract best metrics from training log."""
    best_metrics = {}

    if 'metrics' not in log_data:
        return best_metrics

    for metric_entry in log_data['metrics']:
        epoch = metric_entry.get('epoch', 0)

        for key, value in metric_entry.items():
            if key == 'epoch':
                continue

            if isinstance(value, dict) and 'NDCG_10' in value:
                domain = key.replace('_test', '')
                is_test = '_test' in key

                metric_key = f"{domain}_{'test' if is_test else 'valid'}"

                if metric_key not in best_metrics:
                    best_metrics[metric_key] = {
                        'NDCG_10': 0,
                        'HT_10': 0,
                        'epoch': 0
                    }

                if value['NDCG_10'] > best_metrics[metric_key]['NDCG_10']:
                    best_metrics[metric_key] = {
                        'NDCG_10': value['NDCG_10'],
                        'HT_10': value['HT_10'],
                        'epoch': epoch
                    }

    return best_metrics


def extract_fishertune_stats(log_data):
    """Extract FisherTune statistics."""
    stats = {
        'use_fishertune': False,
        'avg_selection_ratio': 0,
        'final_threshold': 0,
        'num_fim_updates': 0
    }

    if 'fishertune_stats' not in log_data:
        return stats

    ft_stats = log_data['fishertune_stats']
    if not ft_stats:
        return stats

    stats['use_fishertune'] = True

    # Get latest stats
    latest = ft_stats[-1] if ft_stats else {}

    if 'scheduler_stats' in latest:
        sched_stats = latest['scheduler_stats']
        stats['avg_selection_ratio'] = sched_stats.get('selection_ratio', 0)
        stats['final_threshold'] = sched_stats.get('current_threshold', 0)

    # Count FIM updates
    stats['num_fim_updates'] = len([
        s for s in ft_stats
        if 'fim_history' in s and s['fim_history']
    ])

    return stats


def extract_loss_trend(log_data):
    """Extract loss trend."""
    if 'losses' not in log_data:
        return {}

    losses = log_data['losses']
    if not losses:
        return {}

    return {
        'initial_loss': losses[0]['avg_loss'] if losses else 0,
        'final_loss': losses[-1]['avg_loss'] if losses else 0,
        'min_loss': min(l['avg_loss'] for l in losses) if losses else 0,
        'loss_reduction': (
            (losses[0]['avg_loss'] - losses[-1]['avg_loss']) / losses[0]['avg_loss']
            if losses and losses[0]['avg_loss'] > 0 else 0
        )
    }


def analyze_experiment(log_path):
    """Analyze a single experiment."""
    log_data = load_training_log(log_path)

    exp_name = os.path.basename(log_path).replace('_training_log.json', '')

    result = {
        'experiment': exp_name,
        **extract_best_metrics(log_data),
        **extract_fishertune_stats(log_data),
        **extract_loss_trend(log_data)
    }

    return result


def compare_experiments(log_dir='logs'):
    """Compare all experiments in the log directory."""
    log_files = glob.glob(os.path.join(log_dir, '*_training_log.json'))

    if not log_files:
        print(f"No log files found in {log_dir}")
        return None

    results = []
    for log_path in log_files:
        try:
            result = analyze_experiment(log_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {log_path}: {e}")

    return results


def create_summary_table(results):
    """Create summary table from results."""
    if not results:
        return None

    # Extract domain names
    all_domains = set()
    for r in results:
        for key in r.keys():
            if '_valid' in key or '_test' in key:
                all_domains.add(key)

    # Create table rows
    rows = []
    for r in results:
        row = {
            'Experiment': r['experiment'],
            'FisherTune': r.get('use_fishertune', False),
        }

        # Add metrics for each domain
        for domain in sorted(all_domains):
            if domain in r:
                row[f'{domain}_NDCG@10'] = r[domain].get('NDCG_10', 0)
                row[f'{domain}_HT@10'] = r[domain].get('HT_10', 0)
            else:
                row[f'{domain}_NDCG@10'] = 0
                row[f'{domain}_HT@10'] = 0

        # Add FisherTune stats
        row['Selection_Ratio'] = r.get('avg_selection_ratio', 0)
        row['Final_Threshold'] = r.get('final_threshold', 0)
        row['Loss_Reduction'] = r.get('loss_reduction', 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def rank_experiments(df, metric_col):
    """Rank experiments by a specific metric."""
    if df is None or metric_col not in df.columns:
        return None

    ranked = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1  # 1-based ranking
    ranked.index.name = 'Rank'

    return ranked


def generate_report(results, output_path='results/experiment_report.md'):
    """Generate comprehensive experiment report."""
    if not results:
        print("No results to report")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = create_summary_table(results)
    if df is None:
        return

    with open(output_path, 'w') as f:
        f.write("# FisherTune Experiment Results\n\n")
        f.write("## Summary\n\n")

        # Overall statistics
        num_experiments = len(results)
        num_fishertune = sum(1 for r in results if r.get('use_fishertune', False))

        f.write(f"- Total Experiments: {num_experiments}\n")
        f.write(f"- FisherTune Experiments: {num_fishertune}\n")
        f.write(f"- Baseline Experiments: {num_experiments - num_fishertune}\n\n")

        # Best performing experiments
        f.write("## Performance Comparison\n\n")
        f.write("### All Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        # Rank by different metrics
        metric_columns = [col for col in df.columns if 'NDCG@10' in col]

        for metric in metric_columns:
            f.write(f"### Ranked by {metric}\n\n")
            ranked = rank_experiments(df, metric)
            if ranked is not None:
                f.write(ranked[[
                    'Experiment', metric, 'FisherTune',
                    'Selection_Ratio', 'Loss_Reduction'
                ]].head(10).to_markdown())
            f.write("\n\n")

        # FisherTune vs Baseline comparison
        f.write("## FisherTune vs Baseline\n\n")

        baseline_results = [r for r in results if not r.get('use_fishertune', False)]
        fishertune_results = [r for r in results if r.get('use_fishertune', False)]

        if baseline_results and fishertune_results:
            for metric in metric_columns:
                domain = metric.replace('_NDCG@10', '')

                baseline_best = max(
                    baseline_results,
                    key=lambda x: x.get(domain, {}).get('NDCG_10', 0) if isinstance(x.get(domain), dict) else 0
                )
                fishertune_best = max(
                    fishertune_results,
                    key=lambda x: x.get(domain, {}).get('NDCG_10', 0) if isinstance(x.get(domain), dict) else 0
                )

                baseline_score = baseline_best.get(domain, {}).get('NDCG_10', 0) if isinstance(baseline_best.get(domain), dict) else 0
                fishertune_score = fishertune_best.get(domain, {}).get('NDCG_10', 0) if isinstance(fishertune_best.get(domain), dict) else 0

                improvement = (fishertune_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0

                f.write(f"**{domain}**:\n")
                f.write(f"- Baseline Best: {baseline_best['experiment']} ({baseline_score:.4f})\n")
                f.write(f"- FisherTune Best: {fishertune_best['experiment']} ({fishertune_score:.4f})\n")
                f.write(f"- Improvement: {improvement:.2f}%\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        # Find best configuration
        best_overall = max(
            results,
            key=lambda x: sum(
                x.get(k, {}).get('NDCG_10', 0) if isinstance(x.get(k), dict) else 0
                for k in x.keys() if '_test' in k
            )
        )

        f.write(f"### Best Overall Configuration\n\n")
        f.write(f"**{best_overall['experiment']}**\n\n")

        if best_overall.get('use_fishertune', False):
            f.write("FisherTune Settings:\n")
            f.write(f"- Final Selection Ratio: {best_overall.get('avg_selection_ratio', 0):.2%}\n")
            f.write(f"- Final Threshold: {best_overall.get('final_threshold', 0):.4f}\n")
            f.write(f"- Loss Reduction: {best_overall.get('loss_reduction', 0):.2%}\n")

        f.write("\n")

    print(f"Report generated: {output_path}")

    # Also save as CSV for further analysis
    csv_path = output_path.replace('.md', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV data saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze FisherTune experiments')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory containing training logs')
    parser.add_argument('--output', type=str, default='results/experiment_report.md',
                        help='Output report path')
    parser.add_argument('--compare', type=str, nargs='+',
                        help='Specific experiments to compare')

    args = parser.parse_args()

    if args.compare:
        # Compare specific experiments
        results = []
        for exp_name in args.compare:
            log_path = os.path.join(args.log_dir, f'{exp_name}_training_log.json')
            if os.path.exists(log_path):
                results.append(analyze_experiment(log_path))
            else:
                print(f"Warning: {log_path} not found")
    else:
        # Compare all experiments
        results = compare_experiments(args.log_dir)

    if results:
        generate_report(results, args.output)

        # Print quick summary to console
        print("\n" + "=" * 60)
        print("QUICK SUMMARY")
        print("=" * 60)

        df = create_summary_table(results)
        if df is not None:
            print(df.to_string())
    else:
        print("No results to analyze")


if __name__ == "__main__":
    main()
