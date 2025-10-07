#!/usr/bin/env python3
"""
Analyze whether different entities show similar activation trajectories with consistent offsets
or completely different patterns over time.

Key questions:
1. Are activation trajectories parallel (same shape, different offset)?
2. Or do different entities have fundamentally different temporal dynamics?
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import correlation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_experiment_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_trajectories(results):
    """
    Extract activation trajectories for each entity across timesteps.

    Returns:
        dict: {entity_name: {channel_idx: [activations across timesteps]}}
    """
    trajectories = {}

    # Initialize structure
    for step_data in results['steps']:
        for entity_name in step_data['entity_activations'].keys():
            if entity_name not in trajectories:
                trajectories[entity_name] = {}

    # Collect timestep data
    for step_idx, step_data in enumerate(results['steps']):
        for entity_name, activations in step_data['entity_activations'].items():
            for ch_idx, activation in enumerate(activations):
                if ch_idx not in trajectories[entity_name]:
                    trajectories[entity_name][ch_idx] = []
                trajectories[entity_name][ch_idx].append(activation)

    return trajectories

def analyze_trajectory_similarity(trajectories, channel_idx):
    """
    Analyze if trajectories for different entities are similar shapes with offsets.

    Returns dict with:
    - correlations: pairwise correlations between trajectories
    - mean_offsets: average offset between each entity pair
    - variance_ratios: how much variance is explained by offset vs shape
    """
    entity_names = list(trajectories.keys())
    entity_trajectories = {}

    # Get trajectories for this channel
    for entity in entity_names:
        if channel_idx in trajectories[entity]:
            entity_trajectories[entity] = np.array(trajectories[entity][channel_idx])

    if len(entity_trajectories) < 2:
        return None

    results = {
        'correlations': {},
        'mean_offsets': {},
        'normalized_distances': {},
        'is_parallel': False
    }

    # Calculate pairwise correlations and offsets
    entities = list(entity_trajectories.keys())
    correlation_values = []

    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            e1, e2 = entities[i], entities[j]
            traj1 = entity_trajectories[e1]
            traj2 = entity_trajectories[e2]

            # Pearson correlation (shape similarity)
            if len(traj1) > 1 and len(traj2) > 1:
                corr, _ = stats.pearsonr(traj1, traj2)
                results['correlations'][f"{e1}_vs_{e2}"] = corr
                correlation_values.append(corr)

                # Mean offset
                offset = np.mean(traj1 - traj2)
                results['mean_offsets'][f"{e1}_vs_{e2}"] = offset

                # Normalized distance (trajectory shape after removing mean)
                traj1_centered = traj1 - np.mean(traj1)
                traj2_centered = traj2 - np.mean(traj2)
                if np.std(traj1_centered) > 0 and np.std(traj2_centered) > 0:
                    traj1_norm = traj1_centered / np.std(traj1_centered)
                    traj2_norm = traj2_centered / np.std(traj2_centered)
                    norm_dist = np.mean((traj1_norm - traj2_norm)**2)
                    results['normalized_distances'][f"{e1}_vs_{e2}"] = norm_dist

    # Determine if trajectories are parallel
    # Criteria: high correlations and consistent offsets
    if correlation_values:
        avg_correlation = np.mean(correlation_values)
        results['avg_correlation'] = avg_correlation
        results['is_parallel'] = avg_correlation > 0.8

    return results

def plot_trajectory_comparison(trajectories, top_channels, variant_name='standard', output_dir='plots'):
    """
    Plot trajectories for top channels to visualize parallel vs divergent patterns.
    """
    num_channels = len(top_channels)
    fig, axes = plt.subplots(num_channels, 2, figsize=(12, 3*num_channels))
    if num_channels == 1:
        axes = axes.reshape(1, -1)

    entity_colors = {
        'gem': 'gold',
        'blue_key': 'blue',
        'green_key': 'green',
        'red_key': 'red'
    }

    for idx, ch_idx in enumerate(top_channels):
        # Raw trajectories
        ax_raw = axes[idx, 0]

        for entity_name in sorted(trajectories.keys()):
            if ch_idx in trajectories[entity_name]:
                traj = trajectories[entity_name][ch_idx]
                color = entity_colors.get(entity_name, 'gray')
                ax_raw.plot(traj, label=entity_name.replace('_', ' '),
                          color=color, marker='o', markersize=4, linewidth=2, alpha=0.7)

        ax_raw.set_xlabel('Timestep')
        ax_raw.set_ylabel('Activation')
        ax_raw.set_title(f'Channel {ch_idx} - Raw Trajectories')
        ax_raw.legend(loc='best', fontsize=8)
        ax_raw.grid(alpha=0.3)

        # Normalized trajectories (remove mean to show shape)
        ax_norm = axes[idx, 1]

        for entity_name in sorted(trajectories.keys()):
            if ch_idx in trajectories[entity_name]:
                traj = np.array(trajectories[entity_name][ch_idx])
                traj_normalized = traj - np.mean(traj)  # Remove mean offset
                color = entity_colors.get(entity_name, 'gray')
                ax_norm.plot(traj_normalized, label=entity_name.replace('_', ' '),
                           color=color, marker='o', markersize=4, linewidth=2, alpha=0.7)

        ax_norm.set_xlabel('Timestep')
        ax_norm.set_ylabel('Centered Activation')
        ax_norm.set_title(f'Channel {ch_idx} - Mean-Centered Trajectories')
        ax_norm.legend(loc='best', fontsize=8)
        ax_norm.grid(alpha=0.3)
        ax_norm.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / f'trajectory_patterns_{variant_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved trajectory patterns to {output_path}")
    plt.close()

def analyze_offset_vs_shape_variance(trajectories, num_channels=32):
    """
    For each channel, determine how much variance is due to constant offsets
    vs different trajectory shapes.
    """
    results = []

    for ch_idx in range(num_channels):
        # Collect all trajectories for this channel
        all_trajectories = []
        entity_labels = []

        for entity_name in trajectories.keys():
            if ch_idx in trajectories[entity_name]:
                traj = trajectories[entity_name][ch_idx]
                if len(traj) > 1:  # Need at least 2 timesteps
                    all_trajectories.append(traj)
                    entity_labels.append(entity_name)

        if len(all_trajectories) < 2:
            continue

        # Ensure all trajectories have the same length
        min_len = min(len(t) for t in all_trajectories)
        all_trajectories = [t[:min_len] for t in all_trajectories]

        # Convert to numpy array: (n_entities, n_timesteps)
        X = np.array(all_trajectories)

        # Total variance
        total_variance = np.var(X)

        # Variance after removing entity-specific means (shape variance)
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        shape_variance = np.var(X_centered)

        # Variance of the means (offset variance)
        entity_means = np.mean(X, axis=1)
        offset_variance = np.var(entity_means)

        # Proportion of variance explained by offset
        if total_variance > 0:
            offset_ratio = offset_variance / total_variance
            shape_ratio = shape_variance / total_variance
        else:
            offset_ratio = 0
            shape_ratio = 0

        # Calculate average correlation between centered trajectories
        correlations = []
        for i in range(len(X_centered)):
            for j in range(i+1, len(X_centered)):
                if np.std(X_centered[i]) > 0 and np.std(X_centered[j]) > 0:
                    corr, _ = stats.pearsonr(X_centered[i], X_centered[j])
                    correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else 0

        results.append({
            'channel': ch_idx,
            'total_variance': total_variance,
            'offset_variance': offset_variance,
            'shape_variance': shape_variance,
            'offset_ratio': offset_ratio,
            'shape_ratio': shape_ratio,
            'avg_shape_correlation': avg_correlation,
            'is_offset_dominated': offset_ratio > 0.7,
            'is_parallel': offset_ratio > 0.7 and avg_correlation > 0.5
        })

    return results

def plot_variance_decomposition(variance_results, variant_name='standard', output_dir='plots'):
    """
    Plot how variance is decomposed into offset vs shape for each channel.
    """
    channels = [r['channel'] for r in variance_results]
    offset_ratios = [r['offset_ratio'] for r in variance_results]
    shape_ratios = [r['shape_ratio'] for r in variance_results]
    correlations = [r['avg_shape_correlation'] for r in variance_results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Stacked bar chart of variance decomposition
    ax1 = axes[0]
    x = np.arange(len(channels))
    width = 0.8

    ax1.bar(x, offset_ratios, width, label='Offset variance', color='steelblue', alpha=0.8)
    ax1.bar(x, shape_ratios, width, bottom=offset_ratios, label='Shape variance', color='coral', alpha=0.8)

    ax1.set_ylabel('Proportion of Variance')
    ax1.set_title('Variance Decomposition: Offset (Entity Type) vs Shape (Temporal Dynamics)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{c}' for c in channels], fontsize=8)
    ax1.set_xlabel('Channel')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='70% threshold')

    # Correlation of centered trajectories
    ax2 = axes[1]
    colors = ['green' if c > 0.5 else 'gray' for c in correlations]
    ax2.bar(x, correlations, width, color=colors, alpha=0.7)
    ax2.set_ylabel('Average Shape Correlation')
    ax2.set_title('Trajectory Shape Similarity (After Removing Offsets)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{c}' for c in channels], fontsize=8)
    ax2.set_xlabel('Channel')
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Similarity threshold')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / f'variance_decomposition_{variant_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved variance decomposition to {output_path}")
    plt.close()

def main():
    """Run trajectory pattern analysis."""
    import sys

    # Check if we should analyze gem variant and/or different checkpoint
    analyze_gem = '--gem' in sys.argv[1:] if len(sys.argv) > 1 else False
    use_30k = '--30k' in sys.argv[1:] if len(sys.argv) > 1 else False
    use_35k = '--35k' in sys.argv[1:] if len(sys.argv) > 1 else False

    if use_30k:
        checkpoint = "30k"
        timestamp = "20251002_130856"
    elif use_35k:
        checkpoint = "35k"
        timestamp = "20251002_131814"
    else:
        checkpoint = "60k"
        timestamp = "20251002_112108"

    if analyze_gem:
        print(f"Analyzing GEM VARIANT ({checkpoint} checkpoint)...")
        if checkpoint == "60k":
            # 60k files don't have checkpoint in name
            filename = f'src/entity_activation_analysis/results/t_corridor_gem_variant_{timestamp}.json'
        else:
            filename = f'src/entity_activation_analysis/results/t_corridor_gem_variant_{checkpoint}_{timestamp}.json'
        results = load_experiment_results(filename)
        variant_name = f"gem_variant_{checkpoint}"
    else:
        print(f"Analyzing STANDARD VARIANT ({checkpoint} checkpoint)...")
        if checkpoint == "60k":
            # 60k files don't have checkpoint in name
            filename = f'src/entity_activation_analysis/results/t_corridor_standard_{timestamp}.json'
        else:
            filename = f'src/entity_activation_analysis/results/t_corridor_standard_{checkpoint}_{timestamp}.json'
        results = load_experiment_results(filename)
        variant_name = f"standard_{checkpoint}"

    # Extract trajectories
    trajectories = extract_trajectories(results)

    print("=" * 60)
    print("TRAJECTORY PATTERN ANALYSIS")
    print("=" * 60)

    # Analyze variance decomposition for all channels
    variance_results = analyze_offset_vs_shape_variance(trajectories)

    # Sort by offset ratio (most offset-dominated first)
    variance_results.sort(key=lambda x: x['offset_ratio'], reverse=True)

    print("\nTop 10 Offset-Dominated Channels (Parallel Trajectories):")
    print("Channel | Offset% | Shape% | Shape Corr | Parallel?")
    print("-" * 55)
    for r in variance_results[:10]:
        print(f"  {r['channel']:2d}    | {r['offset_ratio']*100:5.1f}%  | "
              f"{r['shape_ratio']*100:5.1f}% |   {r['avg_shape_correlation']:+.2f}    | "
              f"{'YES' if r['is_parallel'] else 'NO'}")

    # Count parallel channels
    parallel_channels = [r for r in variance_results if r['is_parallel']]
    offset_dominated = [r for r in variance_results if r['is_offset_dominated']]

    print(f"\nSummary:")
    print(f"  Parallel channels (offset >70%, corr >0.5): {len(parallel_channels)}/{len(variance_results)}")
    print(f"  Offset-dominated channels (offset >70%): {len(offset_dominated)}/{len(variance_results)}")

    if parallel_channels:
        print(f"  Parallel channel IDs: {sorted([r['channel'] for r in parallel_channels])}")

    # Channels with different patterns
    shape_dominated = [r for r in variance_results if r['shape_ratio'] > 0.5]
    print(f"\nShape-dominated channels (different patterns): {len(shape_dominated)}/{len(variance_results)}")
    if shape_dominated:
        print(f"  Shape-varying channel IDs: {sorted([r['channel'] for r in shape_dominated[:10]])}")

    # Create output directory
    output_dir = Path('src/entity_activation_analysis/plots')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot variance decomposition
    plot_variance_decomposition(variance_results, variant_name, output_dir)

    # Plot trajectories for top offset-dominated and shape-dominated channels
    top_parallel = [r['channel'] for r in variance_results[:5] if r['offset_ratio'] > 0.5]
    if top_parallel:
        print(f"\nPlotting trajectories for top parallel channels: {top_parallel}")
        plot_trajectory_comparison(trajectories, top_parallel, variant_name, output_dir)

    # Detailed analysis for specific interesting channels
    print("\n" + "=" * 60)
    print("DETAILED CHANNEL ANALYSIS")
    print("=" * 60)

    # Analyze top 3 offset-dominated channels in detail
    for r in variance_results[:3]:
        ch_idx = r['channel']
        print(f"\nChannel {ch_idx}:")

        similarity = analyze_trajectory_similarity(trajectories, ch_idx)
        if similarity:
            print(f"  Average correlation between trajectories: {similarity.get('avg_correlation', 0):.3f}")
            print(f"  Parallel trajectories: {similarity['is_parallel']}")

            if similarity['mean_offsets']:
                print("  Mean offsets between entities:")
                for pair, offset in sorted(similarity['mean_offsets'].items())[:5]:
                    print(f"    {pair}: {offset:+.3f}")

    # Save detailed results (convert numpy types to Python types)
    results_output = {
        'variance_decomposition': [
            {k: float(v) if isinstance(v, (np.float32, np.float64, np.bool_)) else v
             for k, v in r.items()}
            for r in variance_results[:10]
        ],
        'parallel_channels': [int(r['channel']) for r in parallel_channels],
        'shape_varying_channels': [int(r['channel']) for r in shape_dominated[:10]],
        'summary': {
            'total_channels': len(variance_results),
            'parallel_count': len(parallel_channels),
            'offset_dominated_count': len(offset_dominated),
            'shape_dominated_count': len(shape_dominated)
        }
    }

    output_file = output_dir / f'trajectory_analysis_{variant_name}.json'
    with open(output_file, 'w') as f:
        json.dump(results_output, f, indent=2)

    print(f"\nDetailed results saved to {output_file}")
    print("\nPlots saved to:")
    print(f"  - {output_dir}/variance_decomposition.png")
    print(f"  - {output_dir}/trajectory_patterns.png")

if __name__ == "__main__":
    main()