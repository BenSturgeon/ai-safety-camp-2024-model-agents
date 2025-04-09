# src/plot_ablation_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse
import glob
from pathlib import Path

# --- Configuration --- (Can be moved to args or kept as defaults)
TARGET_ENTITIES = ['gem', 'blue_key', 'green_key', 'red_key'] # Entities to plot
ENTITY_COLORS = { # Entity-specific colors for potential bar chart
    'gem': 'gold',
    'blue_key': 'royalblue',
    'green_key': 'forestgreen',
    'red_key': 'firebrick'
}
# -------------------

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize channel ablation sweep results.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing the per-channel result CSV files.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the plots. Defaults to results_dir.")
    parser.add_argument("--file_pattern", type=str, default="results_layer*.csv",
                        help="Glob pattern to find the per-channel CSV files.")
    parser.add_argument("--plot_title_prefix", type=str, default="Ablation Results",
                        help="Prefix for plot titles.")
    # Add other args later if needed (e.g., for top N bar chart)

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        parser.error(f"Results directory not found: {args.results_dir}")

    if args.output_dir is None:
        args.output_dir = args.results_dir # Save plots in the same dir as results by default
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def load_and_process_data(results_dir, file_pattern):
    """Load all matching CSVs and calculate collection counts per channel."""
    all_files = glob.glob(os.path.join(results_dir, file_pattern))
    if not all_files:
        print(f"Error: No files found matching pattern '{file_pattern}' in directory '{results_dir}'")
        return None, 0

    print(f"Found {len(all_files)} result files. Loading...")
    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Could not read file {f}. Error: {e}")

    if not df_list:
        print("Error: No valid CSV files could be loaded.")
        return None, 0

    df = pd.concat(df_list, ignore_index=True)
    print(f"Combined data has {len(df)} rows.")

    # --- Data Processing --- 
    # Ensure required columns exist (removed ended_by_gem)
    required_cols = ['channel_kept', 'trial', 'initial_entities', 'remaining_entities']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns in combined CSV: {required_cols}")
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Missing: {missing}")
        return None, 0

    # Infer num_trials from the data
    num_trials = df.groupby('channel_kept')['trial'].nunique().max()
    if pd.isna(num_trials) or num_trials == 0:
        num_trials = df['trial'].max() + 1 # Fallback
    print(f"Inferred number of trials per channel: {num_trials}")

    # Fill NaNs in entity strings
    df['initial_entities'] = df['initial_entities'].fillna('')
    df['remaining_entities'] = df['remaining_entities'].fillna('')

    # Helper function to convert string to set
    def str_to_set(entity_str):
        if pd.isna(entity_str) or not entity_str:
            return set()
        return set(e.strip() for e in entity_str.split(',') if e.strip())

    # Create set columns for efficient checking
    initial_sets = df['initial_entities'].apply(str_to_set)
    remaining_sets = df['remaining_entities'].apply(str_to_set)

    # Re-calculate collection status based purely on initial vs remaining
    print("Re-calculating collection status based *only* on initial vs remaining sets...")
    for entity in TARGET_ENTITIES:
        col_name = f'collected_{entity}'
        # Collected if initially present AND *not* remaining
        df[col_name] = (initial_sets.apply(lambda s: entity in s)) & (~remaining_sets.apply(lambda s: entity in s))

    # Group by channel and sum the boolean columns to get counts
    agg_counts = df.groupby('channel_kept')[[f'collected_{entity}' for entity in TARGET_ENTITIES]].sum()

    # Rename columns for clarity (e.g., collected_gem -> gem)
    agg_counts.columns = [col.replace('collected_', '') for col in agg_counts.columns]

    # Ensure all target entities are present as columns, fill with 0 if missing
    for entity in TARGET_ENTITIES:
        if entity not in agg_counts.columns:
            agg_counts[entity] = 0

    # Reorder columns to match TARGET_ENTITIES order
    agg_counts = agg_counts[TARGET_ENTITIES]

    print("Data processed. Aggregated counts per channel:")
    print(agg_counts.head())

    return agg_counts, num_trials

def plot_heatmap(agg_counts_df, num_trials, output_dir, title_prefix):
    """Generate and save a heatmap of entity collection counts."""
    if agg_counts_df is None or agg_counts_df.empty:
        print("Cannot plot heatmap: No aggregated data.")
        return

    plt.figure(figsize=(max(20, agg_counts_df.shape[0] * 0.15), 6))
    
    # Calculate frequency for heatmap colors, but annotate with counts
    agg_freq_df = agg_counts_df / num_trials 

    sns.heatmap(agg_freq_df.T, # Transpose for channels on x-axis, entities on y
                annot=agg_counts_df.T, # Annotate with the raw counts
                fmt="d",           # Format annotations as integers
                cmap="viridis",    # Colormap (can change)
                linewidths=.5,
                linecolor='lightgray',
                cbar_kws={'label': f'Collection Frequency (Count / {num_trials} trials)'})

    plt.title(f'{title_prefix}: Entity Collection Counts per Active Channel', fontsize=16)
    plt.xlabel("Channel Kept Active (Ablating Others)", fontsize=12)
    plt.ylabel("Entity Type", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f"{title_prefix.replace(' ', '_')}_heatmap.png")
    try:
        plt.savefig(plot_filename)
        print(f"Heatmap saved to: {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error saving heatmap: {e}")
        plt.close()

def main():
    args = parse_args()
    agg_data, num_trials = load_and_process_data(args.results_dir, args.file_pattern)

    if agg_data is not None:
        plot_heatmap(agg_data, num_trials, args.output_dir, args.plot_title_prefix)
        # Add calls to other plotting functions here later (e.g., plot_stacked_bar)

if __name__ == "__main__":
    main() 