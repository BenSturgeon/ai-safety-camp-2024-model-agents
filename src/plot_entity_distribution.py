# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse
from pathlib import Path
import sys

# Entity-specific colors
ENTITY_COLORS = {
    'gem': 'gold',
    'blue_key': 'royalblue',
    'green_key': 'forestgreen',
    'red_key': 'firebrick'
}

# Entities to include in the plot
TARGET_ENTITIES = ['gem', 'blue_key', 'green_key', 'red_key']

def combine_entity_csv_files(base_dir):
    """
    Combine CSV files from multiple entity subdirectories
    
    Args:
        base_dir (str): Base directory containing entity result subdirectories
        
    Returns:
        pd.DataFrame: Combined dataframe with all entity data
    """
    all_data = []
    
    # Check each entity directory
    for entity in TARGET_ENTITIES:
        entity_dir = os.path.join(base_dir, f"{entity}_results")
        if not os.path.exists(entity_dir):
            print(f"Directory not found for {entity}: {entity_dir}")
            continue
            
        # Look for the CSV file
        csv_path = os.path.join(entity_dir, "quantitative_results_conv3a_base.csv")
        if not os.path.exists(csv_path):
            print(f"CSV file not found for {entity}: {csv_path}")
            continue
            
        print(f"Loading data for {entity} from {csv_path}")
        entity_df = pd.read_csv(csv_path)
        
        # Verify the entity name in the data matches our expected entity
        # (Some datasets might use different names internally)
        unique_entities = entity_df['target_entity_name'].unique()
        print(f"Entities found in {entity} data: {unique_entities}")
        
        # Add to our combined dataset
        all_data.append(entity_df)
    
    if not all_data:
        print("No entity data found!")
        return None
        
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data has {len(combined_df)} rows from {len(all_data)} entity files")
    return combined_df

def main(results_dir, output_png=None, num_top_channels=20, auto_select_first_file=True):
    """
    Generate bubble plots showing distribution of successful interventions by entity.
    
    Args:
        results_dir (str): Directory containing quantitative results CSV files
        output_png (str): Path to save the output plot (optional)
        num_top_channels (int): Number of top performing channels to include
        auto_select_first_file (bool): Automatically use the first file found (for interactive use)
    """
    # Try to configure matplotlib for inline display in Jupyter/IPython if running interactively
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        print("Matplotlib configured for inline display")
    except:
        pass  # Silently continue if not in IPython

    print(f"Using results directory: {results_dir}")
    
    # Load combined data from entity subdirectories
    results_df = combine_entity_csv_files(results_dir)
    
    if results_df is None or len(results_df) == 0:
        print("No valid data found. Exiting.")
        return
    
    # Ensure required columns exist
    required_cols = ['outcome', 'target_acquired', 'channel', 'intervention_value', 'target_entity_name']
    missing = [col for col in required_cols if col not in results_df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return
    
    results_df['success'] = (results_df['outcome'] == 'intervention_location') & (results_df['target_acquired'] == False)
    
    # Calculate overall success rate per channel
    overall_success_counts = results_df[results_df['success']].groupby('channel').size()
    overall_valid_trials = results_df[results_df['outcome'] != 'error'].groupby('channel').size()
    
    # Handle cases where a channel might only have error trials
    overall_valid_trials = overall_valid_trials[overall_valid_trials > 0]
    if overall_valid_trials.empty:
        print("No channels with valid (non-error) trials found. Cannot determine top channels.")
        return
    
    overall_success_counts_reindexed = overall_success_counts.reindex(overall_valid_trials.index, fill_value=0)
    overall_success_rate = (overall_success_counts_reindexed / overall_valid_trials * 100)
    
    # Find top channels
    top_channels = pd.Index([])
    if num_top_channels > 0:
        non_zero_success = overall_success_rate[overall_success_rate > 0]
        num_actual_visualize = min(num_top_channels, len(non_zero_success))
        if num_actual_visualize > 0:
            top_channels = non_zero_success.nlargest(num_actual_visualize).index
            print(f"Identified Top {len(top_channels)} channels: {sorted(top_channels.tolist())}")
        else:
            print("No channels had > 0% overall success rate.")
    
    if top_channels.empty:
        print("No top channels identified based on criteria. Skipping plot generation.")
        return
    
    # Filter for successful trials
    successful_trials_df = results_df[(results_df['success'] == True) & (results_df['channel'].isin(top_channels))]
    
    if successful_trials_df.empty:
        print(f"No successful trials found for the top {len(top_channels)} channels. Cannot generate plot.")
        return
    
    print(f"Found {len(successful_trials_df)} successful trials for top channels to plot.")
    
    # Check for supported entities
    entities_in_data = successful_trials_df['target_entity_name'].unique()
    print(f"Entities found in data: {entities_in_data}")
    
    # Add more detailed entity analysis
    entity_counts = successful_trials_df['target_entity_name'].value_counts()
    print("\nEntity distribution in successful trials:")
    for entity, count in entity_counts.items():
        print(f"  {entity}: {count} trials")
    
    normalized_entity_map = {}
    for entity in entities_in_data:
        lower_entity = entity.lower().strip().replace(' ', '_')
        if lower_entity != entity and lower_entity in ENTITY_COLORS:
            normalized_entity_map[entity] = lower_entity
            print(f"Note: Will normalize entity name '{entity}' to '{lower_entity}'")
    
    # Apply normalization if needed
    if normalized_entity_map:
        successful_trials_df['target_entity_name'] = successful_trials_df['target_entity_name'].replace(normalized_entity_map)
        entities_in_data = successful_trials_df['target_entity_name'].unique()
        print(f"Entities after normalization: {entities_in_data}")
    
    # Expand our color mapping to support any entity we find
    for entity in entities_in_data:
        if entity not in ENTITY_COLORS:
            # For any unknown entity, generate a distinct color
            print(f"Adding new entity to color map: {entity}")
            # Use more muted default colors for unknown entities
            if 'blue' in entity.lower():
                ENTITY_COLORS[entity] = 'cornflowerblue'
            elif 'green' in entity.lower():
                ENTITY_COLORS[entity] = 'mediumseagreen'
            elif 'red' in entity.lower():
                ENTITY_COLORS[entity] = 'indianred'
            elif 'gem' in entity.lower():
                ENTITY_COLORS[entity] = 'gold'
            else:
                ENTITY_COLORS[entity] = 'darkgray'
    
    filtered_trials_df = successful_trials_df
    
    entities_present = list(set(filtered_trials_df['target_entity_name'].unique()))
    custom_palette = {entity: ENTITY_COLORS.get(entity, 'darkgray') for entity in entities_present}
    
    print("\nColor mapping for plot:")
    for entity, color in custom_palette.items():
        print(f"  {entity}: {color}")
    
    # Generate plot
    print("Generating entity distribution plot...")
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        print("Seaborn-v0_8-darkgrid style not available, using default.")
        plt.style.use('default')
    
    # Determine figure dimensions based on number of channels
    fig_height = max(6, len(top_channels) * 0.35)
    fig_width = max(12, len(top_channels) * 0.5)
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create the stripplot with custom colors
    sns_plot = sns.stripplot(data=filtered_trials_df,
                  x='channel',
                  y='intervention_value',
                  hue='target_entity_name',
                  palette=custom_palette,  # Use our custom entity colors
                  alpha=0.7,  # Slightly increase opacity for better visibility
                  s=8,  # Size of bubbles
                  jitter=0.25,  # Reduced jitter within each color strip
                  dodge=True,  # Separate strips for each hue (color)
                  zorder=10,  # Ensure points are drawn above grid
                  order=sorted(top_channels.tolist()))
    
    # Style the plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Channel Index")
    plt.ylabel("Successful Intervention Value")
    
    # Create more informative title
    plt.title(f"Distribution of Successful Interventions by Entity Type\nLayer: conv3a | Entities: {len(entities_present)}")
    plt.legend(title='Entity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='both', linestyle='--', alpha=0.6)  # Grid on both axes for better readability
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ensure layout handles the legend
    
    # Save if output_png specified
    if output_png:
        output_dir = os.path.dirname(output_png)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_png, bbox_inches='tight')
        print(f"Entity distribution plot saved to: {output_png}")
    
    # Always display the plot - important for interactive mode
    print("Displaying plot...")
    plt.show()
    
    return plt  

if is_running_in_jupyter():
    # Path to the results directory
    results_dir = "/home/ubuntu/east-ben/ai-safety-camp-2024-model-agents/src/quant_results_parallel_base_conv3a"
    print(f"Running in interactive mode. Using results directory: {results_dir}")
    main(results_dir)    

def is_running_in_jupyter():
    """Check if the current script is running in a Jupyter kernel."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        return False
    except ImportError:
        return False

# For running directly (in command line)
if __name__ == "__main__" and not is_running_in_jupyter():
    parser = argparse.ArgumentParser(description="Generate entity distribution plots with custom colors")
    parser.add_argument("--results_dir", type=str, 
                        default="/home/ubuntu/east-ben/ai-safety-camp-2024-model-agents/src/quant_results_parallel_base_conv3a",
                        help="Directory containing quantitative results CSV files")
    parser.add_argument("--output_png", type=str, default=None,
                        help="Path to save the output PNG file (optional)")
    parser.add_argument("--num_top_channels", type=int, default=20,
                        help="Number of top performing channels to include in the plot")
    
    args = parser.parse_args()
    main(args.results_dir, args.output_png, args.num_top_channels)

# %%