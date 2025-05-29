# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse
from pathlib import Path
import sys
import torch # Added for dead channel detection
import traceback # Add this import

# Attempt to import functions for dead channel detection
try:
    from sae_cnn import load_sae_from_checkpoint, ordered_layer_names as sae_ordered_layer_names
    from detect_dead_channels import identify_dead_channels, export_simple_csv as dc_export_simple_csv
    DETECTION_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules for dead channel detection: {e}. Detection will be skipped if requested.")
    DETECTION_IMPORTS_AVAILABLE = False

# --- Configuration ---
DEFAULT_RESULTS_DIR = "/home/ubuntu/east-ben/ai-safety-camp-2024-model-agents/results/layer8_alpha/sae_alpha_l8_n50_i0.0-5.0/20250510_123004/"
TARGET_ENTITIES = ['gem', 'blue_key', 'green_key', 'red_key'] # Entities to include in the plot
ENTITY_COLORS = { # Entity-specific colors
    'gem': 'gold',
    'blue_key': 'royalblue',
    'green_key': 'forestgreen',
    'red_key': 'firebrick'
}
# -------------------

def is_running_in_jupyter():
    """Check if the current script is running in a Jupyter kernel."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        return False
    except ImportError:
        return False

def load_results_csv(base_dir):
    """
    Load the main results CSV file from the base directory
    
    Args:
        base_dir (str): Base directory containing the results CSV file
        
    Returns:
        Tuple[pd.DataFrame, str]: Loaded dataframe and the layer name found in CSV filename
    """
    # Look for the CSV file directly in the base directory
    csv_files = list(Path(base_dir).glob("quantitative_results_*.csv"))
    
    if not csv_files:
        print(f"No quantitative results CSV files found in {base_dir}")
        return None, None
        
    # Use the first CSV file found
    csv_path = csv_files[0]
    print(f"Loading data from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Extract layer name from the filename
        parts = csv_path.stem.split('_') # E.g., ['quantitative', 'results', 'conv4a', 'sae']
        if len(parts) > 2:
            # Assume layer name is the third part (adjust if needed)
            layer_name = parts[2] 
            print(f"  Inferred layer name '{layer_name}' from filename {csv_path.name}")
        else:
            print(f"  Warning: Could not infer layer name from filename {csv_path.name}")
            layer_name = "UnknownLayer"
            
    except Exception as e:
        print(f"  Error loading CSV {csv_path}: {e}")
        return None, None
    
    # Verify the entities in the data
    unique_entities = df['target_entity_name'].unique()
    print(f"  Entities found in data: {unique_entities}")
    
    return df, layer_name

def main(results_dir, num_top_channels=20, auto_select_first_file=True, dead_channels_csv=None,
         run_detection_if_csv_missing=True, exclude_dead_channels=False,
         sae_checkpoint_for_detection_arg=None, model_checkpoint_for_detection=None,
         detection_samples=256, detection_batch_size=32, detection_threshold=1e-6):
    """
    Generate bubble plots showing distribution of successful interventions by entity.
    
    Args:
        results_dir (str): Directory containing quantitative results CSV file
        num_top_channels (int): Number of top performing channels to include
        auto_select_first_file (bool): Automatically use the first file found (for interactive use)
        dead_channels_csv (str, optional): Path to CSV file listing dead channels.
        run_detection_if_csv_missing (bool): If True and dead_channels_csv is not found/valid, attempt detection.
        exclude_dead_channels (bool): If True, dead channels are excluded from top N and plot.
        sae_checkpoint_for_detection (str, optional): Path to SAE checkpoint for on-the-fly detection.
        model_checkpoint_for_detection (str, optional): Path to base model for on-the-fly detection.
        detection_samples (int): Number of samples for dead channel detection.
        detection_batch_size (int): Batch size for dead channel detection.
        detection_threshold (float): Threshold for dead channel detection.
    """
    # Try to configure matplotlib for inline display in Jupyter/IPython if running interactively
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        print("Matplotlib configured for inline display")
    except:
        pass  # Silently continue if not in IPython

    print(f"Using results directory: {results_dir}")
    
    # Load data from the main CSV file
    results_df, layer_name = load_results_csv(results_dir)
    
    if results_df is None or len(results_df) == 0:
        print("No valid data found. Exiting.")
        return
    
    # --- Load or Detect Dead Channels ---
    dead_channel_set = set()
    loaded_from_csv = False

    # Determine the actual SAE checkpoint path to use
    # sae_checkpoint_for_detection_arg is from the command line
    # sae_checkpoint_for_detection will be the one actually used by the script
    sae_checkpoint_to_use = sae_checkpoint_for_detection_arg 

    if sae_checkpoint_to_use is None and layer_name and layer_name != "UnknownLayer":
        # Only attempt inference if no explicit path is given AND we have a valid layer_name
        print("Attempting to infer SAE checkpoint path as none was explicitly provided...")
        
        # results_dir might be like: quantitative_interventions_sae_python_narrow_range/sae_checkpoint_step_20000000_alpha_l8
        # We want the last part as the stem for the .pt file: sae_checkpoint_step_20000000_alpha_l8
        sae_filename_stem = Path(results_dir.rstrip('/')).name 
        sae_filename = f"{sae_filename_stem}.pt"

        # Try to find the layer number from sae_ordered_layer_names using the layer_name from the CSV
        layer_number_for_path = None
        if sae_ordered_layer_names: # Check if the import was successful
            for num, name_in_map in sae_ordered_layer_names.items():
                if name_in_map == layer_name:
                    layer_number_for_path = num
                    break
        
        if layer_number_for_path is not None:
            # Construct path like: checkpoints_of_interest/layer_8_conv4a/sae_checkpoint_step_20000000_alpha_l8.pt
            inferred_path = Path("checkpoints_of_interest") / f"layer_{layer_number_for_path}_{layer_name}" / sae_filename
            if inferred_path.exists():
                sae_checkpoint_to_use = str(inferred_path)
                print(f"  Inferred and using SAE checkpoint: {sae_checkpoint_to_use}")
            else:
                print(f"  Warning: Inferred SAE checkpoint path does not exist or is not accessible: {inferred_path}")
                print(f"  Please provide --sae_checkpoint_for_detection manually if detection is desired.")
        else:
            print(f"  Warning: Could not determine layer number for '{layer_name}' from sae_ordered_layer_names to infer SAE checkpoint path.")
            print(f"  Please provide --sae_checkpoint_for_detection manually if detection is desired.")

    # If dead_channels_csv was not provided by user, construct default path using current results_dir and layer_name
    effective_dead_channels_csv = dead_channels_csv
    if effective_dead_channels_csv is None and layer_name and layer_name != "UnknownLayer":
        # Attempt to use a generic name first, or a specific one if that was the convention.
        # The output from detect_dead_channels.py is `dead_channels_{layer_name}.csv` or `detected_dead_channels_{layer_name}.csv`
        # Let's check for both if a specific one isn't always used.
        potential_csv_name1 = f"dead_channels_{layer_name}.csv"
        potential_csv_name2 = f"detected_dead_channels_{layer_name}.csv"
        path1 = os.path.join(results_dir, potential_csv_name1)
        path2 = os.path.join(results_dir, potential_csv_name2)
        
        if os.path.exists(path1):
            effective_dead_channels_csv = path1
            print(f"Using default dead channels CSV path: {effective_dead_channels_csv}")
        elif os.path.exists(path2):
            effective_dead_channels_csv = path2
            print(f"Using default dead channels CSV path: {effective_dead_channels_csv}")
        else:
            # If a specific name like "dead_channels_conv4a.csv" was the old default and is expected, check for it too.
            # This makes the transition smoother if old files exist.
            specific_fallback_name = "dead_channels_conv4a.csv"
            specific_fallback_path = os.path.join(results_dir, specific_fallback_name)
            if os.path.exists(specific_fallback_path):
                 effective_dead_channels_csv = specific_fallback_path
                 print(f"Using specific fallback dead channels CSV path: {effective_dead_channels_csv}")
            # else: # No default found, will proceed to detection if enabled

    if effective_dead_channels_csv: # Renamed variable to avoid confusion with arg
        try:
            dead_df = pd.read_csv(effective_dead_channels_csv)
            if 'channel' in dead_df.columns and 'is_dead' in dead_df.columns:
                dead_channel_set = set(dead_df[dead_df['is_dead'] == True]['channel'])
                print(f"Loaded {len(dead_channel_set)} dead channels from {effective_dead_channels_csv}: {sorted(list(dead_channel_set))}")
                loaded_from_csv = True
            elif 'channel_number' in dead_df.columns and 'is_dead' in dead_df.columns: # Legacy
                dead_channel_set = set(dead_df[dead_df['is_dead'] == True]['channel_number'])
                print(f"Loaded {len(dead_channel_set)} dead channels (using 'channel_number') from {effective_dead_channels_csv}: {sorted(list(dead_channel_set))}")
                loaded_from_csv = True
            else:
                print(f"Warning: Dead channels CSV ({effective_dead_channels_csv}) malformed. Skipping.")
        except FileNotFoundError:
            # Only print info if the user explicitly provided the path or if we constructed one and it wasn't found.
            # If dead_channels_csv was None initially and we didn't find a default, this is fine.
            if dead_channels_csv is not None or (effective_dead_channels_csv and not os.path.exists(effective_dead_channels_csv)):
                 print(f"Info: Dead channels CSV not found at {effective_dead_channels_csv}.")
        except Exception as e:
            print(f"Warning: Error loading dead channels CSV {effective_dead_channels_csv}: {e}. Skipping.")
    print("checking dead channels")
    print(loaded_from_csv,run_detection_if_csv_missing, DETECTION_IMPORTS_AVAILABLE)
    if not loaded_from_csv and run_detection_if_csv_missing:
        if not DETECTION_IMPORTS_AVAILABLE:
            print("ERROR: Dead channel detection was requested (or is default behavior), but necessary modules could not be imported. Please check dependencies (e.g., sae_cnn, detect_dead_channels, and their own dependencies like procgen). Exiting.")
            sys.exit(1) # Exit if detection is critical and imports failed
        elif not sae_checkpoint_to_use: # Check the determined path (either inferred or from arg)
            print("Warning: SAE checkpoint for on-the-fly detection not provided and could not be inferred. Skipping detection.")
        else:
            print(f"Attempting to detect dead channels using SAE: {sae_checkpoint_to_use}")
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Infer layer_number from SAE checkpoint path or from results_df's layer_name
                # This part needs to be robust. For now, assume layer_name from results_df is a good hint.
                # And that sae_ordered_layer_names maps it to a number.
                sae_layer_number_for_detection = None
                if layer_name and sae_ordered_layer_names:
                    # Attempt to find layer number from layer_name (e.g. "conv4a")
                    for num, name_in_map in sae_ordered_layer_names.items():
                        if name_in_map == layer_name:
                            sae_layer_number_for_detection = num
                            break
                
                if sae_layer_number_for_detection is None:
                    # Fallback: try to parse from SAE checkpoint path (less reliable here)
                    for part in Path(sae_checkpoint_to_use).parts:
                        if part.startswith("layer_"):
                            try:
                                sae_layer_number_for_detection = int(part.split("_")[1])
                                break
                            except: pass
                
                if sae_layer_number_for_detection is None:
                    raise ValueError(f"Could not determine SAE layer number for detection from SAE path '{sae_checkpoint_to_use}' or layer_name '{layer_name}'.")

                print(f"  Loading SAE for layer {sae_layer_number_for_detection} ({layer_name}) for detection...")
                sae = load_sae_from_checkpoint(sae_checkpoint_to_use, device)
                
                print("  Loading base model for detection...")
                # Use provided model path or let load_interpretable_model find its default
                base_model_path = model_checkpoint_for_detection if model_checkpoint_for_detection else None

                
                print(f"  Running dead channel identification (samples={detection_samples}, batch={detection_batch_size})...")
                # Ensure identify_dead_channels gets the correct layer_name if it needs it for module path
                # The `layer_number` argument for identify_dead_channels is the key.
                
                detected_dead_list, detected_channel_stats = identify_dead_channels(
                    sae, sae_layer_number_for_detection, 
                    num_samples=detection_samples, batch_size=detection_batch_size, threshold=detection_threshold
                )
                print(f"{detected_dead_list=}")
                dead_channel_set = set(detected_dead_list)
                print(f"  Detected {len(dead_channel_set)} dead channels: {sorted(list(dead_channel_set))}")

                # Save the detected dead channels for future use
                if detected_channel_stats:
                    detected_csv_name = f"detected_dead_channels_{layer_name}.csv"
                    detected_csv_path = os.path.join(results_dir, detected_csv_name)
                    try:
                        dc_export_simple_csv(detected_channel_stats, detected_csv_path)
                        # The dc_export_simple_csv already prints a message.
                    except Exception as e_save:
                        print(f"Warning: Could not save detected dead channels to {detected_csv_path}: {e_save}")

            except Exception as e_detect:
                print(f"Warning: Error during dead channel detection: {e_detect}")
                print("Full traceback for dead channel detection error:")
                traceback.print_exc() # This will print the full stack trace
    # --- End Load or Detect Dead Channels ---
    
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
    
    # --- Optionally exclude dead channels from success rate calculations and top channel selection ---
    if exclude_dead_channels and dead_channel_set:
        print(f"Excluding {len(dead_channel_set)} dead channels from top channel consideration: {sorted(list(dead_channel_set))}")
        # Get channels that are NOT dead
        non_dead_channels = overall_success_rate.index.difference(pd.Index(list(dead_channel_set)))
        overall_success_rate = overall_success_rate.loc[non_dead_channels]
        if overall_success_rate.empty:
            print("Warning: After excluding dead channels, no channels remain for top channel selection.")
    # --- End exclude dead channels ---

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
        print("No top channels identified based on criteria (or num_top_channels is 0). Skipping plot generation.")
        return # Exit if no top channels to plot
    
    # Filter for successful trials OF THE TOP CHANNELS
    successful_trials_df = results_df[(results_df['success'] == True) & (results_df['channel'].isin(top_channels))]
    
    # Check if any data exists for the top channels
    if successful_trials_df.empty:
        print(f"No successful trials found for the top {len(top_channels)} channels. Cannot generate plot.")
        return
    
    print(f"Found {len(successful_trials_df)} successful trials for top {len(top_channels)} channels to plot.")
    
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
    
    # Determine figure dimensions based on number of top channels
    num_plot_channels = len(top_channels) 
    fig_height = max(6, num_plot_channels * 0.35) 
    fig_width = max(12, num_plot_channels * 0.5) # Adjusted minimum width
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create the stripplot with custom colors
    sns_plot = sns.stripplot(data=successful_trials_df, 
                  x='channel',
                  y='intervention_value',
                  hue='target_entity_name',
                  palette=custom_palette,
                  alpha=0.7, 
                  s=8, 
                  jitter=0.25, 
                  dodge=True, 
                  zorder=10, 
                  order=sorted(top_channels.tolist())) # Use top_channels for order
    
    # Style the plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Channel Index")
    plt.ylabel("Successful Intervention Value")
    
    # Prepare custom x-tick labels with asterisks for dead channels
    # sns.stripplot uses the 'order' parameter to set its ticks.
    # We will get the current tick positions and labels, then modify the labels.
    ax = plt.gca() # Get current axes
    
    # Ensure ticks are set up by drawing the plot first if not already
    plt.gcf().canvas.draw()

    current_ticks = ax.get_xticks() # These are positions
    
    # The labels for these ticks correspond to sorted(top_channels.tolist())
    # because that's what 'order' in stripplot is set to.
    ordered_channels_for_plot = sorted(top_channels.tolist())

    if len(current_ticks) == len(ordered_channels_for_plot): # Check if ticks match our ordered channels
        new_xtick_labels = []
        for i, tick_pos in enumerate(current_ticks):
            channel_val = ordered_channels_for_plot[i]
            label_str = str(channel_val)
            if dead_channel_set and int(channel_val) in dead_channel_set:
                new_xtick_labels.append(f"{label_str}*")
            else:
                new_xtick_labels.append(label_str)
        ax.set_xticks(current_ticks) # Re-set ticks to ensure positions are correct
        ax.set_xticklabels(new_xtick_labels)
    else:
        print("Warning: Mismatch between number of x-ticks and ordered channels. Skipping dead channel asterisks.")
        # Fallback: just let matplotlib handle labels if mismatch
        # This case should ideally not happen if stripplot behaves as expected with 'order'

    # --- Extract Designation for Plot Title ---
    sae_designation = ""
    if sae_checkpoint_to_use: # If an SAE checkpoint is involved (use the determined path)
        actual_sae_stem = Path(sae_checkpoint_to_use).stem.lower() # e.g., "sae_checkpoint_step_20000000_alpha_l8"
        
        if "_alpha" in actual_sae_stem:
            sae_designation = "Alpha"
        elif "_beta" in actual_sae_stem:
            sae_designation = "Beta"
        elif "_gamma" in actual_sae_stem:
            sae_designation = "Gamma"
        # Add more extraction logic here if needed (e.g., using regex for more complex patterns)

    # Create more informative title using the found layer name and top channel count
    title_layer_name = layer_name if layer_name else "UnknownLayer"
    
    plot_main_title_line = f"Distribution of Successful Interventions by Entity Type (Top {len(top_channels)} Channels)"
    
    subtitle_parts = [f"Layer: {title_layer_name} SAE"]
    if sae_designation:
        subtitle_parts.append(f"Designation: {sae_designation}")
    subtitle_parts.append(f"Entities: {len(entities_present)}")
    plot_subtitle_line = " | ".join(subtitle_parts)
    
    plt.title(f"{plot_main_title_line}\n{plot_subtitle_line}")
    plt.legend(title='Entity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='both', linestyle='--', alpha=0.6) # Grid on both axes
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Construct output PNG path dynamically
    output_filename = f"entity_distribution_plot_{layer_name}_top_{len(top_channels)}_channels.png"
    output_png_path = os.path.join(results_dir, output_filename)
    
    # Save the plot to the determined path
    try:
        plt.savefig(output_png_path, bbox_inches='tight')
        print(f"\nEntity distribution plot saved to: {output_png_path}")
    except Exception as e:
        print(f"\nError saving plot to {output_png_path}: {e}")
    
    # Always display the plot - important for interactive mode
    print("Displaying plot...")
    plt.show()
    
    return plt

# For interactive mode use - will run when imported in Jupyter
if is_running_in_jupyter():
    results_dir = DEFAULT_RESULTS_DIR # Use the configured default
    print(f"Running in interactive mode. Using results directory: {results_dir}")
    main(results_dir)

# For running directly (in command line)
if __name__ == "__main__" and not is_running_in_jupyter():
    parser = argparse.ArgumentParser(description="Generate entity distribution plots with custom colors")
    parser.add_argument("--results_dir", type=str,
                        default=DEFAULT_RESULTS_DIR,
                        help="Directory containing the quantitative results CSV file")
    parser.add_argument("--num_top_channels", type=int, default=50,
                        help="Number of top performing channels to include in the plot")
    parser.add_argument("--dead_channels_csv", type=str, 
                        default=None, # Changed default to None
                        help="Optional path to CSV file listing dead channels. If not given, attempts to find [results_dir]/dead_channels_[layer_name].csv or [results_dir]/detected_dead_channels_[layer_name].csv")
    # New arguments for on-the-fly detection
    parser.add_argument("--skip-on-the-fly-detection",
                        action="store_false",
                        dest="run_detection_if_csv_missing",
                        help="If specified, will NOT attempt on-the-fly dead channel detection if a CSV is not found. Default behavior is to attempt detection.")
    parser.add_argument("--exclude_dead_channels", action="store_true",
                        help="If set, identified dead channels will be excluded from top channel selection and the plot.")
    parser.add_argument("--sae_checkpoint_for_detection", type=str, default=None,
                        dest="sae_checkpoint_for_detection_arg",
                        help="Path to SAE checkpoint for on-the-fly dead channel detection. If not provided, attempts to infer from --results_dir and CSV.")
    parser.add_argument("--model_checkpoint_for_detection", type=str, default=None,
                        help="Path to base model checkpoint for on-the-fly dead channel detection.")
    parser.add_argument("--detection_samples", type=int, default=256,
                        help="Number of samples for dead channel detection.")
    parser.add_argument("--detection_batch_size", type=int, default=32,
                        help="Batch size for dead channel detection.")
    parser.add_argument("--detection_threshold", type=float, default=1e-6,
                        help="Threshold for dead channel detection.")

    args = parser.parse_args()
    main(args.results_dir, args.num_top_channels,
         dead_channels_csv=args.dead_channels_csv,
         run_detection_if_csv_missing=args.run_detection_if_csv_missing,
         exclude_dead_channels=args.exclude_dead_channels,
         sae_checkpoint_for_detection_arg=args.sae_checkpoint_for_detection_arg,
         model_checkpoint_for_detection=args.model_checkpoint_for_detection,
         detection_samples=args.detection_samples,
         detection_batch_size=args.detection_batch_size,
         detection_threshold=args.detection_threshold)

# %%