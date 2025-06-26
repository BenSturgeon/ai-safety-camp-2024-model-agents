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
# TARGET_ENTITIES = ['gem', 'blue_key', 'green_key', 'red_key'] # Entities to include in the plot
TARGET_ENTITIES = [ 'blue_key'] # Entities to include in the plot

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
    # csv_path = csv_files[0] # Old: Load only the first file
    print(f"Found {len(csv_files)} quantitative results CSV files in {base_dir}:")
    for f in csv_files:
        print(f"  - {f.name}")

    all_dfs = []
    layer_name = None # Initialize layer_name

    for i, csv_path in enumerate(csv_files):
        try:
            df_single = pd.read_csv(csv_path)
            all_dfs.append(df_single)
            print(f"  Successfully loaded {csv_path.name} ({len(df_single)} rows)")

            # Infer layer name from the first successfully loaded CSV filename
            if i == 0: 
                parts = csv_path.stem.split('_') # E.g., ['quantitative', 'results', 'conv4a', 'base'] or ['quantitative', 'results', 'conv4a', 'base', 'worker', '0-9']
                if len(parts) >= 3:
                    # Attempt to find a non-numeric part after 'results' that isn't 'sae' or 'base' or 'worker'
                    # This logic is a bit more robust to handle worker suffixes.
                    # Layer name is typically parts[2] if no worker suffix, or before 'sae'/'base' if they exist.
                    # Example: quantitative_results_conv4a_base_worker_0-9.csv -> layer_name = conv4a
                    # Example: quantitative_results_8_sae_worker_0-9.csv -> layer_name = 8 (as string)
                    potential_layer_name_index = 2
                    if parts[potential_layer_name_index].lower() in ["sae", "base"] and len(parts) > potential_layer_name_index + 1:
                        # This case is unlikely if format is strictly quantitative_results_LAYER_TYPE...
                        # but handle if layer name might be after type, e.g. quantitative_results_sae_LAYER...
                        pass # Keep index at 2, or adjust if needed
                    
                    layer_name = parts[potential_layer_name_index]
                    print(f"    Inferred layer name '{layer_name}' from filename {csv_path.name}")
                else:
                    print(f"    Warning: Could not infer layer name from filename {csv_path.name}")

        except Exception as e:
            print(f"  Error loading CSV {csv_path}: {e}")
            # Continue to try loading other files
    
    if not all_dfs:
        print("No CSV files could be successfully loaded.")
        return None, None
    
    # Concatenate all loaded dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully concatenated {len(all_dfs)} CSV files into a single DataFrame with {len(df)} total rows.")
    
    # If layer_name is still None after trying all files, set a default
    if layer_name is None:
        layer_name = "UnknownLayer"
        print(f"  Warning: Layer name could not be inferred from any CSV. Defaulting to '{layer_name}'.")
            
    # Verify the entities in the data
    unique_entities = df['target_entity_name'].unique()
    print(f"  Entities found in data: {unique_entities}")
    
    return df, layer_name

def main(results_dir, num_top_channels=20, auto_select_first_file=True, dead_channels_csv=None,
         run_detection_if_csv_missing=True, exclude_dead_channels=False,
         sae_checkpoint_for_detection_arg=None, model_checkpoint_for_detection=None,
         detection_samples=256, detection_batch_size=32, detection_threshold=1e-6,
         focused_entities=None, specific_channels=None):
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
        focused_entities (list[str], optional): Specific entities to focus on for top channel selection and plotting.
        specific_channels (list[int], optional): Specific channel numbers to plot, bypassing top N selection.
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
    
    # === Start: Prepare normalized entity names and handle focused_entities ===
    # Add a normalized version of target_entity_name for robust matching
    # Using str(x) in lambda for robustness in case of non-string data
    results_df['normalized_target_entity_name'] = results_df['target_entity_name'].apply(
        lambda x: str(x).lower().strip().replace(' ', '_')
    )

    normalized_focused_entities = []
    if focused_entities:
        # Normalize the input focused_entities list
        normalized_focused_entities = sorted([e.lower().strip().replace(' ', '_') for e in focused_entities])
        print(f"Processing with focus on entities: {focused_entities} (normalized and sorted: {normalized_focused_entities})")
    # === End: Prepare normalized entity names ===
    
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
    required_cols = ['outcome', 'target_acquired', 'channel', 'intervention_value', 'target_entity_name', 'normalized_target_entity_name']
    missing = [col for col in required_cols if col not in results_df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return
    
    results_df['success'] = (results_df['outcome'] == 'intervention_location') & (results_df['target_acquired'] == False)
    
    # === Start: Modify success calculation based on focused_entities ===
    df_for_success_calc = results_df
    if focused_entities and normalized_focused_entities:
        print(f"Filtering data for top channel selection based on focused entities: {focused_entities}")
        # Use the 'normalized_target_entity_name' column for filtering
        temp_df = results_df[results_df['normalized_target_entity_name'].isin(normalized_focused_entities)]
        if not temp_df.empty:
            df_for_success_calc = temp_df
            print(f"  Filtered data for success calculation to {len(df_for_success_calc)} rows.")
        else:
            print(f"  Warning: No data found for the focused entities: {focused_entities}. Top channels will be selected based on all entities.")
            # df_for_success_calc remains results_df, so processing continues with all entities
    
    # Calculate overall success rate per channel using df_for_success_calc
    overall_success_counts = df_for_success_calc[df_for_success_calc['success']].groupby('channel').size()
    overall_valid_trials = df_for_success_calc[df_for_success_calc['outcome'] != 'error'].groupby('channel').size()
    # === End: Modify success calculation ===
    
    # Handle cases where a channel might only have error trials
    overall_valid_trials = overall_valid_trials[overall_valid_trials > 0]
    if overall_valid_trials.empty:
        if focused_entities:
            print(f"No channels with valid (non-error) trials found for focused entities: {focused_entities}. Cannot determine top channels.")
        else:
            print("No channels with valid (non-error) trials found. Cannot determine top channels.")
        return
    
    overall_success_counts_reindexed = overall_success_counts.reindex(overall_valid_trials.index, fill_value=0)
    overall_success_rate = (overall_success_counts_reindexed / overall_valid_trials * 100)
    
    # --- Optionally exclude dead channels from success rate calculations and top channel selection ---
    if exclude_dead_channels and dead_channel_set:
        print(f"Excluding {len(dead_channel_set)} dead channels from top channel consideration: {sorted(list(dead_channel_set))}")
        non_dead_channels = overall_success_rate.index.difference(pd.Index(list(dead_channel_set)))
        overall_success_rate = overall_success_rate.loc[non_dead_channels]
        if overall_success_rate.empty:
            print("Warning: After excluding dead channels, no channels remain for top channel selection.")
    # --- End exclude dead channels ---

    # --- Determine channels to plot: either top N or specifically provided --- 
    channels_to_plot_pd_index = pd.Index([]) # Use a pandas Index for consistency
    is_specific_channel_plot = False

    if specific_channels:
        is_specific_channel_plot = True
        # Convert list of ints to pandas Index of ints for consistency with top_channels
        channels_to_plot_pd_index = pd.Index(sorted(list(set(specific_channels)))) # Deduplicate and sort
        print(f"Plotting specifically requested channels: {channels_to_plot_pd_index.tolist()}")
        # If specific_channels are given, num_top_channels is ignored for selection, but used for title/filename if needed.
    elif num_top_channels > 0:
        non_zero_success = overall_success_rate[overall_success_rate > 0]
        num_actual_visualize = min(num_top_channels, len(non_zero_success))
        if num_actual_visualize > 0:
            channels_to_plot_pd_index = non_zero_success.nlargest(num_actual_visualize).index
            print(f"Identified Top {len(channels_to_plot_pd_index)} channels: {sorted(channels_to_plot_pd_index.tolist())}")
        else:
            print("No channels had > 0% overall success rate for top channel selection.")
    else: # num_top_channels is 0 or less
        print("num_top_channels is <= 0, no top channels will be selected.")

    if channels_to_plot_pd_index.empty:
        if specific_channels:
            print("Warning: --specific_channels provided, but the list was empty or resulted in no channels to plot. Skipping plot.")
        else:
            print("No channels identified to plot (either top N or specific). Skipping plot generation.")
        return
    # --- End channel determination ---
    
    # Filter for successful trials OF THE CHANNELS TO PLOT
    successful_trials_df = results_df[(results_df['success'] == True) & (results_df['channel'].isin(channels_to_plot_pd_index))]
    
    # === Start: Filter successful_trials_df by focused_entities for plotting ===
    if focused_entities and normalized_focused_entities:
        # Filter successful_trials_df to only include the focused entities using their normalized names
        successful_trials_df = successful_trials_df[successful_trials_df['normalized_target_entity_name'].isin(normalized_focused_entities)]
        
        if successful_trials_df.empty:
            print(f"No successful trials found for the focused entities {focused_entities} among the selected channels. Cannot generate plot.")
            return
        # This print is now more specific if filtering occurred
        # print(f"Filtered plot data to {len(successful_trials_df)} successful trials for focused entities: {focused_entities}.") 
    # === End: Filter successful_trials_df for plotting ===

    # Check if any data exists for the selected channels (and focused entities, if specified)
    if successful_trials_df.empty:
        # Message refined based on whether focused_entities was active
        if focused_entities:
             print(f"No successful trials found for the focused entities {focused_entities} (for the selected channels). Cannot generate plot.")
        else:
            print(f"No successful trials found for the selected channels. Cannot generate plot.")
        return
    
    print(f"Found {len(successful_trials_df)} successful trials for {len(channels_to_plot_pd_index)} selected channels to plot.")
    if focused_entities: # Add context if data was filtered for plotting
        print(f"  (Plot data is specifically for entities: {focused_entities})")
        
    # Original entity names from the data that will actually be plotted
    entities_to_plot_original_names = successful_trials_df['target_entity_name'].unique()
    print(f"Entities to be plotted (original names from data): {entities_to_plot_original_names}")
    
    # Detailed entity counts for what's being plotted
    entity_counts_in_plot_data = successful_trials_df['target_entity_name'].value_counts()
    print("\nEntity distribution in data to be plotted:")
    for entity, count in entity_counts_in_plot_data.items():
        print(f"  '{str(entity)}': {count} trials") # Use str(entity) for robustness
    
    # Dynamically expand ENTITY_COLORS for any new entities found in the plot data.
    # ENTITY_COLORS uses normalized keys (e.g., 'blue_key').
    for entity_original_name_from_data in entities_to_plot_original_names:
        # Normalize the name from data to match the expected key format in ENTITY_COLORS
        normalized_name_for_color_lookup = str(entity_original_name_from_data).lower().strip().replace(' ', '_')
        
        if normalized_name_for_color_lookup not in ENTITY_COLORS:
            print(f"Notice: Entity '{entity_original_name_from_data}' (normalized to '{normalized_name_for_color_lookup}') not in predefined ENTITY_COLORS. Assigning a default color.")
            # Default color generation logic
            if 'blue' in normalized_name_for_color_lookup:
                ENTITY_COLORS[normalized_name_for_color_lookup] = 'cornflowerblue'
            elif 'green' in normalized_name_for_color_lookup:
                ENTITY_COLORS[normalized_name_for_color_lookup] = 'mediumseagreen'
            elif 'red' in normalized_name_for_color_lookup:
                ENTITY_COLORS[normalized_name_for_color_lookup] = 'indianred'
            elif 'gem' in normalized_name_for_color_lookup:
                ENTITY_COLORS[normalized_name_for_color_lookup] = 'lightgoldenrodyellow' 
            else:
                ENTITY_COLORS[normalized_name_for_color_lookup] = 'darkgray'
    
    # Create the custom_palette for seaborn
    # Keys are the original entity names (as they appear in 'target_entity_name' column and used by 'hue')
    # Values are the colors fetched from ENTITY_COLORS using the normalized version of the name
    custom_palette = {
        original_name: ENTITY_COLORS.get(str(original_name).lower().strip().replace(' ', '_'), 'darkgray')
        for original_name in entities_to_plot_original_names
    }
    
    print("\nColor mapping for plot:")
    for entity, color in custom_palette.items():
        print(f"  '{str(entity)}': {color}") # Quoting entity for clarity
    
    # `filtered_trials_df` is `successful_trials_df`
    # `entities_present` for subtitle is `entities_to_plot_original_names`
    
    print("Generating entity distribution plot...")
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        print("Seaborn-v0_8-darkgrid style not available, using default.")
        plt.style.use('default')
    
    # Determine figure dimensions based on number of top channels
    num_plot_channels = len(channels_to_plot_pd_index) 
    fig_height = max(6, num_plot_channels * 0.35) 
    fig_width = max(12, num_plot_channels * 0.5) # Adjusted minimum width
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create the stripplot with custom colors
    sns_plot = sns.stripplot(data=successful_trials_df, 
                  x='channel',
                  y='intervention_value',
                  hue='target_entity_name', # Use original 'target_entity_name' for hue
                  palette=custom_palette,
                  alpha=0.7, 
                  s=8, 
                  jitter=0.25, 
                  dodge=True, 
                  zorder=10, 
                  order=sorted(channels_to_plot_pd_index.tolist())) # Use channels_to_plot_pd_index for order
    
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
    
    # The labels for these ticks correspond to sorted(channels_to_plot_pd_index.tolist())
    # because that's what 'order' in stripplot is set to.
    ordered_channels_for_plot = sorted(channels_to_plot_pd_index.tolist())

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
    
    # === Start: Modify plot title based on focused_entities and specific_channels ===
    entity_desc_for_title = "All Entities"
    if focused_entities:
        # Use the user-provided (original, non-normalized) names for the title
        if len(focused_entities) == 1:
            entity_desc_for_title = f"Entity: {focused_entities[0]}"
        else:
            entity_desc_for_title = f"Entities: {', '.join(focused_entities)}"
            
    # Adjust title based on whether specific channels or top N are plotted
    if is_specific_channel_plot:
        channel_desc_for_title = f"Specific Channels ({len(channels_to_plot_pd_index)})"
    else:
        channel_desc_for_title = f"Top {len(channels_to_plot_pd_index)} Channels"
            
    plot_main_title_line = f"Distribution of Successful Interventions ({entity_desc_for_title}, {channel_desc_for_title})"
    
    model_type_display = "Base" # Default to Base
    if sae_checkpoint_to_use: # If an SAE checkpoint path is determined/used
        model_type_display = "SAE"

    subtitle_parts = [f"Layer: {title_layer_name} {model_type_display}"]
    if model_type_display == "SAE" and sae_designation: # Only add designation if it's an SAE and designation exists
        subtitle_parts.append(f"Designation: {sae_designation}")
    
    # Use the count of unique entity names that are actually in the plotted data
    subtitle_parts.append(f"Entities Displayed: {len(entities_to_plot_original_names)}")
    plot_subtitle_line = " | ".join(subtitle_parts)
    
    plt.title(f"{plot_main_title_line}\n{plot_subtitle_line}")
    # === End: Modify plot title ===

    plt.legend(title='Entity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='both', linestyle='--', alpha=0.6) # Grid on both axes
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # === Start: Modify output filename based on focused_entities and specific_channels ===
    output_filename_prefix = "entity_distribution_plot"
    focus_suffix_for_filename = "all_entities" # Default if not focused
    if focused_entities:
        normalized_focused_entities_filename = sorted([e.lower().strip().replace(' ', '_') for e in focused_entities])
        focus_suffix_for_filename = "_".join(normalized_focused_entities_filename)
        output_filename_prefix += "_focused"
    
    channel_info_for_filename = f"top_{len(channels_to_plot_pd_index)}"
    if is_specific_channel_plot:
        # Maybe too long if many specific channels. Could use a hash or simply "specific_Nchannels"
        if len(channels_to_plot_pd_index) <= 5: # Arbitrary limit for listing channels in filename
            channel_info_for_filename = f"specific_channels_{'_'.join(map(str, channels_to_plot_pd_index.tolist()))}"
        else:
            channel_info_for_filename = f"specific_{len(channels_to_plot_pd_index)}_channels"
        
    output_filename = f"{output_filename_prefix}_{focus_suffix_for_filename}_layer_{layer_name}_{channel_info_for_filename}.png"
    output_png_path = os.path.join(results_dir, output_filename)
    # === End: Modify output filename ===
    
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
    parser.add_argument("--focused_entities", type=str, nargs='+', default=None,
                        help="Optional: Specify one or more entity names (e.g., 'blue_key' 'red_key') to focus on. If not set, considers all entities.")
    parser.add_argument("--specific_channels", type=int, nargs='+', default=None,
                        help="Optional: Specify one or more channel numbers to plot directly, bypassing top N selection.")

    args = parser.parse_args()
    main(args.results_dir, args.num_top_channels,
         dead_channels_csv=args.dead_channels_csv,
         run_detection_if_csv_missing=args.run_detection_if_csv_missing,
         exclude_dead_channels=args.exclude_dead_channels,
         sae_checkpoint_for_detection_arg=args.sae_checkpoint_for_detection_arg,
         model_checkpoint_for_detection=args.model_checkpoint_for_detection,
         detection_samples=args.detection_samples,
         detection_batch_size=args.detection_batch_size,
         detection_threshold=args.detection_threshold,
         focused_entities=args.focused_entities,
         specific_channels=args.specific_channels)

# %%