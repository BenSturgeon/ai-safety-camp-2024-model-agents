# %%
import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import random
import copy
import sys

import utils.helpers as helpers
from sae_cnn import load_sae_from_checkpoint, ordered_layer_names
# Removed unused imageio and plt imports for now
# import imageio
# import matplotlib.pyplot as plt

# Keep environment creation if needed later, but heist.create_venv used now
from src.utils.create_intervention_mazes import create_fork_maze, create_corners_maze
from src.create_bias_corrected_fork_maze import create_bias_corrected_fork_maze, create_bias_corrected_corners_maze

from utils.helpers import run_episode_and_get_final_state
from utils.heist import (
    EnvState,
    ENTITY_TYPES,
    KEY_COLORS,
    create_venv, # Import create_venv directly
    copy_venv
)

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Entity Constants ---
GEM = 'gem'
KEY_BLUE = 'blue_key'
KEY_GREEN = 'green_key'
KEY_RED = 'red_key'
COLOR_IDX_TO_ENTITY_NAME = {
    0: KEY_BLUE,
    1: KEY_GREEN,
    2: KEY_RED,
}
# --- End Entity Constants ---


# --- Global variable for hook ---
# This will be updated in the main loop for each channel
FEATURES_TO_ZERO = []
# Need global references to sae and device for the hook
sae = None # sae can be None if it's a base model ablation
model = None
# ---

# --- Hook Definition for SAE ---
# Define activation hook (needs access to global FEATURES_TO_ZERO, sae, device)
def hook_sae_activations(module, input, output):
    global FEATURES_TO_ZERO, sae, device # Access globals

    # Ensure SAE is loaded and on the correct device within the hook if needed
    # This might be redundant if they are set globally before hook registration,
    # but ensures safety if the hook is called before main setup is complete.
    if sae is None:
        print("Warning: SAE not loaded when hook called.")
        return output
    sae.to(device)

    with torch.no_grad():
        output_tensor = output.to(device)

        # Encode to get latent activations
        latent_acts = sae.encode(output_tensor) # Shape (batch, features, H', W')

        # --- Zero out specified features ---
        modified_acts = latent_acts
        if FEATURES_TO_ZERO:
            modified_acts = latent_acts.clone()
            num_features = modified_acts.shape[1]
            # Ensure FEATURES_TO_ZERO contains valid indices for this SAE
            valid_indices = [idx for idx in FEATURES_TO_ZERO if 0 <= idx < num_features]
            if valid_indices:
                 modified_acts[:, valid_indices, :, :] = 0.0
            # Avoid excessive printing in the loop
            # if len(valid_indices) != len(FEATURES_TO_ZERO):
            #      print(f"Warning: Invalid/out-of-range feature indices encountered.")

        # Decode using the potentially modified activations
        reconstructed_output = sae.decode(modified_acts)

        try:
            reconstructed_output = reconstructed_output.reshape_as(output)
        except RuntimeError as e:
             # Avoid printing error for every step, maybe log frequency?
             print(f"Warning: Could not reshape reconstructed output. Shapes: Recon={reconstructed_output.shape}, Original={output.shape}. Error: {e}") # DEBUG: Log this error
             return output # Return original if reshape fails

    return reconstructed_output
# --- End Hook Definition for SAE ---

# --- New Hook Definition for Base Model Channel Ablation ---
def hook_base_channel_ablation(module, input, output):
    global FEATURES_TO_ZERO, device # Access globals

    with torch.no_grad():
        output_tensor = output.to(device)
        modified_output = output_tensor.clone()

        if FEATURES_TO_ZERO:
            num_output_channels = modified_output.shape[1] # Assumes B, C, H, W
            # Ensure FEATURES_TO_ZERO contains valid indices
            valid_indices = [idx for idx in FEATURES_TO_ZERO if 0 <= idx < num_output_channels]
            if valid_indices:
                modified_output[:, valid_indices, :, :] = 0.0
            # Optional: Add warning for invalid indices
            # if len(valid_indices) != len(FEATURES_TO_ZERO):
            #      print(f"Warning: Invalid/out-of-range feature indices encountered for base layer ablation.")

        # No decoding needed, return modified activations directly
        return modified_output
# --- End New Hook Definition ---


# --- Helper Functions ---
def get_module(model_to_search, layer_name):
    """Helper to get a specific module/layer from the model by its name."""
    module = model_to_search
    for element in layer_name.split("."):
        if "[" in element:
            base, idx = element.rstrip("]").split("[")
            module = getattr(module, base)[int(idx)]
        else:
            module = getattr(module, element)
    return module

def set_to_string(entity_set):
    """Converts a set of entity strings to a sorted, comma-separated string."""
    if not entity_set:
        return ""
    return ",".join(sorted(list(entity_set)))
# --- End Helper Functions ---


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run channel ablation sweep experiment.")
    parser.add_argument("--model_path", type=str, default="../model_interpretable.pt", help="Path to the base interpretable model.")
    parser.add_argument("--sae_checkpoint_path", type=str, default=None, required=False, help="Path to the SAE checkpoint (.pt file). If None, runs base model layer ablation.")
    parser.add_argument("--layer_spec", type=str, required=True, help="SAE layer number (e.g., '8') or layer name (e.g., 'conv4a') if SAE run. Direct model layer name (e.g., 'encoder.conv0') if base model run.")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials per channel.")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode.")
    parser.add_argument("--output_dir", type=str, default="channel_ablation_results", help="Directory to save results CSV.")
    parser.add_argument("--start_channel", type=int, default=0, help="Starting channel index (inclusive).")
    parser.add_argument("--end_channel", type=int, default=None, help="Ending channel index (exclusive). If None, runs all channels specified by SAE or total_channels_for_base_layer.")
    parser.add_argument("--no_save_gifs", action="store_true", help="Disable saving a GIF for the first trial of each channel.")
    parser.add_argument("--total_channels_for_base_layer", type=int, default=None, help="Total channels in the base model layer, REQUIRED if not an SAE run (i.e., sae_checkpoint_path is not provided).")
    parser.add_argument("--maze_type", type=str, default="fork", choices=["fork", "corners"], help="Type of maze to use for the experiment.")
    parser.add_argument("--bias_direction", type=str, default=None, choices=["up", "down", "left", "right"], help="Bias direction for maze orientation (optional).")

    args = parser.parse_args()

    args.is_sae_run = args.sae_checkpoint_path is not None

    if not os.path.exists(args.model_path):
         parser.error(f"Model path not found: {args.model_path}")
    if args.is_sae_run and not os.path.exists(args.sae_checkpoint_path):
         parser.error(f"SAE checkpoint path specified but not found: {args.sae_checkpoint_path}")
    if not args.is_sae_run and args.total_channels_for_base_layer is None:
        parser.error("--total_channels_for_base_layer is required when --sae_checkpoint_path is not provided.")
    if not args.is_sae_run and args.total_channels_for_base_layer is not None and args.total_channels_for_base_layer <= 0:
        parser.error("--total_channels_for_base_layer must be positive.")

    # --- Resolve layer_spec ---
    actual_layer_number = None # Primarily for SAE runs for naming/reference
    actual_layer_name = None
    spec_input = args.layer_spec

    try:
        # Try to interpret layer_spec as an integer (layer number)
        num_spec = int(spec_input)
        if num_spec in ordered_layer_names:
            actual_layer_number = num_spec
            actual_layer_name = ordered_layer_names[num_spec]
        else:
            parser.error(f"Invalid layer number specified: {num_spec}. Valid integer options are: {list(ordered_layer_names.keys())}")
    except ValueError:
        # Interpret layer_spec as a string (layer name)
        if spec_input in ordered_layer_names.values():
            for num, name_val in ordered_layer_names.items():
                if name_val == spec_input:
                    actual_layer_number = num
                    actual_layer_name = name_val
                    break
        else:
            parser.error(f"Invalid layer name specified: '{spec_input}'. Valid string options are: {list(ordered_layer_names.values())}")

    if actual_layer_name is None: # Fallback, should be set by logic above.
        parser.error(f"Could not resolve layer_spec '{spec_input}' to a layer name.")

    args.actual_layer_number = actual_layer_number # This will be None for base model runs
    args.actual_layer_name = actual_layer_name     # This will be the target layer name for get_module
    # --- End Resolve layer_spec ---

    os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    global FEATURES_TO_ZERO, sae, model, device # Declare modification of globals

    args = parse_args()
    layer_name = args.actual_layer_name # This is the name used by get_module

    should_save_gifs_for_first_trial = not args.no_save_gifs

    run_description = ""
    if args.is_sae_run:
        # args.actual_layer_number should be set if is_sae_run due to parse_args logic
        run_description = f"SAE Ablation for Layer {args.actual_layer_number} ({layer_name}) using SAE: {args.sae_checkpoint_path}"
    else:
        run_description = f"Base Model Channel Ablation for Layer {layer_name}"
    print(f"Starting: {run_description}", flush=True)

    if should_save_gifs_for_first_trial:
        print("GIF saving for the first trial of each channel is ENABLED.", flush=True)
    else:
        print("GIF saving for the first trial of each channel is DISABLED (due to --no_save_gifs flag).", flush=True)

    # --- Load Model and SAE (conditionally) ---
    print("Loading base model...", flush=True)
    model = helpers.load_interpretable_model(model_path=args.model_path).to(device)
    model.eval()

    num_channels = 0
    if args.is_sae_run:
        print("Loading SAE model...", flush=True)
        # sae_checkpoint_path is guaranteed to exist by parse_args if is_sae_run
        sae = load_sae_from_checkpoint(args.sae_checkpoint_path).to(device)
        sae.eval()
        num_channels = sae.hidden_channels
        print(f"SAE loaded. Number of channels: {num_channels}", flush=True)
    else: # Base model run
        # sae global remains None
        # total_channels_for_base_layer is guaranteed to be set by parse_args
        num_channels = args.total_channels_for_base_layer
        print(f"Base model ablation configured for layer '{layer_name}' with {num_channels} channels.", flush=True)
    # --- End Load ---

    # --- Hook Setup ---
    module_to_hook = get_module(model, layer_name)
    handle = None
    if args.is_sae_run:
        if sae is None: # Should not happen if is_sae_run and loading succeeded
             raise RuntimeError("SAE is None during an SAE run setup, check SAE loading.")
        handle = module_to_hook.register_forward_hook(hook_sae_activations)
        print(f"SAE Hook registered on module: {layer_name}", flush=True)
    else: # Base model run
        handle = module_to_hook.register_forward_hook(hook_base_channel_ablation)
        print(f"Base Model Channel Ablation Hook registered on module: {layer_name}", flush=True)
    # --- End Hook Setup ---

    # --- Setup for filenames ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Single timestamp for the run
    safe_name_part = ""
    if args.is_sae_run:
        # args.actual_layer_number should be valid if is_sae_run
        safe_sae_name_suffix = os.path.basename(args.sae_checkpoint_path).replace('.pt', '')
        safe_name_part = f"sae_layer{args.actual_layer_number}_{safe_sae_name_suffix}"
    else: # Base model run
        safe_name_part = f"baselayer_{layer_name.replace('.', '_')}" # Use actual layer_name
    # ---

    start_channel = args.start_channel
    end_channel = args.end_channel if args.end_channel is not None else num_channels

    # --- Experiment Loops ---
    print(f"Iterating through channels {start_channel} to {end_channel-1}", flush=True)
    for channel_to_keep in range(start_channel, end_channel):
        FEATURES_TO_ZERO = [i for i in range(num_channels) if i != channel_to_keep]
        channel_results = []

        trial_iterator = tqdm(range(args.num_trials), desc=f"Trials Ch {channel_to_keep}", leave=False)
        for trial_idx in trial_iterator:
            venv_trial = None # Ensure venv_trial is defined for finally block
            initial_entities = set() # Initialize for each trial

            try:
                # --- Create Venv and Get Initial Entities FOR EACH TRIAL ---
                # print(f"Creating new fork maze environment for Ch {channel_to_keep}, Trial {trial_idx}...", flush=True) # Optional debug
                if args.maze_type == "fork":
                    if args.bias_direction:
                        _, venv_trial = create_bias_corrected_fork_maze(bias_direction=args.bias_direction)
                    else:
                        _, venv_trial = create_fork_maze() # Create fresh maze for each trial
                elif args.maze_type == "corners":
                    if args.bias_direction:
                        _, venv_trial = create_bias_corrected_corners_maze(bias_direction=args.bias_direction)
                    else:
                        _, venv_trial = create_corners_maze() # Create fresh maze for each trial
                else:
                    raise ValueError(f"Invalid maze_type: {args.maze_type}")
                # print("Trial environment created.", flush=True) # Optional debug

                # print("Determining initial entities for this trial...", flush=True) # Optional debug
                initial_state_bytes = venv_trial.env.callmethod("get_state")[0]
                initial_env_state = EnvState(initial_state_bytes)
                if initial_env_state.entity_exists(ENTITY_TYPES["gem"]):
                    initial_entities.add(GEM)
                for color_idx, entity_name_local in COLOR_IDX_TO_ENTITY_NAME.items(): # Use a different variable name to avoid conflict
                    if initial_env_state.entity_exists(ENTITY_TYPES["key"], color_idx):
                            initial_entities.add(entity_name_local)
                # print(f"Initial entities for this trial: {initial_entities}", flush=True) # Optional debug
                # --- End Venv and Initial Entities for Trial ---

                # 2. Run Simulation
                save_this_gif = should_save_gifs_for_first_trial and trial_idx == 0
                gif_dir = os.path.join(args.output_dir, f"gifs_ch_{channel_to_keep}")
                if save_this_gif:
                    os.makedirs(gif_dir, exist_ok=True)
                gif_path = os.path.join(gif_dir, f"trial_{trial_idx}_fork.gif") if save_this_gif else None

                total_reward, frames, last_state_bytes, ended_by_gem, ended_by_timeout = run_episode_and_get_final_state(
                    venv=venv_trial,
                    model=model,
                    filepath=gif_path,
                    save_gif=save_this_gif,
                    episode_timeout=args.max_steps,
                    is_procgen_env=True
                )

                # 3. Calculate Remaining Entities (Simplified)
                remaining_entities = set() 
                if last_state_bytes:
                    try:
                        final_env_state = EnvState(last_state_bytes)

                        # --- Gem Logic --- 
                        if final_env_state.count_entities(ENTITY_TYPES["gem"]) > 0:
                            remaining_entities.add(GEM)

                        # --- Key Logic (Remains if count is 2) ---
                        for color_idx, entity_name in COLOR_IDX_TO_ENTITY_NAME.items():
                            key_count = final_env_state.count_entities(ENTITY_TYPES["key"], color_idx)
                            if key_count == 2:
                                remaining_entities.add(entity_name)
                                
                    except Exception as e_state:
                        print(f"Warning: Error parsing final state for Ch {channel_to_keep}, Trial {trial_idx}: {e_state}")

                collected_entities = initial_entities - remaining_entities

                if not ended_by_gem and GEM in collected_entities:
                     collected_entities.remove(GEM)
                elif ended_by_gem and GEM in initial_entities:
                     collected_entities.add(GEM)

                # 5. Record Result for this trial/channel
                channel_results.append({
                    "channel_kept": channel_to_keep,
                    "trial": trial_idx,
                    "maze_type": args.maze_type,
                    "initial_entities": set_to_string(initial_entities),
                    "remaining_entities": set_to_string(remaining_entities), # Save the simplified remaining set
                    "collected_entities": set_to_string(collected_entities),   # Save the robust collected set
                    "total_reward": total_reward,
                    "ended_by_gem": ended_by_gem,
                    "ended_by_timeout": ended_by_timeout,
                    "steps_taken": len(frames) if save_this_gif else -1
                })

            except Exception as e_trial:
                print(f"ERROR during Ch {channel_to_keep}, Trial {trial_idx}: {e_trial}")
                # Record error state for this trial/channel
                channel_results.append({
                    "channel_kept": channel_to_keep,
                    "trial": trial_idx,
                    "maze_type": args.maze_type,
                    "initial_entities": "ERROR",
                    "remaining_entities": "ERROR", # <<< Update error reporting
                    "collected_entities": "ERROR", # <<< Update error reporting
                    "total_reward": np.nan,
                    "ended_by_gem": False,
                    "ended_by_timeout": False,
                    "steps_taken": -1
                 })
            finally:
                if venv_trial:
                    venv_trial.close()

        # --- Save results for THIS completed channel ---
        if channel_results:
            channel_df = pd.DataFrame(channel_results)
            # Create a filename specific to this channel
            channel_csv_filename = os.path.join(args.output_dir, f"results_{safe_name_part}_ch{channel_to_keep}_{timestamp}.csv")
            try:
                channel_df.to_csv(channel_csv_filename, index=False)
                # Optionally print confirmation, but might clutter tqdm output
                # tqdm.write(f"Saved results for channel {channel_to_keep} to {channel_csv_filename}")
            except Exception as e_csv:
                tqdm.write(f"\nError saving results for channel {channel_to_keep} to CSV: {e_csv}")
        # --- End saving for channel ---

    # --- End Loops ---

    # --- Cleanup ---
    handle.remove()
    print("Hook removed.", flush=True)
    # venv_original is no longer used globally
    # if venv_original:
    #     venv_original.close()
    #     print("Original environment closed.", flush=True)
    # --- End Cleanup ---

    # --- Final Message (No overall saving needed anymore) ---
    print("\nExperiment finished. Per-channel results saved in:", args.output_dir, flush=True)
    print("You will need to combine the CSV files from that directory for overall analysis.", flush=True)
    # --- End Final Message ---

if __name__ == "__main__":
    main()

# %%