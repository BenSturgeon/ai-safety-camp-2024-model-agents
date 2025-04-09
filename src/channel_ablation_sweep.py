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
from src.utils.environment_modification_experiments import create_trident_maze

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
sae = None
model = None
# ---

# --- Hook Definition ---
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
             # print(f"Warning: Could not reshape reconstructed output. Shapes: Recon={reconstructed_output.shape}, Original={output.shape}. Error: {e}")
             return output # Return original if reshape fails

    return reconstructed_output
# --- End Hook Definition ---


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
    parser.add_argument("--sae_checkpoint_path", type=str, required=True, help="Path to the SAE checkpoint (.pt file).")
    parser.add_argument("--layer_number", type=int, required=True, help="SAE layer number (e.g., 8 for conv4a). Check sae_cnn.py.")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials per channel.")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode.")
    parser.add_argument("--output_dir", type=str, default="channel_ablation_results", help="Directory to save results CSV.")
    parser.add_argument("--start_channel", type=int, default=0, help="Starting channel index (inclusive).")
    parser.add_argument("--end_channel", type=int, default=None, help="Ending channel index (exclusive). If None, runs all channels.")
    parser.add_argument("--save_gifs", action="store_true", help="Save a GIF for the first trial of each channel.")

    args = parser.parse_args()


    if not os.path.exists(args.model_path):
         parser.error(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.sae_checkpoint_path):
         parser.error(f"SAE checkpoint path not found: {args.sae_checkpoint_path}")
    if args.layer_number not in ordered_layer_names:
         parser.error(f"Invalid layer number: {args.layer_number}. Valid options are: {list(ordered_layer_names.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    global FEATURES_TO_ZERO, sae, model, device # Declare modification of globals

    args = parse_args()
    layer_name = ordered_layer_names[args.layer_number]
    print(f"Starting ablation sweep for Layer {args.layer_number} ({layer_name}) using SAE: {args.sae_checkpoint_path}", flush=True)

    # --- Load Model and SAE ---
    print("Loading base model...", flush=True)
    model = helpers.load_interpretable_model(model_path=args.model_path).to(device)
    model.eval()
    print("Loading SAE model...", flush=True)
    sae = load_sae_from_checkpoint(args.sae_checkpoint_path).to(device)
    sae.eval()
    num_channels = sae.hidden_channels
    print(f"SAE loaded. Number of channels: {num_channels}", flush=True)
    # --- End Load ---

    # --- Create Original Venv Once ---
    print("Creating original trident maze environment...", flush=True)
    _, venv_original = create_trident_maze()
    print("Original environment created.", flush=True)
    # --- Get Initial Entities Once ---
    print("Determining initial entities...", flush=True)
    initial_entities = set()
    try:
        initial_state_bytes = venv_original.env.callmethod("get_state")[0]
        initial_env_state = EnvState(initial_state_bytes)
        if initial_env_state.entity_exists(ENTITY_TYPES["gem"]):
            initial_entities.add(GEM)
        for color_idx, entity_name in COLOR_IDX_TO_ENTITY_NAME.items():
            if initial_env_state.entity_exists(ENTITY_TYPES["key"], color_idx):
                    initial_entities.add(entity_name)
        print(f"Initial entities (fixed for trident maze): {initial_entities}", flush=True)
    except Exception as e_init:
        print(f"ERROR: Could not determine initial entities from original venv: {e_init}", flush=True)
        venv_original.close() # Close if we can't proceed
        return # Exit if we can't get initial state
    # --- End Initial Entities ---

    # --- Hook Setup ---
    module_to_hook = get_module(model, layer_name)
    handle = module_to_hook.register_forward_hook(hook_sae_activations)
    print(f"Hook registered on module: {layer_name}", flush=True)
    # --- End Hook Setup ---

    # --- Setup for filenames ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Single timestamp for the run
    safe_sae_name = os.path.basename(args.sae_checkpoint_path).replace('.pt', '')
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

            try:
                # Copy the original venv for this trial
                venv_trial = copy_venv(venv_original, 0)

                # 2. Run Simulation
                save_this_gif = args.save_gifs and trial_idx == 0
                gif_dir = os.path.join(args.output_dir, f"gifs_ch_{channel_to_keep}")
                if save_this_gif:
                    os.makedirs(gif_dir, exist_ok=True)
                gif_path = os.path.join(gif_dir, f"trial_{trial_idx}_trident.gif") if save_this_gif else None

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
                    "maze_type": "trident",
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
                    "maze_type": "trident",
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
            channel_csv_filename = os.path.join(args.output_dir, f"results_layer{args.layer_number}_{safe_sae_name}_ch{channel_to_keep}_{timestamp}.csv")
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
    if venv_original:
        venv_original.close()
        print("Original environment closed.", flush=True)
    # --- End Cleanup ---

    # --- Final Message (No overall saving needed anymore) ---
    print("\nExperiment finished. Per-channel results saved in:", args.output_dir, flush=True)
    print("You will need to combine the CSV files from that directory for overall analysis.", flush=True)
    # --- End Final Message ---

if __name__ == "__main__":
    main()

# %%
