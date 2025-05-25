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
from src.utils.environment_modification_experiments import create_trident_maze, create_passages_box_maze

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
FEATURES_TO_ZERO = []
CHANNEL_TO_KEEP = None
max_kept_activation_this_trial = 0.0
sae = None
model = None
# ---

# --- Hook Definition ---
def ablation_hook(module, input, output):
    global CHANNEL_TO_KEEP, max_kept_activation_this_trial, sae, device
    if CHANNEL_TO_KEEP is None: return output # Skip if no channel specified

    with torch.no_grad():
        current_max = 0.0
        # Ensure output is a tensor and on the correct device
        # Handle potential tuples (common in base model layers)
        if isinstance(output, tuple):
             output_tensor = output[0].to(device)
        elif isinstance(output, torch.Tensor):
             output_tensor = output.to(device)
        else:
             print(f"Warning: Unexpected hook output type {type(output)}")
             return output # Cannot process

        # SAE Path
        if sae is not None:
            latent_acts = sae.encode(output_tensor)
            if not isinstance(latent_acts, torch.Tensor) or latent_acts.ndim < 2: return output # Basic check
            # Adapt indexing based on latent_acts dimensions (might be B, F or B, F, H, W)
            kept_latent_slice = latent_acts[:, CHANNEL_TO_KEEP]
            current_max = kept_latent_slice.max().item()
            modified_latent = torch.zeros_like(latent_acts)
            modified_latent[:, CHANNEL_TO_KEEP] = kept_latent_slice
            reconstructed = sae.decode(modified_latent)
            try: final_output = reconstructed.reshape_as(output_tensor) # Reshape based on tensor
            except RuntimeError: final_output = output # Fallback to original tuple/tensor
        # Base Path
        else:
            if not isinstance(output_tensor, torch.Tensor) or output_tensor.ndim < 2: return output # Basic check
            # Adapt indexing based on output_tensor dimensions
            kept_output_slice = output_tensor[:, CHANNEL_TO_KEEP]
            current_max = kept_output_slice.max().item()
            modified_output_tensor = torch.zeros_like(output_tensor)
            modified_output_tensor[:, CHANNEL_TO_KEEP] = kept_output_slice
            # If original output was tuple, return modified tensor in a tuple?
            # This might break subsequent layers if they expect a tuple.
            # Safest might be to modify the original tensor within the tuple if possible,
            # but hooks officially should return the same type structure.
            # For simplicity now, returning only the modified tensor.
            # Caution: This might cause downstream errors in the base model run.
            final_output = modified_output_tensor 

        max_kept_activation_this_trial = max(max_kept_activation_this_trial, current_max)
        # Return the potentially modified output structure
        if isinstance(output, tuple):
             # Attempt to return modified tensor within tuple structure
             if sae is not None: 
                 # Assuming SAE reconstruction replaces the first element
                 return (final_output,) + output[1:] 
             else: 
                 # Base model case: Replace the first element of the original tuple
                 # This assumes the main activation is the first element and preserves others
                 return (final_output,) + output[1:] 
        else:
             # If input was not a tuple, return the modified tensor
             return final_output
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
    parser.add_argument("--sae_checkpoint_path", type=str, default=None, help="Path to the SAE checkpoint (.pt file). If omitted, runs base model ablation.")
    parser.add_argument("--layer_spec", type=str, required=True, help="SAE layer number (int) OR base model layer name (str).")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials per channel.")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode.")
    parser.add_argument("--output_dir", type=str, default="channel_ablation_results", help="Directory to save results CSV.")
    parser.add_argument("--start_channel", type=int, default=0, help="Starting channel index (inclusive).")
    parser.add_argument("--end_channel", type=int, default=None, help="Ending channel index (exclusive). If None, runs all channels.")
    parser.add_argument("--save_gifs", action="store_true", help="Save a GIF for the first trial of each channel.")

    args = parser.parse_args()

    # Validation
    is_sae_run = args.sae_checkpoint_path is not None
    if is_sae_run:
        if not os.path.exists(args.sae_checkpoint_path):
            parser.error(f"SAE checkpoint path specified but not found: {args.sae_checkpoint_path}")
        try:
            layer_num_test = int(args.layer_spec)
            if layer_num_test not in ordered_layer_names:
                 parser.error(f"Invalid layer number for SAE: {args.layer_spec}")
        except ValueError:
            parser.error("If --sae_checkpoint_path is provided, --layer_spec must be an integer layer number.")
    else:
        if not isinstance(args.layer_spec, str):
             parser.error("If --sae_checkpoint_path is NOT provided, --layer_spec must be a string layer name.")
        # We'll check if layer_spec exists in the model later in main()

    os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    global FEATURES_TO_ZERO, CHANNEL_TO_KEEP, max_kept_activation_this_trial, sae, model, device

    args = parse_args()
    is_sae_run = args.sae_checkpoint_path is not None

    print(f"Starting ablation sweep for {'SAE layer' if is_sae_run else 'Base layer'}: {args.layer_spec}")

    # --- Load Model --- 
    print("Loading base model...", flush=True)
    model = helpers.load_interpretable_model(model_path=args.model_path).to(device)
    model.eval()
    # ---
    
    num_channels = -1
    layer_name = ""
    module_to_hook = None

    # --- Conditional Setup --- 
    if is_sae_run:
        print("Loading SAE model...", flush=True)
        sae = load_sae_from_checkpoint(args.sae_checkpoint_path).to(device)
        sae.eval()
        num_channels = sae.hidden_channels
        layer_name = ordered_layer_names[int(args.layer_spec)]
        print(f"SAE loaded. Target layer: {layer_name}, Channels: {num_channels}", flush=True)
        try:
            module_to_hook = get_module(model, layer_name)
        except Exception as e:
             print(f"Error getting module {layer_name} for SAE hook: {e}")
             return
    else: # Base model run
        sae = None # Ensure global SAE is None
        layer_name = args.layer_spec
        try:
            # Validate that the layer_spec exists as a module in the model
            module_to_hook = get_module(model, layer_name)
            
            # Attempt to determine num_channels from module
            if hasattr(module_to_hook, 'out_channels'):
                 num_channels = module_to_hook.out_channels
            elif hasattr(module_to_hook, 'weight') and module_to_hook.weight is not None and module_to_hook.weight.ndim >= 1:
                 num_channels = module_to_hook.weight.shape[0] # Usually for Conv layers
            elif hasattr(module_to_hook, 'out_features') and module_to_hook.out_features is not None:
                 num_channels = module_to_hook.out_features # Usually for Linear layers
            else:
                 # Try to get it from the output shape during a dummy forward pass? - More complex
                 # For now, rely on explicit attributes.
                 raise ValueError(f"Cannot automatically determine channel/feature count for base layer '{layer_name}'. Check layer attributes.")
            
            print(f"Base model layer target: '{layer_name}', Channels: {num_channels}", flush=True)
            
        except AttributeError as e:
            print(f"Error: Base layer specified '{layer_name}' not found as a direct attribute or nested module in the model.")
            print(f"Please check the layer name against the model definition (e.g., 'conv3a', 'fc1'). Error details: {e}")
            return
        except ValueError as e:
             print(f"Error determining channel count for base layer {layer_name}: {e}")
             return
        except Exception as e:
             print(f"Unexpected error getting module or channel count for base layer {layer_name}: {e}")
             return
    # ---

    # --- Create Original Venv Once ---
    print("Creating original box randomised maze environment...", flush=True)
    _, venv_original = create_passages_box_maze()
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
        print(f"Initial entities (randomised for box maze): {initial_entities}", flush=True)
    except Exception as e_init:
        print(f"ERROR: Could not determine initial entities from original venv: {e_init}", flush=True)
        venv_original.close() # Close if we can't proceed
        return # Exit if we can't get initial state
    # --- End Initial Entities ---

    # --- Hook Setup ---
    handle = module_to_hook.register_forward_hook(ablation_hook) 
    print(f"Hook registered on module: {layer_name}", flush=True)
    # ---

    # --- Setup for filenames ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_type = "sae" if is_sae_run else "base"
    safe_layer_id = args.layer_spec.replace('.', '_') if not is_sae_run else f"sae{args.layer_spec}"
    # Use more specific name if SAE path provided for SAE runs
    if is_sae_run:
         safe_name_part = os.path.basename(args.sae_checkpoint_path).replace('.pt', '')
    else:
         safe_name_part = layer_name.replace('.','_') # Use layer name for base model runs
    # ---

    start_channel = args.start_channel
    end_channel = args.end_channel if args.end_channel is not None else num_channels

    # --- Experiment Loops ---
    print(f"Iterating through channels {start_channel} to {end_channel-1}", flush=True)
    for channel_idx in range(start_channel, end_channel):
        CHANNEL_TO_KEEP = channel_idx # Set global for hook
        channel_results = []

        trial_iterator = tqdm(range(args.num_trials), desc=f"Trials Ch {CHANNEL_TO_KEEP}", leave=False)
        for trial_idx in trial_iterator:
            venv_trial = None 
            max_kept_activation_this_trial = 0.0 # Reset max activation tracker
            
            # Keep try-finally ONLY for closing the environment safely
            try: 
                # Copy the original venv for this trial
                obs, venv_trial = create_passages_box_maze()

                # 2. Run Simulation (Errors here will now propagate)
                save_this_gif = args.save_gifs and trial_idx == 0
                gif_dir = os.path.join(args.output_dir, f"gifs_ch_{CHANNEL_TO_KEEP}")
                if save_this_gif:
                    os.makedirs(gif_dir, exist_ok=True)
                gif_path = os.path.join(gif_dir, f"trial_{trial_idx}_box.gif") if save_this_gif else None

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
                    final_env_state = EnvState(last_state_bytes)

                    # # # --- Gem Logic --- 
                    # if final_env_state.count_entities(ENTITY_TYPES["gem"]) > 0:
                    #     remaining_entities.add(GEM)

                    # --- Key Logic (Remains if count is 2) ---
                    for color_idx, entity_name in COLOR_IDX_TO_ENTITY_NAME.items():
                        key_count = final_env_state.count_entities(ENTITY_TYPES["key"], color_idx)
                        if key_count == 2:
                            remaining_entities.add(entity_name)
                            
                print(f"Ended by Gem: {ended_by_gem}")
                collected_entities = initial_entities - remaining_entities
                if ended_by_gem:
                    collected_entities.add(GEM)
                else:
                    collected_entities.remove(GEM)
                    remaining_entities.add(GEM)
                print(f"remaining_entities {remaining_entities}, collected_entities {collected_entities}")

                # if ended_by_gem and GEM in initial_entities:
                #      collected_entities.add(GEM)

                # 5. Record Result
                channel_results.append({
                    "channel_kept": CHANNEL_TO_KEEP, 
                    "trial": trial_idx,
                    "maze_type": "box",
                    "initial_entities": set_to_string(initial_entities),
                    "remaining_entities": set_to_string(remaining_entities),
                    "collected_entities": set_to_string(collected_entities),
                    "max_kept_activation": max_kept_activation_this_trial,
                    "total_reward": total_reward,
                    "ended_by_gem": ended_by_gem,
                    "ended_by_timeout": ended_by_timeout,
                    "steps_taken": len(frames) if save_this_gif else -1
                })

            finally:
                # Ensure environment is closed even if an error occurred mid-trial
                if venv_trial:
                    venv_trial.close()

        # --- Save results per channel --- (Errors here will now propagate)
        if channel_results:
            channel_df = pd.DataFrame(channel_results)
            channel_csv_filename = os.path.join(args.output_dir, f"results_{run_type}_{safe_name_part}_ch{CHANNEL_TO_KEEP}_{timestamp}.csv")
            # Removed try-except around saving
            channel_df.to_csv(channel_csv_filename, index=False)
            
        # ---

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
