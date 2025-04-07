import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import glob
import re
from tqdm import tqdm
import datetime
import json # For saving list args
import imageio # Add imageio import
from PIL import Image # Ensure PIL is imported for resizing

# Project imports
from base_model_intervention import BaseModelInterventionExperiment
from sae_spatial_intervention import SAEInterventionExperiment, ordered_layer_names as sae_ordered_layer_names
from utils.environment_modification_experiments import create_multi_entity_no_locks_maze, create_trident_maze
from utils import helpers, heist # Import helpers to use its functions
from zero_ablation_classes import BaseModelZeroAblationExperiment, SAEZeroAblationExperiment # Import new classes

# Constants
ENTITY_CODE_DESCRIPTION = { 3: "gem", 4: "blue_key", 5: "green_key", 6: "red_key", 7: "blue_lock", 8: "green_lock", 9: "red_lock" }
ENTITY_DESCRIPTION_CODE = {v: k for k, v in ENTITY_CODE_DESCRIPTION.items()}
DEFAULT_MODEL_BASE_DIR = "models"
DEFAULT_SAE_CHECKPOINT_DIR = "checkpoints"
GEM_CODE = 3
KEY_CODES = {4, 5, 6} # Set of key codes

# --- Helper Functions (Copied/Adapted from helpers.py and quantitative_intervention_experiment.py) ---

def find_latest_checkpoint(layer_number, layer_name, base_dir="checkpoints"):
    """Finds the latest SAE checkpoint file for a given layer."""
    layer_dir = os.path.join(base_dir, f"layer_{layer_number}_{layer_name}")
    if not os.path.isdir(layer_dir):
        return None

    latest_step = -1
    latest_file = None
    # Search for files matching the pattern checkpoint_STEPNUM.pt
    pattern = os.path.join(layer_dir, "checkpoint_*.pt")
    for f in glob.glob(pattern):
        match = re.search(r"checkpoint_(\d+)\.pt", os.path.basename(f))
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_file = f
    return latest_file

# --- End Helper Functions ---

# --- Modify Helper Function for Plotting Activations to Image ---
def plot_activations_to_image(activation_tensor, title="SAE Activations", channels_to_plot=None):
    """
    Plots specified SAE activation channels and returns as a NumPy image array.
    If channels_to_plot is None, plots all channels (original behavior).
    """
    # Ensure tensor is on CPU and detached
    act_tensor_cpu = activation_tensor.detach().cpu()
    # Remove batch dimension if present, assume (Features, H, W)
    plot_tensor_full = act_tensor_cpu.squeeze(0) if act_tensor_cpu.ndim > 3 else act_tensor_cpu

    # --- Select specific channels if requested ---
    plot_tensor_selected = plot_tensor_full
    num_total_features = plot_tensor_full.shape[0]
    channels_actually_plotted = list(range(num_total_features)) # Default to all

    if channels_to_plot is not None:
        # Filter out invalid indices just in case
        valid_channels = [c for c in channels_to_plot if 0 <= c < num_total_features]
        if valid_channels:
            plot_tensor_selected = plot_tensor_full[valid_channels, :, :]
            channels_actually_plotted = valid_channels
            title += f" (Showing Ablated Features: {channels_actually_plotted})"
        else:
            # If no valid channels provided, maybe plot nothing or fallback? Fallback for now.
            print(f"Warning: No valid channels provided in channels_to_plot={channels_to_plot}. Plotting all.")
            title += " (Plotting All Features - No valid ablated channels specified)"
    # --- End channel selection ---

    # Use a temporary dict key to avoid potential conflicts if layer name is 'activations'
    plot_dict_key = "selected_activations"
    fig = helpers.plot_layer_channels(
        {plot_dict_key: plot_tensor_selected},
        plot_dict_key,
        return_image=True # Get the figure object
    )
    if fig is None:
        print("Warning: plot_layer_channels returned None.")
        return None

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(height, width, 3)
    plt.close(fig) # Close the figure to free memory
    return image_array
# --- End Helper Function Modification ---

def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-ablation intervention experiments.")

    # Model/Layer Config
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the base model checkpoint (.pt file). If None, searches in --model_base_dir.")
    parser.add_argument("--model_base_dir", type=str, default=DEFAULT_MODEL_BASE_DIR,
                        help="Base directory to search for the latest model checkpoint if --model_path is not specified.")
    parser.add_argument("--sae_checkpoint_path", type=str, default=None,
                        help="Path to the SAE checkpoint (.pt file). If None and --is_sae, searches in --sae_checkpoint_dir.")
    parser.add_argument("--sae_checkpoint_dir", type=str, default=DEFAULT_SAE_CHECKPOINT_DIR,
                        help="Base directory to search for the latest SAE checkpoint if --sae_checkpoint_path is not specified.")
    parser.add_argument("--layer_spec", type=str, required=True,
                        help="Layer specification. For base model: layer name (e.g., 'conv_seqs.2.res_block1.conv1'). For SAE: layer number (e.g., '8').")
    parser.add_argument("--is_sae", action="store_true",
                        help="Indicates that the intervention targets an SAE layer specified by number in --layer_spec.")

    # Experiment Parameters
    parser.add_argument("--required_entities", type=str, default="blue_key,green_key,red_key,gem",
                        help="Comma-separated list of entity names that MUST be present (e.g., 'blue_key,green_key,red_key,gem')")
    parser.add_argument("--zero_target_entity", type=str, required=True,
                        help="The name of the entity used for success criteria (e.g., 'blue_key') - determines which key *not* to collect.")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials per channel/group")
    parser.add_argument("--max_steps", type=int, default=250,
                        help="Maximum steps per trial")

    # --- Ablation Specification Group ---
    ablation_group = parser.add_mutually_exclusive_group(required=True)
    ablation_group.add_argument("--start_channel", type=int, default=argparse.SUPPRESS, # Use SUPPRESS to check presence later
                        help="Ablate channels/features individually, starting from this index (runs up to num_channels/features).")
    ablation_group.add_argument("--ablation_groups", type=str, default=argparse.SUPPRESS,
                        help="Ablate groups of channels/features simultaneously. Provide a JSON string representing a list of lists, e.g., '[[1, 5, 10], [22, 23], [4]]'.")
    # --- End Ablation Specification Group ---

    # Output Config
    parser.add_argument("--output_dir", type=str, default="zero_ablation_results",
                        help="Directory to save results (CSV and plots)")
    parser.add_argument("--num_top_channels_visualize", type=int, default=20,
                         help="Number of top channels to include in the result visualization")

    # --- Ensure this argument is present ---
    parser.add_argument("--visualize_sae_activations", action="store_true",
                        help="If set and --is_sae, visualize the SAE activations for the first trial of the first ablated group/channel.")
    # ---

    args = parser.parse_args()

    # --- Argument Processing and Validation ---
    # Process required entities
    args.required_entity_codes = []
    for name in args.required_entities.split(','):
        name = name.strip().lower() # Ensure lowercase
        if name in ENTITY_DESCRIPTION_CODE:
            args.required_entity_codes.append(ENTITY_DESCRIPTION_CODE[name])
        else:
            parser.error(f"Invalid entity name in --required_entities: {name}. Valid names: {list(ENTITY_DESCRIPTION_CODE.keys())}")
    if not args.required_entity_codes:
        parser.error("No valid required entities specified.")
    args.required_entity_codes = sorted(list(set(args.required_entity_codes))) # Ensure unique and sorted

    # Process zero target entity
    args.zero_target_entity_name = args.zero_target_entity.strip().lower() # Ensure lowercase
    if args.zero_target_entity_name not in ENTITY_DESCRIPTION_CODE:
        parser.error(f"Invalid entity name for --zero_target_entity: {args.zero_target_entity_name}. Valid names: {list(ENTITY_DESCRIPTION_CODE.keys())}")
    args.zero_target_entity_code = ENTITY_DESCRIPTION_CODE[args.zero_target_entity_name]
    if args.zero_target_entity_code not in args.required_entity_codes:
        parser.error(f"--zero_target_entity '{args.zero_target_entity_name}' must be included in --required_entities.")
    if args.zero_target_entity_code == GEM_CODE:
         parser.error("--zero_target_entity cannot be 'gem'. It must be one of the keys.")

    # Determine other keys required for success calculation
    args.other_required_key_codes = sorted(list(
        (set(args.required_entity_codes) & KEY_CODES) - {args.zero_target_entity_code}
    ))
    if not args.other_required_key_codes:
         print("Warning: No 'other' keys specified in required_entities besides the zero_target_entity.")

    # Determine Model Path
    if not args.model_path:
        print(f"--model_path not specified, searching for latest model checkpoint in '{args.model_base_dir}'...")
        args.model_path = helpers.find_latest_model_checkpoint(base_dir=args.model_base_dir)
        if not args.model_path:
            parser.error(f"Could not automatically find a model checkpoint in '{args.model_base_dir}' or its subdirectories. Please specify the path using --model_path.")
        print(f"Found latest model checkpoint: {args.model_path}")
    elif not os.path.exists(args.model_path):
        parser.error(f"Specified model path not found: {args.model_path}")

    # Layer and Channel Setup
    args.layer_number = None
    args.layer_name = None

    if args.is_sae:
        try:
            args.layer_number = int(args.layer_spec)
        except ValueError:
            parser.error(f"Invalid layer_spec for SAE: '{args.layer_spec}'. Must be an integer layer number.")

        args.layer_name = sae_ordered_layer_names.get(args.layer_number)
        if not args.layer_name:
             parser.error(f"Invalid SAE layer number: {args.layer_number}. Check sae_spatial_intervention.ordered_layer_names.")

        # Auto-detect SAE checkpoint if path not provided
        if not args.sae_checkpoint_path:
            print(f"--sae_checkpoint_path not specified, searching for latest checkpoint for layer {args.layer_number} ({args.layer_name}) in '{args.sae_checkpoint_dir}'...")
            args.sae_checkpoint_path = find_latest_checkpoint(args.layer_number, args.layer_name, base_dir=args.sae_checkpoint_dir)
            if not args.sae_checkpoint_path:
                expected_dir = os.path.join(args.sae_checkpoint_dir, f"layer_{args.layer_number}_{args.layer_name}")
                parser.error(f"Could not automatically find SAE checkpoint for layer {args.layer_number} ({args.layer_name}) in directory structure '{expected_dir}/'. Please specify the path using --sae_checkpoint_path.")
            print(f"Found latest checkpoint: {args.sae_checkpoint_path}")
        elif not os.path.exists(args.sae_checkpoint_path):
             parser.error(f"Specified SAE checkpoint path not found: {args.sae_checkpoint_path}")

    else: # Base model
        args.layer_name = args.layer_spec

    # --- Process Ablation Specification ---
    args.ablation_mode = None
    args.indices_to_ablate = [] # Will store individual indices or groups (lists)

    if hasattr(args, 'start_channel'):
        args.ablation_mode = 'individual'
        # We'll generate the individual indices later, after getting num_channels/features
        print(f"Ablation Mode: Individual channels starting from index {args.start_channel}")
    elif hasattr(args, 'ablation_groups'):
        args.ablation_mode = 'group'
        try:
            parsed_groups = json.loads(args.ablation_groups)
            if not isinstance(parsed_groups, list) or not all(isinstance(g, list) for g in parsed_groups):
                raise ValueError("Input must be a list of lists.")
            if not all(isinstance(i, int) for g in parsed_groups for i in g):
                 raise ValueError("All elements within the inner lists must be integers.")
            args.indices_to_ablate = parsed_groups
            print(f"Ablation Mode: Group ablation with {len(args.indices_to_ablate)} groups: {args.indices_to_ablate}")
            if not args.indices_to_ablate:
                 parser.error("--ablation_groups cannot be an empty list '[]'.")
        except json.JSONDecodeError:
            parser.error("--ablation_groups must be a valid JSON string representing a list of lists.")
        except ValueError as e:
             parser.error(f"Invalid format for --ablation_groups: {e}")
    else:
        # Should not happen due to mutually_exclusive_group(required=True)
        parser.error("Either --start_channel or --ablation_groups must be specified.")
    # --- End Process Ablation Specification ---

    # --- End Argument Processing ---

    return args

def run_ablation_simulation(experiment, venv, max_steps, zero_target_entity_code, required_entity_codes, current_trial_number):
    """
    Runs a single zero-ablation simulation episode using the provided experiment object.
    The experiment object handles the intervention hook.
    Returns a dictionary including the list of observations.
    """
    observation = venv.reset()
    done = False
    steps = 0
    obs_list = [] # List to store observations for GIF

    # Store initial observation
    if observation is not None and observation.shape[0] > 0:
         initial_frame_raw = observation[0] # Keep as float for now
         # --- Add Shape Check, Scale, Convert, and Transpose ---
         if initial_frame_raw.ndim == 3 and initial_frame_raw.shape[0] in [1, 3, 4]: # Check for CxHxW
             # Scale from [0, 1] (assumed) to [0, 255]
             initial_frame_scaled = (initial_frame_raw * 255.0)
             # Convert to uint8
             initial_frame_uint8 = initial_frame_scaled.astype(np.uint8)
             # Transpose from (C, H, W) to (H, W, C)
             initial_frame_transposed = np.transpose(initial_frame_uint8, (1, 2, 0))
             obs_list.append(initial_frame_transposed)
         elif initial_frame_raw.ndim == 2: # Check for HxW (grayscale)
             # Scale from [0, 1] (assumed) to [0, 255]
             initial_frame_scaled = (initial_frame_raw * 255.0)
             # Convert to uint8
             initial_frame_uint8 = initial_frame_scaled.astype(np.uint8)
             obs_list.append(initial_frame_uint8) # No transpose needed
         else:
             print(f"Warning (Initial Obs): Skipping frame with unexpected shape {initial_frame_raw.shape} for GIF.")
         # --- End Shape Check, Scale, Convert, and Transpose ---

    # Map entity codes (3-9) to type and theme/color used by heist.py EnvState
    def _get_type_theme(code):
        if code == 3: return (9, 0) # Gem
        if code == 4: return (2, 0) # Blue Key
        if code == 5: return (2, 1) # Green Key
        if code == 6: return (2, 2) # Red Key
        # Locks should have been removed, but handle just in case
        if code == 7: return (1, 0) # Blue Lock
        if code == 8: return (1, 1) # Green Lock
        if code == 9: return (1, 2) # Red Lock
        raise ValueError(f"Unknown entity code: {code}")

    try:
        # Hook is already set by experiment.set_channels/features_to_zero() before calling this function

        # Get initial state and entity positions
        initial_state = heist.state_from_venv(venv, 0)
        initial_positions = {}
        for code in required_entity_codes:
            try:
                ent_type, ent_theme = _get_type_theme(code)
                pos = initial_state.get_entity_position(ent_type, ent_theme)
                initial_positions[code] = pos
                if pos is None:
                    # This shouldn't happen if create_multi_entity_no_locks_maze worked
                    print(f"\nWarning: Required entity {ENTITY_CODE_DESCRIPTION.get(code, code)} not found in initial state (Trial {current_trial_number}). Maze creation might have failed silently.")
                    # Return error state if a required entity is missing initially
                    return {"outcome": f"error_missing_initial_{code}", "collected_entity_codes": [], "zero_target_collected": False, "all_other_keys_collected": False, "gem_collected": False, "final_player_pos": None, "observations": []}
            except ValueError as e:
                 print(f"\nError getting type/theme for entity code {code}: {e}")
                 return {"outcome": f"error_entity_mapping_{code}", "collected_entity_codes": [], "zero_target_collected": False, "all_other_keys_collected": False, "gem_collected": False, "final_player_pos": None, "observations": []}

        current_positions = initial_positions.copy()
        collected_entity_codes = []

        # --- Simulation Loop ---
        while not done and steps < max_steps:
            # Process observation (ensure correct format for model)
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=experiment.device)
            if obs_tensor.ndim == 3: # If single observation (C, H, W)
                obs_tensor = obs_tensor.unsqueeze(0) # Add batch dimension -> (1, C, H, W)
            elif obs_tensor.ndim == 4 and obs_tensor.shape[0] != 1: # If multiple envs, take first
                 obs_tensor = obs_tensor[0].unsqueeze(0)

            # Model forward pass (will trigger hook managed by experiment object)
            with torch.no_grad():
                # Access model via experiment object
                # action, value = experiment.model(obs_tensor) # For SAE, this runs base model + SAE hook
                # Use helper function to get the action
                action = helpers.generate_action(experiment.model, obs_tensor, is_procgen_env=True)

                

            # Step environment
            observation, reward, done, info = venv.step(action)
            done = done[0] # Assuming n_envs=1

            # Store observation after step
            if observation is not None and observation.shape[0] > 0:
                 current_frame_raw = observation[0] # Keep as float
                 # --- Add Shape Check, Scale, Convert, and Transpose ---
                 if current_frame_raw.ndim == 3 and current_frame_raw.shape[0] in [1, 3, 4]: # Check for CxHxW
                     # Scale from [0, 1] (assumed) to [0, 255]
                     current_frame_scaled = (current_frame_raw * 255.0)
                     # Convert to uint8
                     current_frame_uint8 = current_frame_scaled.astype(np.uint8)
                     # Transpose from (C, H, W) to (H, W, C)
                     current_frame_transposed = np.transpose(current_frame_uint8, (1, 2, 0))
                     obs_list.append(current_frame_transposed)
                 elif current_frame_raw.ndim == 2: # Check for HxW
                     # Scale from [0, 1] (assumed) to [0, 255]
                     current_frame_scaled = (current_frame_raw * 255.0)
                     # Convert to uint8
                     current_frame_uint8 = current_frame_scaled.astype(np.uint8)
                     obs_list.append(current_frame_uint8) # No transpose needed
                 else:
                     # Print a warning if the shape is wrong
                     print(f"Warning (Step {steps}): Skipping frame with unexpected shape {current_frame_raw.shape} for GIF.")
                 # --- End Shape Check, Scale, Convert, and Transpose ---

            steps += 1
        # --- End Simulation Loop ---

        # Get final state
        final_state = heist.state_from_venv(venv, 0)
        final_player_pos = final_state.mouse_pos

        # Determine collected entities
        collected_entity_codes = []
        for code in required_entity_codes:
            initial_pos = initial_positions.get(code)
            if initial_pos is not None: # Only check if it existed initially
                try:
                    ent_type, ent_theme = _get_type_theme(code)
                    final_pos = final_state.get_entity_position(ent_type, ent_theme)
                    if final_pos is None: # It existed initially but not finally -> collected
                        collected_entity_codes.append(code)
                except ValueError:
                    pass # Error handled during initial check

        # Check specific success conditions
        zero_target_collected = zero_target_entity_code in collected_entity_codes
        gem_collected = GEM_CODE in collected_entity_codes

        # Check if all *other* required keys were collected
        other_required_key_codes = (set(required_entity_codes) & KEY_CODES) - {zero_target_entity_code}
        collected_keys = set(collected_entity_codes) & KEY_CODES
        all_other_keys_collected = other_required_key_codes.issubset(collected_keys)

        outcome = 'timeout' if steps >= max_steps else 'finished'

        # Hook is disabled outside this function in the main loop's finally block

        return {
            "final_player_pos": final_player_pos,
            "outcome": outcome,
            "collected_entity_codes": collected_entity_codes,
            "zero_target_collected": zero_target_collected,
            "all_other_keys_collected": all_other_keys_collected,
            "gem_collected": gem_collected,
            "observations": obs_list # Add the collected observations
        }

    except Exception as e:
         # Handle potential errors during simulation
         print(f"\nError during simulation (Trial {current_trial_number}): {e}")
         import traceback
         traceback.print_exc()
         # Return an error state dictionary
         final_player_pos = None # Or get last known position if possible
         try: # Attempt to get final position even on error
             final_state = heist.state_from_venv(venv, 0)
             final_player_pos = final_state.get_player_pos()
         except: pass
         return {
             "outcome": f"error_simulation_{type(e).__name__}",
             "collected_entity_codes": [],
             "zero_target_collected": False,
             "all_other_keys_collected": False,
             "gem_collected": False,
             "final_player_pos": final_player_pos,
             "observations": [] # Add empty list on error
         }

def main():
    args = parse_args()
    print("Starting zero-ablation experiment with args:")
    # Use json.dumps for pretty printing args dict
    print(json.dumps(vars(args), indent=2))


    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    # Create subdirectory for successful GIFs
    gif_dir = os.path.join(args.output_dir, "success_gifs")
    os.makedirs(gif_dir, exist_ok=True)
    results = []
    # --- End Setup ---

    # --- Load Model & Experiment Class ---
    # REMOVE model loading here, as the classes handle it
    # print(f"Loading base model from: {args.model_path}")
    # model = helpers.load_interpretable_model(model_path=args.model_path)
    # model.to(device)
    # model.eval()

    num_channels_or_features = None
    experiment = None
    init_kwargs = {} # Initialize empty, populate based on class

    try:
        if args.is_sae:
            print(f"Initializing SAEZeroAblationExperiment...")
            experiment = SAEZeroAblationExperiment(
                model_path=args.model_path,
                sae_checkpoint_path=args.sae_checkpoint_path,
                layer_name=args.layer_name,
                layer_number=args.layer_number,
                device=device
            )
            num_channels_or_features = experiment.get_num_features()
            if num_channels_or_features is None:
                 raise RuntimeError("Could not determine number of SAE features.")
            print(f"Targeting SAE for layer {args.layer_name} with {num_channels_or_features} features.")

        else: # Base model
            print(f"Initializing BaseModelZeroAblationExperiment...")
            experiment = BaseModelZeroAblationExperiment(
                model_path=args.model_path,
                target_layer=args.layer_name,
                device=device
            )
            num_channels_or_features = experiment.get_num_channels()
            if num_channels_or_features is None:
                 raise RuntimeError(f"Could not determine number of channels for base layer {args.layer_name}.")
            print(f"Targeting base layer {args.layer_name} with {num_channels_or_features} channels.")

    except Exception as e:
        print(f"\nError during experiment initialization: {e}")
        import traceback
        traceback.print_exc()
        return # Exit if initialization fails
    # --- End Initialization ---

    # --- Finalize Ablation Indices for 'individual' mode ---
    if args.ablation_mode == 'individual':
        if args.start_channel >= num_channels_or_features:
             print(f"Error: --start_channel ({args.start_channel}) is >= number of channels/features ({num_channels_or_features}). No channels to ablate.")
             args.indices_to_ablate = []
        else:
             # Generate list of individual indices to iterate through
             args.indices_to_ablate = list(range(args.start_channel, num_channels_or_features))
             print(f"Will ablate indices individually from {args.start_channel} to {num_channels_or_features - 1}")

    if not args.indices_to_ablate:
         print("No indices or groups specified for ablation. Exiting.")
         return
    # ---

    # --- Experiment Loop ---
    total_simulations = len(args.indices_to_ablate) * args.num_trials
    print(f"\nStarting experiment loops. Total simulations planned: {total_simulations}")
    experiment_count = 0
    first_ablation_setting_processed = False # Tracks if the first group/channel is done
    visualization_gif_generated = False # Tracks if the GIF has been made

    # Determine loop description based on mode
    loop_desc = f"Ablation Groups ({args.zero_target_entity_name})" if args.ablation_mode == 'group' else f"Channels/Features ({args.zero_target_entity_name})"

    # Iterate through individual indices OR groups of indices
    for index_or_group in tqdm(args.indices_to_ablate, desc=loop_desc):
        # Determine the list of indices to actually ablate for this iteration
        current_ablation_list = index_or_group if args.ablation_mode == 'group' else [index_or_group]
        group_str_repr = json.dumps(current_ablation_list) # For logging/results

        # Loop over trials for the current channel/group
        trial_loop_desc = f"Trials (Ablating: {group_str_repr})"
        for trial in tqdm(range(args.num_trials), desc=trial_loop_desc, leave=False):
            experiment_count += 1
            venv_trial = None
            # Determine if this specific trial should generate the visualization GIF
            should_generate_visualization_gif = (
                args.is_sae and
                args.visualize_sae_activations and
                not visualization_gif_generated and # Only generate once
                trial == 0 and # Only for the first trial
                not first_ablation_setting_processed # Only for the first ablation group/channel
            )

            try:
                # Start SAE activation capture if needed
                if should_generate_visualization_gif:
                    experiment.start_activation_sequence_capture()

                # Create the special environment
                # --- Use create_trident_maze (or appropriate function) ---
                # Ensure the correct environment creation function is called
                initial_obs_trial, venv_trial = create_trident_maze()
                # ---

                if venv_trial is None:
                    tqdm.write(f"  Skipping trial {trial+1} for channel {index_or_group} due to environment creation failure.")
                    results.append({
                        "layer_name": args.layer_name, "layer_is_sae": args.is_sae, "sae_layer_number": args.layer_number if args.is_sae else None,
                        "ablated_indices_group": group_str_repr, # Store the group
                        "trial": trial + 1,
                        "zero_target_entity_code": args.zero_target_entity_code,
                        "zero_target_entity_name": args.zero_target_entity_name,
                        "required_entity_codes": json.dumps(args.required_entity_codes),
                        "other_required_key_codes": json.dumps(args.other_required_key_codes),
                        "outcome": "error_maze_creation",
                        "collected_entity_codes": "[]",
                        "zero_target_collected": False, "all_other_keys_collected": False, "gem_collected": False,
                        "success": False,
                        "final_player_pos_y": -1, "final_player_pos_x": -1,
                    })
                    continue

                # Set the intervention for this specific trial
                if args.is_sae:
                    experiment.set_features_to_zero(current_ablation_list)
                else:
                    experiment.set_channels_to_zero(current_ablation_list)

                # Run the simulation/trial
                trial_result = run_trial(
                    venv=venv_trial,
                    model=experiment.model,
                    max_steps=args.max_steps,
                    capture_observations=should_generate_visualization_gif
                )

                # Stop SAE activation capture
                if should_generate_visualization_gif:
                    experiment.stop_activation_sequence_capture()

                # --- Generate and Save Combined Observation/Activation GIF ---
                if should_generate_visualization_gif:
                    print("Generating post-ablation SAE activation sequence visualization GIF...")
                    captured_observations = trial_result.get("observations", [])
                    captured_activations_sequence = experiment.get_captured_activation_sequence()

                    # --- Fix: Align lists by removing the initial observation frame ---
                    if len(captured_observations) == len(captured_activations_sequence) + 1:
                        print(f"DEBUG: Aligning lists. Removing first observation frame. Original lengths: Obs={len(captured_observations)}, Act={len(captured_activations_sequence)}")
                        captured_observations = captured_observations[1:] # Slice to remove the first element
                    # ---

                    if captured_observations and captured_activations_sequence and len(captured_observations) == len(captured_activations_sequence):
                        combined_frames = []
                        num_frames = len(captured_observations)
                        # --- Ensure PIL Image is available ---
                        try:
                            from PIL import Image
                        except ImportError:
                             print("ERROR: Pillow (PIL) is required for GIF generation but couldn't be imported inside the loop.")
                             # Skip GIF generation if PIL not found here
                             captured_observations = [] # Prevent further processing
                        # ---

                        if captured_observations: # Proceed only if PIL was imported and obs exist
                            for i in tqdm(range(num_frames), desc="Generating GIF frames", leave=False):
                                obs_frame = captured_observations[i]
                                act_tensor = captured_activations_sequence[i]

                                act_image = plot_activations_to_image(
                                    act_tensor,
                                    title=f"SAE Activations (Step {i}, Ablating: {current_ablation_list})"
                                )

                                if act_image is not None:
                                    # --- Restore Resizing and Concatenation ---
                                    try:
                                        # Get observation height for resizing
                                        obs_h = obs_frame.shape[0]
                                        obs_w = obs_frame.shape[1]

                                        # Resize activation image using PIL
                                        act_pil = Image.fromarray(act_image)
                                        # Calculate new width maintaining aspect ratio
                                        act_h, act_w = act_image.shape[0], act_image.shape[1]
                                        new_act_w = int(act_w * (obs_h / act_h))
                                        act_resized_pil = act_pil.resize((new_act_w, obs_h), Image.Resampling.LANCZOS)
                                        act_resized = np.array(act_resized_pil)

                                        # Pad observation width if activation image is wider
                                        if new_act_w > obs_w:
                                             pad_width = new_act_w - obs_w
                                             # Pad with black color (0)
                                             obs_frame_padded = np.pad(obs_frame, ((0,0), (0, pad_width), (0,0)), mode='constant', constant_values=0)
                                        else:
                                             obs_frame_padded = obs_frame

                                        # Pad activation width if observation image is wider
                                        if obs_w > new_act_w:
                                             pad_width = obs_w - new_act_w
                                             act_resized_padded = np.pad(act_resized, ((0,0), (0, pad_width), (0,0)), mode='constant', constant_values=0)
                                        else:
                                             act_resized_padded = act_resized


                                        # Concatenate horizontally
                                        combined_frame = np.concatenate((obs_frame_padded, act_resized_padded), axis=1)
                                        combined_frames.append(combined_frame)
                                        # --- End Restore ---
                                    except Exception as resize_err:
                                        print(f"Warning: Error resizing/combining frame {i}: {resize_err}")
                                else:
                                    print(f"Warning: Skipping frame {i} due to activation plotting error.")

                            if combined_frames:
                                vis_gif_filename = os.path.join(gif_dir, f"post_ablation_sae_activations_layer{args.layer_number}_group_{group_str_repr}.gif")
                                try:
                                    imageio.mimsave(vis_gif_filename, combined_frames, fps=10)
                                    print(f"Saved combined activation GIF to {vis_gif_filename}")
                                    visualization_gif_generated = True # Set flag AFTER successful save
                                except Exception as gif_err:
                                    print(f"Error saving combined activation GIF: {gif_err}")
                            else:
                                print("Warning: No combined frames were generated for the activation GIF.")
                    else:
                        print(f"Warning: Skipping activation GIF generation for trial {trial+1} due to missing data or inconsistent lengths between observations ({len(captured_observations if captured_observations else [])}) and activations ({len(captured_activations_sequence if captured_activations_sequence else [])}).")
                # --- End GIF Generation ---

                # Calculate success for this trial
                success = (
                    trial_result.get('outcome') == 'completed' and
                    not trial_result.get('zero_target_collected', True) and
                    trial_result.get('all_other_keys_collected', False) and
                    trial_result.get('gem_collected', False)
                )

                # --- Save GIF if successful ---
                if success:
                    observations_for_gif = trial_result.get("observations", [])
                    if observations_for_gif:
                        # Include group info in success filename
                        success_gif_filename = os.path.join(
                            gif_dir,
                            f"success_{args.layer_name}_{'sae' if args.is_sae else 'base'}_ablating{group_str_repr}_trial{trial+1}.gif"
                        )
                        try:
                            imageio.mimsave(success_gif_filename, observations_for_gif, fps=10)
                            tqdm.write(f"  Saved successful trajectory GIF: {success_gif_filename}")
                        except Exception as gif_e:
                            tqdm.write(f"  Warning: Failed to save GIF for successful trial {trial+1}, group {group_str_repr}: {gif_e}")
                # --- End Save GIF ---

                # Record result
                final_pos = trial_result.get("final_player_pos")
                results.append({
                    "layer_name": args.layer_name,
                    "layer_is_sae": args.is_sae,
                    "sae_layer_number": args.layer_number if args.is_sae else None,
                    "ablated_indices_group": group_str_repr, # Store the group JSON string
                    "trial": trial + 1,
                    "zero_target_entity_code": args.zero_target_entity_code,
                    "zero_target_entity_name": args.zero_target_entity_name,
                    "required_entity_codes": json.dumps(args.required_entity_codes),
                    "other_required_key_codes": json.dumps(args.other_required_key_codes),
                    "outcome": trial_result.get("outcome", "error"),
                    "collected_entity_codes": json.dumps(trial_result.get("collected_entity_codes", [])),
                    "zero_target_collected": trial_result.get("zero_target_collected", True),
                    "all_other_keys_collected": trial_result.get("all_other_keys_collected", False),
                    "gem_collected": trial_result.get("gem_collected", False),
                    "success": success,
                    "final_player_pos_y": final_pos[0] if final_pos else -1,
                    "final_player_pos_x": final_pos[1] if final_pos else -1,
                })

            except Exception as e:
                 tqdm.write(f"\nRuntime Error during trial {trial+1} for channel {index_or_group}: {e}")
                 import traceback
                 traceback.print_exc() # Print full traceback for debugging
                 results.append({
                     "layer_name": args.layer_name, "layer_is_sae": args.is_sae, "sae_layer_number": args.layer_number if args.is_sae else None,
                     "ablated_indices_group": group_str_repr, # Store group
                     "trial": trial + 1,
                     "zero_target_entity_code": args.zero_target_entity_code,
                     "zero_target_entity_name": args.zero_target_entity_name,
                     "required_entity_codes": json.dumps(args.required_entity_codes),
                     "other_required_key_codes": json.dumps(args.other_required_key_codes),
                     "outcome": f"error_runtime_{type(e).__name__}",
                     "collected_entity_codes": "[]",
                     "zero_target_collected": False, "all_other_keys_collected": False, "gem_collected": False,
                     "success": False,
                     "final_player_pos_y": -1, "final_player_pos_x": -1,
                 })
            finally:
                # Disable intervention and close env
                if experiment is not None:
                    experiment.disable_zeroing()
                if venv_trial is not None:
                    venv_trial.close()

        # Mark first setting as processed after all trials for it are done
        if not first_ablation_setting_processed:
            first_ablation_setting_processed = True

    # --- End Experiment Loop ---


    # --- Save Results ---
    print("\nExperiment loop complete. Saving results...")
    results_df = pd.DataFrame(results)
    safe_layer_name = args.layer_name.replace('.', '_').replace('/', '_')
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(args.output_dir, f"zero_ablation_results_{safe_layer_name}_{args.zero_target_entity_name}_{'sae' if args.is_sae else 'base'}_{args.ablation_mode}_{timestamp_str}.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    # --- End Save Results ---


    # --- Visualize Results ---
    print("Generating visualizations...")
    if not results_df.empty:
        # Calculate success rate per group/channel
        valid_trials_df = results_df[~results_df['outcome'].str.contains('error', na=False)]

        if not valid_trials_df.empty:
            # Group by the 'ablated_indices_group' column
            success_counts = valid_trials_df[valid_trials_df['success']].groupby('ablated_indices_group').size()
            total_valid_trials = valid_trials_df.groupby('ablated_indices_group').size()
            success_counts_reindexed = success_counts.reindex(total_valid_trials.index, fill_value=0)
            success_rate = (success_counts_reindexed / total_valid_trials.replace(0, np.nan) * 100).fillna(0)

            # --- Visualization Update ---
            # The previous bar plot showed top individual channels.
            # For group ablation, plotting success rate per group might be more appropriate.
            # We'll plot all groups if the number is reasonable, or top N otherwise.

            num_groups_to_plot = len(success_rate)
            plot_title_suffix = "All Groups"
            if num_groups_to_plot > args.num_top_channels_visualize: # Reuse arg, though name is less fitting now
                 num_groups_to_plot = args.num_top_channels_visualize
                 plot_title_suffix = f"Top {num_groups_to_plot} Groups by Success Rate"

            if num_groups_to_plot > 0:
                 plot_data = success_rate.nlargest(num_groups_to_plot).sort_values(ascending=False)

                 if not plot_data.empty:
                     plt.style.use('seaborn-v0_8-darkgrid')
                     fig, ax = plt.subplots(figsize=(max(10, len(plot_data)*0.6), 6)) # Adjust figsize based on number of groups
                     plot_data.plot(kind='bar', ax=ax)
                     ax.set_title(f"Zero-Ablation Success Rate (Target: {args.zero_target_entity_name})\nLayer: {args.layer_name} ({'SAE' if args.is_sae else 'Base'}) - {plot_title_suffix}")
                     ax.set_xlabel("Ablated Indices Group (JSON String)")
                     ax.set_ylabel("Success Rate (% of valid trials)")
                     ax.set_ylim(0, 105)
                     # Ensure x-ticks match the plotted data index (group strings)
                     ax.set_xticks(range(len(plot_data)))
                     ax.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=8) # Adjust fontsize if needed
                     plt.tight_layout()

                     plot_filename = os.path.join(args.output_dir, f"zero_ablation_success_{safe_layer_name}_{args.zero_target_entity_name}_{'sae' if args.is_sae else 'base'}_{args.ablation_mode}_top{num_groups_to_plot}.png")
                     plt.savefig(plot_filename)
                     print(f"Success rate plot saved to {plot_filename}")
                     plt.close(fig)
                 else:
                      print("Skipping visualization: No groups had > 0% success rate among valid trials.")
            else:
                 print("Skipping visualization: No groups to visualize.")
            # --- End Visualization Update ---
        else:
             print("Skipping visualization: No valid (non-error) trials found in results.")
    else:
        print("Skipping visualization: No results were generated.")
    # --- End Visualize Results ---

    print(f"\nZero-ablation experiment finished for target: {args.zero_target_entity_name}.")

# --- Ensure run_trial function is defined correctly ---
def run_trial(venv, model, max_steps, capture_observations=False):
    """
    Runs a single trial (episode) in the environment.
    Optionally captures observation frames and collects entity codes.
    """
    obs = venv.reset()
    done = False
    step = 0
    total_reward = 0
    frames = [] # Only populated if capture_observations is True
    collected_codes_this_trial = set() # Use a set to avoid duplicates
    final_player_pos = None
    outcome = "timeout" # Default outcome

    while not done and step < max_steps:
        # Capture observation frame if requested
        if capture_observations:
            # Render the environment frame BEFORE taking the step
            rendered_frame = venv.render(mode='rgb_array')
            if rendered_frame is not None:
                 frames.append(rendered_frame)
            else:
                 print(f"Warning: venv.render returned None at step {step}")


        with torch.no_grad():
            # Determine device from model parameters
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                 print("Error: Model has no parameters. Cannot determine device.")
                 model_device = torch.device("cpu")

            # Prepare observation tensor
            # Assuming obs[0] is the relevant part if it's a tuple/list (from VecEnv)
            current_obs_data = obs[0] if isinstance(obs, (list, tuple)) else obs

            # --- Fix 1 (Revised): Handle potential extra dimension ---
            # Ensure current_obs_data is a NumPy array first
            if not isinstance(current_obs_data, np.ndarray):
                 print(f"Warning: current_obs_data is not a numpy array (type: {type(current_obs_data)}). Attempting conversion.")
                 try:
                     current_obs_data = np.array(current_obs_data)
                 except Exception as e:
                     print(f"ERROR: Failed to convert current_obs_data to numpy array: {e}")
                     # Handle error appropriately - maybe raise or break
                     raise RuntimeError("Failed to process observation data.") from e

            # Check shape and squeeze if necessary (e.g., if shape is (1, H, W, C))
            if current_obs_data.ndim == 4 and current_obs_data.shape[0] == 1:
                current_obs_data = current_obs_data.squeeze(0) # Remove leading dimension of size 1

            # Expecting 3D (H, W, C) at this point
            if current_obs_data.ndim != 3:
                 print(f"ERROR: Unexpected observation dimension after processing: {current_obs_data.ndim}. Shape: {current_obs_data.shape}")
                 # Handle error - maybe raise or break
                 raise RuntimeError(f"Unexpected observation dimension: {current_obs_data.ndim}")

            # Convert HWC to CHW, then add Batch dim
            obs_tensor = torch.tensor(current_obs_data, dtype=torch.float32, device=model_device).permute(2, 0, 1).unsqueeze(0)
            # ---

            # Get action from model
            try:
                policy_dist, value = model(obs_tensor)
                # --- Fix 2: Ensure action is numpy array (should already be) ---
                # This calculation should yield np.array([action_value]) for batch size 1
                action = policy_dist.logits.argmax(dim=-1).cpu().numpy()
                # ---
            except Exception as model_err:
                 print(f"Error during model forward pass or action selection: {model_err}")
                 # Fallback to random action
                 action = np.array([venv.action_space.sample()]) # Ensure it's a numpy array

        # Step environment
        try:
            # Pass the numpy array directly (e.g., np.array([action_value]))
            obs, reward, done, info = venv.step(action)
            total_reward += reward
            step += 1

            # --- Collect entity codes from info ---
            # Handle VecEnv wrapper potentially returning list for info
            info_dict = info[0] if isinstance(info, list) and len(info) > 0 else info
            if isinstance(info_dict, dict) and 'collected_entity_codes' in info_dict:
                 newly_collected = set(info_dict['collected_entity_codes'])
                 collected_codes_this_trial.update(newly_collected)
            # ---

            # Store final player position if available
            if isinstance(info_dict, dict) and 'player_pos' in info_dict:
                 final_player_pos = info_dict['player_pos'] # Assuming (y, x) tuple

        except Exception as env_err:
             print(f"Error during environment step: {env_err}")
             # Print action type/shape for debugging
             print(f"DEBUG: Action type={type(action)}, shape={getattr(action, 'shape', 'N/A')}, value={action}")
             done = True # End trial on environment error
             outcome = f"error_env_step_{type(env_err).__name__}"


    # Determine final outcome
    if not done and step >= max_steps:
        outcome = "timeout"
    elif done and outcome == "timeout": # If done flag was set by env without error
         outcome = "completed" # Assume normal completion if done is True

    # Capture final frame if needed
    if capture_observations and not done: # Capture last frame if timeout
        rendered_frame = venv.render(mode='rgb_array')
        if rendered_frame is not None:
            frames.append(rendered_frame)

    return {
        "outcome": outcome,
        "total_reward": total_reward,
        "steps": step,
        "observations": frames, # List of RGB arrays
        "collected_entity_codes": sorted(list(collected_codes_this_trial)), # Return sorted list
        "final_player_pos": final_player_pos,
    }
# --- End run_trial function ---


if __name__ == "__main__":
    main() 