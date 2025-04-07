import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import imageio
import glob
import re
from tqdm import tqdm
import datetime

# Procgen environment
from procgen import ProcgenEnv
from procgen.gym_registration import make_env
from utils import heist

# Project imports
from base_model_intervention import BaseModelInterventionExperiment
from sae_spatial_intervention import SAEInterventionExperiment, ordered_layer_names
from utils.environment_modification_experiments import create_box_maze
from utils import helpers

# Constants
ENTITY_CODE_DESCRIPTION = {
    3: "gem",
    4: "blue_key",
    5: "green_key",
    6: "red_key",
    7: "blue_lock",
    8: "green_lock",
    9: "red_lock"
}
ENTITY_DESCRIPTION_CODE = {v: k for k, v in ENTITY_CODE_DESCRIPTION.items()}

# Default locations
DEFAULT_MODEL_BASE_DIR = "models"
DEFAULT_SAE_CHECKPOINT_DIR = "checkpoints"

# Log header for intervention reach events
INTERVENTION_REACHED_LOG_HEADER = "Timestamp,Trial,Step,Channel,InterventionValue,TargetEntityName,ActivationPosY,ActivationPosX,TargetEnvPosY,TargetEnvPosX\n"

def find_latest_checkpoint(layer_number, layer_name, base_dir=DEFAULT_SAE_CHECKPOINT_DIR):
    """Finds the latest SAE checkpoint for a given layer based on step count."""
    layer_dir_name = f"layer_{layer_number}_{layer_name}"
    layer_dir = os.path.join(base_dir, layer_dir_name)

    if not os.path.isdir(layer_dir):
        print(f"Warning: Checkpoint directory not found: {layer_dir}")
        return None

    checkpoints = glob.glob(os.path.join(layer_dir, "*.pt"))
    if not checkpoints:
        print(f"Warning: No .pt files found in {layer_dir}")
        return None

    latest_checkpoint = None
    max_steps = -1

    step_pattern = re.compile(r"_steps_(\d+)\.pt$")

    for ckpt_path in checkpoints:
        match = step_pattern.search(os.path.basename(ckpt_path))
        if match:
            steps = int(match.group(1))
            if steps > max_steps:
                max_steps = steps
                latest_checkpoint = ckpt_path

    if latest_checkpoint is None:
         print(f"Warning: Could not parse step count from SAE checkpoint names in {layer_dir}. Using file with latest modification time.")
         try:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
         except Exception as e:
             print(f"Error finding latest SAE checkpoint by time: {e}")
             latest_checkpoint = checkpoints[-1]

    return latest_checkpoint



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run quantitative intervention experiments across channels.")

    # Model and Layer Configuration
    parser.add_argument("--model_path", type=str, default="../model_interpretable.pt",
                        help=f"Path to the base model checkpoint. If omitted, tries to find the latest in --model_base_dir.")
    parser.add_argument("--model_base_dir", type=str, default=DEFAULT_MODEL_BASE_DIR,
                        help="Base directory to search for the latest model checkpoint if --model_path is not specified.")
    parser.add_argument("--sae_checkpoint_path", type=str, default=None,
                        help=f"Path to the SAE checkpoint. If --is_sae is set and this is omitted, tries to find the latest in {DEFAULT_SAE_CHECKPOINT_DIR}/layer_{{layer_spec}}_{{layer_name}}/")
    parser.add_argument("--layer_spec", type=str, required=True,
                        help="Layer specification: either base model layer name (e.g., 'conv_seqs.2.res_block1.conv1') or SAE layer number (e.g., '18')")
    parser.add_argument("--is_sae", action="store_true",
                        help="Flag indicating the layer_spec refers to an SAE layer number")

    # Experiment Parameters
    parser.add_argument("--target_entities", type=str, default="gem,blue_key,green_key,red_key",
                        help="Comma-separated list of entity names to target (e.g., 'gem,blue_key')")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials per channel")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum number of steps per trial simulation")
    parser.add_argument("--intervention_position", type=str, default="2,1",
                        help="Intervention position as 'y,x' (e.g., '2,1')")
    parser.add_argument("--intervention_radius", type=int, default=1,
                        help="Radius for the intervention patch (0 for single point)")
    parser.add_argument("--start_channel", type=int, default=0,
                        help="Starting channel index for the experiment loop (default: 0)")

    # Modified Intervention Value Argument
    parser.add_argument("--intervention_value", type=str, default="3.0",
                        help="Intervention value. Either a fixed float (e.g., '3.0') or a range 'min,max' (e.g., '0.2,2.0') to iterate over trials.")

    # Output Configuration
    parser.add_argument("--output_dir", type=str, default="quantitative_intervention_results",
                        help="Directory to save results (CSV and plots)")
    parser.add_argument("--num_top_channels_visualize", type=int, default=20,
                         help="Number of top channels to include in the result visualization")

    args = parser.parse_args()

    # Process target entities
    args.target_entities = [ENTITY_DESCRIPTION_CODE[name.strip()] for name in args.target_entities.split(',') if name.strip() in ENTITY_DESCRIPTION_CODE]
    if not args.target_entities:
        parser.error("No valid target entities specified.")

    # Parse intervention position
    try:
        pos_parts = args.intervention_position.split(',')
        if len(pos_parts) != 2:
             raise ValueError("Position must have two parts")
        args.intervention_position = (int(pos_parts[0]), int(pos_parts[1]))
    except Exception as e:
        parser.error(f"Invalid format for --intervention_position '{args.intervention_position}'. Use 'y,x' format (e.g., '2,1'). Error: {e}")

    # Parse intervention value (new logic)
    args.intervention_is_range = False
    args.intervention_min = None
    args.intervention_max = None
    args.intervention_fixed_value = None

    if ',' in args.intervention_value:
        try:
            parts = args.intervention_value.split(',')
            if len(parts) != 2:
                raise ValueError("Range must have exactly two parts separated by a comma.")
            min_val = float(parts[0].strip())
            max_val = float(parts[1].strip())
            if min_val >= max_val:
                raise ValueError(f"Minimum value ({min_val}) must be less than maximum value ({max_val}).")
            args.intervention_min = min_val
            args.intervention_max = max_val
            args.intervention_is_range = True
            print(f"Using intervention value range: {args.intervention_min} to {args.intervention_max}")
        except ValueError as e:
            parser.error(f"Invalid format for --intervention_value range '{args.intervention_value}'. Use 'min,max' format (e.g., '0.2,2.0'). Error: {e}")
    else:
        try:
            args.intervention_fixed_value = float(args.intervention_value)
            args.intervention_is_range = False
            print(f"Using fixed intervention value: {args.intervention_fixed_value}")
        except ValueError:
            parser.error(f"Invalid format for --intervention_value '{args.intervention_value}'. Must be a float or 'min,max' range.")

    return args


def run_simulation(experiment, venv, max_steps, target_entity_code, is_sae_run, intervention_position, intervention_config=None, collect_frames=False, intervention_reached_log_path=None, current_channel_index=None, current_intervention_value=None, current_trial_number=None, target_entity_name=None):
    """
    Runs a single simulation episode, optionally applying an intervention and/or collecting activations/frames.

    Args:
        experiment: The initialized BaseModelInterventionExperiment or SAEInterventionExperiment object.
        venv: The Procgen environment instance.
        max_steps (int): Max steps for the episode.
        target_entity_code (int): Code of the target entity in the maze (3-9).
        is_sae_run (bool): Flag indicating whether the run is for an SAE layer.
        intervention_position (tuple): (y, x) position for intervention checks in *activation* coordinates.
        intervention_config (list, optional): Intervention configuration for experiment.set_intervention().
        collect_frames (bool): If True, collect rendered frames.
        intervention_reached_log_path (str, optional): Path to the log file for intervention reach events.
        current_channel_index (int, optional): The current channel index being tested.
        current_intervention_value (float, optional): The intervention value used in this trial.
        current_trial_number (int, optional): The current trial number for this channel/entity.
        target_entity_name (str, optional): The name of the target entity.

    Returns:
        dict: Contains 'final_player_pos', 'outcome', 'target_acquired', 'initial_target_pos'
              and optionally 'frames'.
    """
    observation = venv.reset()
    done = False
    steps = 0
    collected_frames = [] if collect_frames else None
    handle = None

    try:
        # Register Hook
        if is_sae_run:
            if hasattr(experiment, '_hook_sae_activations'):
                hook_func = experiment._hook_sae_activations
            else:
                 raise AttributeError("SAEExperiment object missing '_hook_sae_activations' method.")
        else:
            if hasattr(experiment, '_hook_activations'):
                hook_func = experiment._hook_activations
            else:
                 raise AttributeError("BaseModelExperiment object missing '_hook_activations' method.")

        handle = experiment.module.register_forward_hook(hook_func)

        # Reset experiment state and set intervention if needed
        experiment.original_activations = [] # Reset buffers for base model
        experiment.modified_activations = []
        if hasattr(experiment, 'sae_activations'): # Reset buffers for SAE
             experiment.sae_activations = []
             experiment.modified_sae_activations = []

        if intervention_config:
            experiment.set_intervention(intervention_config)
        else:
            experiment.disable_intervention()

        # Get actual entity positions from the state *after* reset
        state = heist.state_from_venv(venv, 0)
        player_pos = state.mouse_pos # (y, x)
        if player_pos is None:
            tqdm.write(f"\nWarning: Could not retrieve initial player position using state.mouse_pos: returned None")
        elif np.isnan(player_pos[0]) or np.isnan(player_pos[1]):
            tqdm.write(f"\nWarning: Initial player position contains NaN {player_pos}")
            player_pos = None # Treat NaN as None

        # Map entity codes (3-9) to type and theme/color used by heist.py EnvState
        def _get_type_theme(code):
            if code == 3: return (9, 0) # Gem: Type 9, Theme 0 (assumed)
            if code == 4: return (2, 0) # Blue Key: Type 2, Theme 0
            if code == 5: return (2, 1) # Green Key: Type 2, Theme 1
            if code == 6: return (2, 2) # Red Key: Type 2, Theme 2
            if code == 7: return (1, 0) # Blue Lock: Type 1, Theme 0
            if code == 8: return (1, 1) # Green Lock: Type 1, Theme 1
            if code == 9: return (1, 2) # Red Lock: Type 1, Theme 2
            raise ValueError(f"Unknown entity code: {code}")

        try:
            target_type, target_color = _get_type_theme(target_entity_code)
        except ValueError as e:
             print(f"\nError: {e}")
             # Handle error appropriately - maybe skip trial or return error state
             return {"outcome": "error_entity_mapping", "final_player_pos": None, "initial_target_pos": None, "target_acquired": False, "frames": None}

        # Find entity positions using type and color/theme
        initial_target_pos = state.get_entity_position(target_type, target_color)

        if initial_target_pos is None:
            print(f"\nWarning: Target entity {ENTITY_CODE_DESCRIPTION.get(target_entity_code, 'Unknown')} (Type {target_type}, Theme {target_color}) not found in initial state.")

        # Store integer grid positions for comparison
        initial_target_grid_pos = (int(initial_target_pos[0]), int(initial_target_pos[1])) if initial_target_pos else None

        # Coordinate Scaling Logic
        act_h, act_w = None, None
        env_h, env_w = state.world_dim, state.world_dim
        target_env_pos = None
        layer_name = getattr(experiment, 'layer_name', None)
        layer_number = getattr(experiment, 'layer_number', None)

        # Map layer names/numbers to their activation dimensions
        if layer_name == 'input' or layer_number == 0:
            act_h, act_w = 64, 64  # Input: 64×64×3
        elif layer_name == 'conv1a' or layer_number == 2:
            act_h, act_w = 64, 64  # After Conv Block 1: 32×32×16
        elif layer_name == 'conv2a' or layer_number == 4:
            act_h, act_w = 32, 32  # After Conv Block 2: 16×16×32
        elif layer_name == 'conv2b' or layer_number == 5:
            act_h, act_w = 32, 32  # After Conv Block 2: 16×16×32
        elif layer_name == 'conv3a' or layer_number == 6:
            act_h, act_w = 16, 16    # After Conv Block 3: 8×8×32
        elif layer_name == 'conv4a' or layer_number == 8:
            act_h, act_w = 8, 8    # After Conv Block 4: 4×4×32

        if act_h is not None and act_w is not None and act_h > 0 and act_w > 0:
            scale_y = env_h / act_h
            scale_x = env_w / act_w
            target_env_pos = (int(intervention_position[0] * scale_y), int(intervention_position[1] * scale_x))
        else:
            target_env_pos = None # Ensure it's None if scaling failed

        reached_target_area = False
        logged_reach_this_trial = False

        while not done and steps < max_steps:
            if collect_frames:
                collected_frames.append(venv.render("rgb_array"))

            # Observation processing
            if isinstance(observation, tuple):
                 obs_np = observation[0]['rgb']
            elif isinstance(observation, dict):
                 obs_np = observation['rgb']
            elif isinstance(observation, np.ndarray):
                obs_np = observation
            else:
                 raise TypeError(f"Unexpected observation format: {type(observation)}")

            obs_tensor = torch.tensor(helpers.observation_to_rgb(obs_np), dtype=torch.float32).to(experiment.device)

            with torch.no_grad():
                outputs = experiment.model(obs_tensor)

            # Get action
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple) and hasattr(outputs[0], 'logits'):
                 logits = outputs[0].logits
            elif isinstance(outputs, torch.Tensor):
                 logits = outputs
            else:
                 try:
                     logits = getattr(outputs, 'pi_logits', None)
                     if logits is None:
                         raise ValueError("Could not reliably extract logits from model output.")
                 except Exception:
                      raise ValueError(f"Could not extract logits from model output of type {type(outputs)}.")

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            action = torch.multinomial(probabilities, num_samples=1).squeeze(-1).cpu().numpy()

            observation, reward, done, info = venv.step(action)
            steps += 1

            # Check if agent reached target area
            if not reached_target_area and target_env_pos is not None:
                current_state = heist.state_from_venv(venv, 0)
                current_player_pos = current_state.mouse_pos
                if current_player_pos and not (np.isnan(current_player_pos[0]) or np.isnan(current_player_pos[1])):
                    current_player_grid_pos = (int(current_player_pos[0]), int(current_player_pos[1]))
                    if current_player_grid_pos == target_env_pos:
                        reached_target_area = True


                        # Log the first time the agent reaches the target environment cell in this trial
                        if intervention_reached_log_path and not logged_reach_this_trial:
                            try:
                                timestamp = datetime.datetime.now().isoformat()
                                log_entry = (
                                    f"{timestamp},"
                                    f"{current_trial_number if current_trial_number is not None else 'N/A'},"
                                    f"{steps},"
                                    f"{current_channel_index if current_channel_index is not None else 'N/A'},"
                                    f"{current_intervention_value if current_intervention_value is not None else 'N/A'},"
                                    f"{target_entity_name if target_entity_name else 'N/A'},"
                                    f"{intervention_position[0]},"
                                    f"{intervention_position[1]},"
                                    f"{target_env_pos[0]},"
                                    f"{target_env_pos[1]}\n"
                                )
                                with open(intervention_reached_log_path, 'a') as log_f:
                                    log_f.write(log_entry)
                                logged_reach_this_trial = True
                            except Exception as e_log:
                                tqdm.write(f"  [Warning] Failed to write to intervention reach log '{intervention_reached_log_path}': {e_log}")

        # Collect final frame if requested
        if collect_frames:
             collected_frames.append(venv.render("rgb_array"))

        # Get final state
        final_state = heist.state_from_venv(venv, 0)

        # Get final player position (float and grid cell)
        final_player_pos = None

        final_player_pos = final_state.mouse_pos # (y, x)


        # Check if target entity was acquired
        final_target_pos = final_state.get_entity_position(target_type, target_color)
        target_acquired = (initial_target_pos is not None) and (final_target_pos is None)

        # Determine outcome
        outcome = 'other'
        if reached_target_area:
             outcome = 'intervention_location'

        # Clean up intervention state
        experiment.disable_intervention()

        return {
            "final_player_pos": final_player_pos,
            "outcome": outcome,
            "initial_target_pos": initial_target_pos,
            "target_acquired": target_acquired,
            "frames": collected_frames
        }

    finally:
        # Ensure Hook Removal
        if handle is not None:
            handle.remove()


def main():
    args = parse_args()
    print("Starting quantitative intervention experiment with args:", args)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    # Determine Model Path
    model_path = args.model_path
    if not model_path:
        print(f"--model_path not specified, searching for latest model checkpoint in '{args.model_base_dir}'...")
        model_path = find_latest_model_checkpoint(base_dir=args.model_base_dir)
        if not model_path:
            raise ValueError(f"Could not automatically find a model checkpoint in '{args.model_base_dir}' or its subdirectories. Please specify the path using --model_path.")
        print(f"Found latest model checkpoint: {model_path}")
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"Specified model path not found: {model_path}")

    # Layer and Channel Setup
    num_channels = None
    sae_checkpoint_path = args.sae_checkpoint_path
    layer_number = None
    layer_name = None

    if args.is_sae:
        try:
            layer_number = int(args.layer_spec)
        except ValueError:
            raise ValueError(f"Invalid layer_spec for SAE: '{args.layer_spec}'. Must be an integer layer number.")

        layer_name = ordered_layer_names.get(layer_number)
        if not layer_name:
             raise ValueError(f"Invalid SAE layer number: {layer_number}. Check sae_cnn.ordered_layer_names.")

        # Auto-detect SAE checkpoint if path not provided
        if not sae_checkpoint_path:
            print(f"--sae_checkpoint_path not specified, searching for latest checkpoint for layer {layer_number} ({layer_name}) in '{DEFAULT_SAE_CHECKPOINT_DIR}'...")
            sae_checkpoint_path = find_latest_checkpoint(layer_number, layer_name, base_dir=DEFAULT_SAE_CHECKPOINT_DIR)
            if not sae_checkpoint_path:
                expected_dir = os.path.join(DEFAULT_SAE_CHECKPOINT_DIR, f"layer_{layer_number}_{layer_name}")
                raise ValueError(f"Could not automatically find SAE checkpoint for layer {layer_number} ({layer_name}) in directory structure '{expected_dir}/'. Please specify the path using --sae_checkpoint_path.")
            print(f"Found latest checkpoint: {sae_checkpoint_path}")
        elif not os.path.exists(sae_checkpoint_path):
             raise FileNotFoundError(f"Specified SAE checkpoint path not found: {sae_checkpoint_path}")

        print(f"Targeting SAE layer {layer_number} ({layer_name}) from {sae_checkpoint_path}")
        experiment_class = SAEInterventionExperiment
        init_kwargs = {
            "model_path": model_path,
            "sae_checkpoint_path": sae_checkpoint_path,
            "layer_number": layer_number,
            "device": device
        }
        try:
            temp_exp = experiment_class(**init_kwargs)
            # Use hidden_channels instead of d_sae for ConvSAE
            num_channels = temp_exp.sae.hidden_channels
            del temp_exp
        except Exception as e:
             print(f"Error loading SAE (path: {sae_checkpoint_path}) to determine channel count: {e}")
             raise

    else: # Base model layer
        layer_name = args.layer_spec
        if args.sae_checkpoint_path:
            print("Warning: --sae_checkpoint_path provided but --is_sae is not set. The path will be ignored.")

        print(f"Targeting base model layer: {layer_name}")
        experiment_class = BaseModelInterventionExperiment
        init_kwargs = {
            "model_path": model_path,
            "target_layer": layer_name,
            "device": device
        }
        try:
            temp_exp = experiment_class(**init_kwargs)
            module = temp_exp.module
            if hasattr(module, 'out_channels'):
                 num_channels = module.out_channels
            else:
                 print(f"Warning: Cannot automatically determine channel count for layer {layer_name} via out_channels.")
                 weight_param = next((p for name, p in module.named_parameters() if 'weight' in name), None)
                 if weight_param is not None:
                      if weight_param.ndim >= 1:
                           num_channels = weight_param.shape[0]
                           print(f"Inferred channel count {num_channels} from weight shape.")
                      else:
                           raise ValueError("Weight parameter found but has unexpected dimensions.")
                 else:
                      raise ValueError(f"Cannot determine channel count for layer {layer_name}.")
            del temp_exp
        except Exception as e:
            print(f"Error initializing experiment/module to determine channel count: {e}")
            raise

    if num_channels is None:
        raise RuntimeError("Failed to determine the number of channels for the specified layer.")
    if layer_name is None:
        raise RuntimeError("Internal error: layer_name was not set.")
    print(f"Layer '{layer_name}' has {num_channels} channels.")

    # Experiment Loop
    total_experiments = len(args.target_entities) * (num_channels - args.start_channel) * args.num_trials
    print(f"\nStarting experiment loops. Total simulations planned: {total_experiments}")
    first_trial_debug_gif_saved = False

    # Initialize experiment object once
    experiment = experiment_class(**init_kwargs)

    # --- Intervention Reach Log Setup ---
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')

    # Determine intervention value string for filename
    if hasattr(args, 'intervention_is_range') and args.intervention_is_range:
        value_str = f"range{args.intervention_min}-{args.intervention_max}"
    elif hasattr(args, 'intervention_fixed_value'):
        value_str = f"fixed{args.intervention_fixed_value}"
    else:
        value_str = f"val{args.intervention_value}"

    intervention_log_filename = os.path.join(args.output_dir, f"intervention_reached_log_{safe_layer_name}_{value_str}_{timestamp_str}.csv")
    intervention_log_enabled = False
    try:
        with open(intervention_log_filename, 'w') as f_log:
            f_log.write(INTERVENTION_REACHED_LOG_HEADER)
        intervention_log_enabled = True
        print(f"Intervention reach events will be logged to: {intervention_log_filename}")
    except Exception as e_log_init:
        print(f"Warning: Could not create intervention reach log file '{intervention_log_filename}'. Logging disabled. Error: {e_log_init}")
        intervention_log_filename = None
    # --- End Log Setup ---

    # Outer loop over target entities
    for target_entity_code in tqdm(args.target_entities, desc="Target Entities"):
        target_entity_name = ENTITY_CODE_DESCRIPTION[target_entity_code]
        tqdm.write(f"\n--- Target Entity: {target_entity_name} ({target_entity_code}) ---")

        # Create the maze with only the target entity
        env_args = {"entity1": target_entity_code, "entity2": None}

        # Inner loop over channels
        channel_loop_desc = f"Channels ({target_entity_name}) [{args.start_channel}-{num_channels-1}]"
        for channel_index in tqdm(range(args.start_channel, num_channels), desc=channel_loop_desc, leave=False):

            # Intervention Trials Loop
            for trial in range(args.num_trials):

                # --- Determine Intervention Value ---
                intervention_value = None
                if hasattr(args, 'intervention_is_range') and args.intervention_is_range:
                    if args.num_trials > 1:
                        step_size = (args.intervention_max - args.intervention_min) / (args.num_trials - 1)
                        intervention_value = args.intervention_min + trial * step_size
                    else: # Handle edge case of only 1 trial
                        intervention_value = (args.intervention_min + args.intervention_max) / 2.0
                elif hasattr(args, 'intervention_fixed_value'):
                     intervention_value = args.intervention_fixed_value
                else:
                    # Fallback
                    tqdm.write("[Warning] Could not determine intervention range/fixed value, using raw --intervention_value.")
                    intervention_value = args.intervention_value
                # --- End Intervention Value Determination ---

                # Determine if this is the very first trial overall (for saving GIF)
                is_first_trial_for_gif = (
                    not first_trial_debug_gif_saved and
                    target_entity_code == args.target_entities[0] and
                    channel_index == args.start_channel and
                    trial == 0
                )

                intervention_config = [{
                    "channel": channel_index,
                    "position": args.intervention_position,
                    "value": intervention_value,
                    "radius": args.intervention_radius,
                }]

                venv_trial = None # Initialize to prevent potential NameError in finally if create_box_maze *does* fail for other reasons
                observations, venv_trial = create_box_maze(**env_args)

                trial_result = run_simulation(
                    experiment=experiment,
                    venv=venv_trial, # Use venv_trial defined above
                    max_steps=args.max_steps,
                    target_entity_code=target_entity_code,
                    is_sae_run=args.is_sae,
                    intervention_position=args.intervention_position,
                    intervention_config=intervention_config,
                    collect_frames=is_first_trial_for_gif,
                    # Logging info
                    intervention_reached_log_path=intervention_log_filename,
                    current_channel_index=channel_index,
                    current_intervention_value=intervention_value,
                    current_trial_number=trial + 1,
                    target_entity_name=target_entity_name
                )

                # Keep finally block to close the environment
                try:
                    pass
                finally:
                     # Ensure venv_trial exists before closing
                     if venv_trial is not None:
                         venv_trial.close()

                # Save Debug GIF
                if is_first_trial_for_gif and trial_result.get("frames"):
                    try:
                         safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
                         # Entity-specific directory
                         debug_gif_dir = os.path.join(args.output_dir, safe_layer_name, target_entity_name)
                         os.makedirs(debug_gif_dir, exist_ok=True)
                         # Construct filename
                         gif_filename = os.path.join(debug_gif_dir, f"debug_gif_ch{channel_index}_trial{trial+1}_pos{args.intervention_position[0]}_{args.intervention_position[1]}.gif")
                         imageio.mimsave(gif_filename, trial_result["frames"], fps=10)
                         tqdm.write(f"    Saved debug GIF for first trial to {gif_filename}")
                         first_trial_debug_gif_saved = True
                    except Exception as e_gif:
                         tqdm.write(f"    Error saving debug GIF: {e_gif}")


                # Record result
                results.append({
                    "layer_name": layer_name,
                    "layer_is_sae": args.is_sae,
                    "sae_layer_number": layer_number if args.is_sae else None,
                    "channel": channel_index,
                    "trial": trial + 1,
                    "target_entity_code": target_entity_code,
                    "target_entity_name": target_entity_name,
                    "outcome": trial_result.get("outcome", "error"),
                    "target_acquired": trial_result.get("target_acquired", True),
                    "initial_target_pos_y": trial_result.get("initial_target_pos")[0] if trial_result.get("initial_target_pos") else -1,
                    "initial_target_pos_x": trial_result.get("initial_target_pos")[1] if trial_result.get("initial_target_pos") else -1,
                    "final_player_pos_y": trial_result.get("final_player_pos")[0] if trial_result.get("final_player_pos") else -1,
                    "final_player_pos_x": trial_result.get("final_player_pos")[1] if trial_result.get("final_player_pos") else -1,
                    "intervention_value": intervention_value,
                    "intervention_radius": args.intervention_radius,
                    "intervention_pos_y": args.intervention_position[0],
                    "intervention_pos_x": args.intervention_position[1],
                })


    # Save Results
    print("\nExperiment loop complete. Saving results...")
    results_df = pd.DataFrame(results)
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    csv_filename = os.path.join(args.output_dir, f"quantitative_results_{safe_layer_name}_{'sae' if args.is_sae else 'base'}.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    # Visualize Results
    print("Generating visualizations...")
    if not results_df.empty:
        try:
            plt.style.use('default')
        except OSError:
            print("Seaborn-v0_8-darkgrid style not available, using default.")
            plt.style.use('default')

        # outcome is 'intervention_location' AND target_acquired is False
        results_df['success'] = (results_df['outcome'] == 'intervention_location') & (results_df['target_acquired'] == False)

        success_counts = results_df[results_df['success']].groupby(
            ['target_entity_name', 'channel']
        ).size()
        total_valid_trials = results_df[results_df['outcome'] != 'error'].groupby(
            ['target_entity_name', 'channel']
        ).size()

        # Ensure indices match before division, fill missing with 0
        success_counts_reindexed = success_counts.reindex(total_valid_trials.index, fill_value=0)

        # Avoid division by zero
        success_rate = (success_counts_reindexed / total_valid_trials.replace(0, np.nan) * 100).fillna(0)
        success_rate = success_rate.unstack(level='channel', fill_value=0)

        # Calculate overall success rate
        overall_success_counts = results_df[results_df['success']].groupby('channel').size()
        overall_valid_trials = results_df[results_df['outcome'] != 'error'].groupby('channel').size()

        # Ensure indices match before division
        overall_success_counts_reindexed = overall_success_counts.reindex(overall_valid_trials.index, fill_value=0)

        overall_success_rate = (overall_success_counts_reindexed / overall_valid_trials.replace(0, np.nan) * 100).fillna(0)
        overall_success_rate = overall_success_rate.fillna(0)

        num_to_visualize = min(args.num_top_channels_visualize, len(overall_success_rate))
        if num_to_visualize > 0:
            # Filter out channels with 0% success rate before finding top N
            non_zero_success = overall_success_rate[overall_success_rate > 0]
            # Handle case where non_zero_success might be shorter than num_to_visualize
            num_actual_visualize = min(num_to_visualize, len(non_zero_success))
            if num_actual_visualize > 0:
                top_channels = non_zero_success.nlargest(num_actual_visualize).index
            else:
                top_channels = pd.Index([])
                print("Skipping visualization: No channels had > 0% success rate.")
        else:
            top_channels = pd.Index([])
            print("Skipping visualization: num_top_channels_visualize is 0 or less.")


        # Only proceed if we have channels to visualize
        if not top_channels.empty:

            # Create descriptive string for plot titles/filenames
            if args.intervention_is_range:
                value_title_str = f"Value Range: {args.intervention_min}-{args.intervention_max}"
            else:
                value_title_str = f"Fixed Value: {args.intervention_fixed_value}"
            # value_str is already defined above

            # --- Plotting Loop (Per-Entity and Overall) ---
            success_rate_filtered_cols = success_rate.reindex(columns=top_channels, fill_value=0)
            entity_names_sorted = sorted([ENTITY_CODE_DESCRIPTION[code] for code in args.target_entities])
            channel_labels = sorted(top_channels.tolist())

            for target_name in entity_names_sorted:
                # Entity-specific directory
                entity_plot_dir = os.path.join(args.output_dir, safe_layer_name, target_name)
                os.makedirs(entity_plot_dir, exist_ok=True)

                # --- Generate and Save Overall Plot (Inside Entity Folder) ---
                fig_overall, ax_overall = plt.subplots(1, 1, figsize=(max(10, len(top_channels)*0.6), 6))
                overall_success_rate.loc[top_channels].sort_values(ascending=False).plot(kind='bar', ax=ax_overall)
                ax_overall.set_title(f"Overall Success Rate @ {args.intervention_position} (Act Coords)\\n{value_title_str} - Top {len(top_channels)} Channels ({layer_name})")
                ax_overall.set_xlabel("Channel Index")
                ax_overall.set_ylabel("Success Rate (% of valid trials)")
                if channel_labels:
                    ax_overall.set_xticks(range(len(channel_labels)))
                    ax_overall.set_xticklabels(channel_labels, rotation=45, ha='right')
                else:
                    ax_overall.set_xticks([])
                    ax_overall.set_xticklabels([])
                fig_overall.tight_layout()
                overall_plot_filename = os.path.join(entity_plot_dir, f"overall_success_rate_{value_str}_{'sae' if args.is_sae else 'base'}_top{len(top_channels)}.png")
                fig_overall.savefig(overall_plot_filename)
                print(f"Overall success rate plot saved to {overall_plot_filename}")
                plt.close(fig_overall)
                # --- End Overall Plot Saving ---

                # --- Generate and Save Per-Target Plot ---
                fig_entity, ax_entity = plt.subplots(1, 1, figsize=(max(10, len(top_channels)*0.6), 5))
                
                ax_entity.set_ylabel("Success Rate (% of valid trials)")
                ax_entity.set_ylim(0, 105)
                ax_entity.set_xlabel("Channel Index")
                if channel_labels:
                    ax_entity.set_xticks(range(len(channel_labels)))
                    ax_entity.set_xticklabels(channel_labels, rotation=45, ha='right')
                else:
                    ax_entity.set_xticks([])
                    ax_entity.set_xticklabels([])

                if target_name in success_rate_filtered_cols.index:
                    data_to_plot = success_rate_filtered_cols.loc[target_name]
                    data_to_plot = data_to_plot.reindex(channel_labels).fillna(0)
                    data_to_plot.plot(kind='bar', ax=ax_entity)
                    ax_entity.set_title(f"Success Rate for Target: {target_name} ({layer_name})\n{value_title_str} - Top {len(channel_labels)} Channels")
                else:
                    pd.Series(0, index=channel_labels).plot(kind='bar', ax=ax_entity, color='lightgrey')
                    ax_entity.set_title(f"Success Rate for Target: {target_name} ({layer_name})\n{value_title_str} - No Success in Top Channels")

                entity_plot_filename = os.path.join(entity_plot_dir, f"success_rate_{value_str}_{'sae' if args.is_sae else 'base'}.png")
                
                fig_entity.tight_layout()
                fig_entity.savefig(entity_plot_filename)
                print(f"Per-target success rate plot saved to {entity_plot_filename}")
                plt.close(fig_entity)
            # --- End Plotting Loop ---

            # --- Plot: Scatter plot of successful intervention values vs channel ---
            # Filter results to only include successful trials for the top channels
            successful_trials_df = results_df[(results_df['success'] == True) & (results_df['channel'].isin(top_channels))]

            if not successful_trials_df.empty:
                scatter_plot_dir = os.path.join(args.output_dir, safe_layer_name, "overall_results")
                os.makedirs(scatter_plot_dir, exist_ok=True)

                fig_height = max(6, len(top_channels) * 0.35)
                fig_width = max(12, len(top_channels) * 0.5)
                plt.figure(figsize=(fig_width, fig_height))
                
                sns.stripplot(data=successful_trials_df,
                                x='channel',
                                y='intervention_value',
                                hue='target_entity_name', 
                                palette='tab10',
                                alpha=0.7,
                                s=15,
                                jitter=True,
                                order=sorted(top_channels.tolist()))
                
                plt.xticks(rotation=45, ha='right')
                plt.xlabel("Channel Index")
                plt.ylabel("Successful Intervention Value Applied")
                plt.title(f"Successful Intervention Values per Channel ({layer_name})\n{value_title_str} - Top {len(top_channels)} Channels")
                plt.legend(title='Target Entity', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                plt.tight_layout(rect=[0, 0, 0.9, 1])
                
                scatter_filename = os.path.join(scatter_plot_dir, f"successful_interventions_scatter_{value_str}_{'sae' if args.is_sae else 'base'}.png")
                plt.savefig(scatter_filename, bbox_inches='tight')
                print(f"Successful interventions scatter plot saved to {scatter_filename}")
                plt.close()
            else:
                print("Skipping successful interventions scatter plot: No successful trials found for top channels.")

        else:
             print("Skipping visualization: No channels achieved successful outcomes or top_channels is empty based on criteria.")
    else:
        print("Skipping visualization: No results were generated.")

    print("\nQuantitative experiment finished.")

if __name__ == "__main__":
    main() 