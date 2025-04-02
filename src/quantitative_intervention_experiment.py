import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import imageio # Needed by environment?
import glob # For finding checkpoints
import re # For parsing step count from filenames
from tqdm import tqdm # Import tqdm

# Procgen environment
from procgen import ProcgenEnv
from procgen.gym_registration import make_env
from utils import heist # Import heist itself

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

# Based on the box_maze pattern in environment_modification_experiments.py
PLAYER_START_PATTERN_POS = (6, 3)

DEFAULT_ALT_ENTITY_CODE = ENTITY_DESCRIPTION_CODE["green_key"] # 5
DEFAULT_MAIN_ENTITY_CODE = ENTITY_DESCRIPTION_CODE["blue_key"] # 4

# Default locations
DEFAULT_MODEL_BASE_DIR = "models"
DEFAULT_SAE_CHECKPOINT_DIR = "checkpoints"

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

def find_latest_model_checkpoint(base_dir=DEFAULT_MODEL_BASE_DIR):
    """Finds the latest base model checkpoint based on step count or modification time."""
    if not os.path.isdir(base_dir):
        print(f"Warning: Model checkpoint base directory not found: {base_dir}")
        return None

    # Search potentially nested directories, e.g., models/maze_I/
    checkpoints = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
    if not checkpoints:
        print(f"Warning: No .pt files found in {base_dir} or its subdirectories.")
        return None

    latest_checkpoint = None
    max_steps = -1

    # Adjust regex if model checkpoints have a different naming pattern
    # Example: checkpoint_78643200_steps.pt
    step_pattern = re.compile(r"checkpoint_(\d+)_steps\.pt$")

    for ckpt_path in checkpoints:
        match = step_pattern.search(os.path.basename(ckpt_path))
        if match:
            steps = int(match.group(1))
            if steps > max_steps:
                max_steps = steps
                latest_checkpoint = ckpt_path

    # Fallback if no files match the step pattern
    if latest_checkpoint is None:
         print(f"Warning: Could not parse step count from model checkpoint names in {base_dir}. Using file with latest modification time.")
         try:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
         except Exception as e:
             print(f"Error finding latest model checkpoint by time: {e}")
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
                        help="Layer specification: either base model layer name (e.g., 'conv4a') or SAE layer number (e.g., '8')")
    parser.add_argument("--is_sae", action="store_true",
                        help="Flag indicating the layer_spec refers to an SAE layer number")

    # Experiment Parameters
    parser.add_argument("--target_entities", type=str, default="gem,blue_key,green_key,red_key",
                        help="Comma-separated list of entity names to target (e.g., 'gem,blue_key')")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials per channel")
    parser.add_argument("--calibration_factor", type=float, default=2.0,
                        help="Factor to multiply max activation by for intervention strength (ignored if calibration skipped)")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum number of steps per trial simulation")
    parser.add_argument("--intervention_position", type=str, default="2,1",
                        help="Intervention position as 'y,x' (e.g., '2,1')")
    parser.add_argument("--intervention_radius", type=int, default=1,
                        help="Radius for the intervention patch (0 for single point)")
    parser.add_argument("--start_channel", type=int, default=0,
                        help="Starting channel index for the experiment loop (default: 0)")
    parser.add_argument("--intervention_value", type=float, default=3.0,
                        help="Fixed value to use for the intervention (default: 3.0)")

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

    return args


def run_simulation(experiment, venv, max_steps, target_entity_code, is_sae_run, intervention_position, intervention_config=None, collect_activations_for_channel=None, collect_frames=False):
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
        collect_activations_for_channel (int, optional): If set, collect activations for this specific channel.
        collect_frames (bool): If True, collect rendered frames.

    Returns:
        dict: Contains 'final_player_pos', 'outcome', 'target_acquired', 'initial_target_pos'
              and optionally 'activations' and 'frames'.
    """
    observation = venv.reset()
    done = False
    steps = 0
    collected_channel_activations = [] if collect_activations_for_channel is not None else None
    collected_frames = [] if collect_frames else None
    handle = None # Initialize handle

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
        # Use the mouse_pos property
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
             return {"outcome": "error_entity_mapping", "final_player_pos": None, "activations": None, "initial_target_pos": None, "target_acquired": False, "frames": None}

        # Find entity positions using type and color/theme
        initial_target_pos = state.get_entity_position(target_type, target_color)

        if initial_target_pos is None:
            print(f"\nWarning: Target entity {ENTITY_CODE_DESCRIPTION.get(target_entity_code, 'Unknown')} (Type {target_type}, Theme {target_color}) not found in initial state.")

        # Store integer grid positions for comparison
        initial_target_grid_pos = (int(initial_target_pos[0]), int(initial_target_pos[1])) if initial_target_pos else None

        # Coordinate Scaling Logic
        act_h, act_w = None, None
        env_h, env_w = state.world_dim, state.world_dim # Assuming square environment grid
        target_env_pos = None
        layer_name = getattr(experiment, 'layer_name', None)
        layer_number = getattr(experiment, 'layer_number', None)

        if layer_number == 8 or layer_name == 'conv4a':
            act_h, act_w = 8, 8
        elif layer_number == 6 or layer_name == 'conv3a':
            act_h, act_w = 16, 16
        # Add more layers here if needed
        else:
             # Attempt to infer from first activation if available (less reliable)
            if is_sae_run and hasattr(experiment, 'sae') and hasattr(experiment.sae, 'hidden_channels'):
                 # This gives channels, not spatial dims. Need hook output.
                 pass # Cannot reliably get spatial dims here before hook runs
            elif not is_sae_run and hasattr(experiment, 'module'):
                 # Try to get from module output shape (if module is conv layer)
                 try:
                      # Create dummy input matching observation shape
                      dummy_input = torch.zeros(1, 3, env_h, env_w, device=experiment.device) # Assuming RGB input
                      # Pass through relevant part of model up to target layer
                      # This is complex, skipping for now. Need activation shape from hook.
                      pass
                 except Exception:
                      pass
            if act_h is None or act_w is None:
                tqdm.write(f"  [Warning] Could not determine activation dims for layer '{layer_name if layer_name else layer_number}'. Using 1:1 mapping (likely incorrect).")
                act_h, act_w = env_h, env_w # Fallback to 1:1


        if act_h is not None and act_w is not None and act_h > 0 and act_w > 0: # Ensure non-zero dimensions
            scale_y = env_h / act_h
            scale_x = env_w / act_w
            target_env_pos = (int(intervention_position[0] * scale_y), int(intervention_position[1] * scale_x))
        else:
            target_env_pos = None # Ensure it's None if scaling failed

        reached_target_area = False # Flag to track if target area was ever reached

        while not done and steps < max_steps:
            # Collect frame if requested BEFORE taking step
            if collect_frames:
                collected_frames.append(venv.render("rgb_array"))

            # Observation processing
            if isinstance(observation, tuple):
                 obs_np = observation[0]['rgb']
            elif isinstance(observation, dict):
                 obs_np = observation['rgb']
            elif isinstance(observation, np.ndarray): # Handle case where obs is directly the array
                obs_np = observation
            else:
                 raise TypeError(f"Unexpected observation format: {type(observation)}")

            # Print Env/Obs Shape on first step
            if steps == 0:
                tqdm.write(f"  [Sim Debug] Observation shape (obs_np): {obs_np.shape}")

            obs_tensor = torch.tensor(helpers.observation_to_rgb(obs_np), dtype=torch.float32).to(experiment.device)

            with torch.no_grad():
                outputs = experiment.model(obs_tensor)

            # Activation Collection
            if collect_activations_for_channel is not None:
                activation_source = None
                if is_sae_run:
                    if hasattr(experiment, 'sae_activations') and experiment.sae_activations:
                        activation_source = experiment.sae_activations
                else:
                     if hasattr(experiment, 'original_activations') and experiment.original_activations:
                          activation_source = experiment.original_activations

                if activation_source:
                    if isinstance(activation_source, list) and activation_source:
                        last_step_acts = activation_source[-1]
                        if isinstance(last_step_acts, torch.Tensor) and last_step_acts.ndim == 3 and last_step_acts.shape[0] > collect_activations_for_channel:
                            channel_act = last_step_acts[collect_activations_for_channel].clone()
                            collected_channel_activations.append(channel_act)

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
                current_state = heist.state_from_venv(venv, 0) # Get current state
                current_player_pos = current_state.mouse_pos
                if current_player_pos and not (np.isnan(current_player_pos[0]) or np.isnan(current_player_pos[1])):
                    current_player_grid_pos = (int(current_player_pos[0]), int(current_player_pos[1]))
                    if current_player_grid_pos == target_env_pos:
                        reached_target_area = True
                        tqdm.write(f"  [Debug] Agent reached target environment position {target_env_pos} at step {steps}")

        # Collect final frame if requested
        if collect_frames:
             collected_frames.append(venv.render("rgb_array"))

        # Get final state
        final_state = heist.state_from_venv(venv, 0)

        # Get final player position (float and grid cell)
        final_player_pos = None
        final_player_grid_pos = None
        final_player_pos = final_state.mouse_pos # (y, x)
        if final_player_pos:
             # Re-introduce NaN check
             if np.isnan(final_player_pos[0]) or np.isnan(final_player_pos[1]):
                  tqdm.write(f"  Warning: Final player position contains NaN {final_player_pos}. Setting grid pos to None.")
                  final_player_grid_pos = None
             else:
                  final_player_grid_pos = (int(final_player_pos[0]), int(final_player_pos[1]))
        else:
             tqdm.write(f"  Warning: final_state.mouse_pos returned None.")
             final_player_grid_pos = None # Ensure grid pos is None if float pos is None

        # Check if target entity was acquired
        final_target_pos = final_state.get_entity_position(target_type, target_color)
        target_acquired = (initial_target_pos is not None) and (final_target_pos is None)

        # Determine outcome based on whether the target area was ever reached
        outcome = 'other'
        if reached_target_area: # Check if flag was set during the loop
             outcome = 'intervention_location'

        # Clean up intervention state
        experiment.disable_intervention()

        return {
            "final_player_pos": final_player_pos,
            "outcome": outcome,
            "activations": collected_channel_activations,
            "initial_target_pos": initial_target_pos,
            "target_acquired": target_acquired,
            "frames": collected_frames # Return collected frames
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
            "model_path": model_path, # Use determined model path
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
    experiment_count = 0
    first_trial_debug_gif_saved = False

    # Initialize experiment object once
    experiment = experiment_class(**init_kwargs)

    # Outer loop over target entities
    for target_entity_code in tqdm(args.target_entities, desc="Target Entities"):
        target_entity_name = ENTITY_CODE_DESCRIPTION[target_entity_code]
        tqdm.write(f"\n--- Target Entity: {target_entity_name} ({target_entity_code}) ---")

        # Create the maze with only the target entity
        env_args = {"entity1": target_entity_code, "entity2": None}

        # Inner loop over channels
        channel_loop_desc = f"Channels ({target_entity_name}) [{args.start_channel}-{num_channels-1}]"
        for channel_index in tqdm(range(args.start_channel, num_channels), desc=channel_loop_desc, leave=False):

            # Use fixed intervention value from args (calibration is skipped)
            intervention_value = args.intervention_value
            max_activation = np.nan

            # Intervention Trials Loop
            for trial in range(args.num_trials):
                experiment_count += 1

                # Determine if this is the very first trial overall (for saving GIF)
                is_first_trial_for_gif = (
                    not first_trial_debug_gif_saved and
                    target_entity_code == args.target_entities[0] and
                    channel_index == args.start_channel and
                    trial == 0
                )

                intervention_config = [{
                    "channel": channel_index,
                    "position": args.intervention_position, # Use position from args (activation coords)
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
                    intervention_position=args.intervention_position, # Pass activation coords
                    intervention_config=intervention_config,
                    collect_activations_for_channel=None,
                    collect_frames=is_first_trial_for_gif # Collect frames only for first trial
                )

                # Keep finally block to close the environment
                try: # Add a minimal try block just for the finally clause
                    pass # No operation needed here
                finally:
                     # Ensure venv_trial exists before trying to close
                     if venv_trial is not None:
                         venv_trial.close()

                # Save Debug GIF
                if is_first_trial_for_gif and trial_result.get("frames"):
                    try:
                         safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
                         gif_filename = os.path.join(args.output_dir, f"debug_gif_{safe_layer_name}_{target_entity_name}_ch{channel_index}_trial{trial+1}_pos{args.intervention_position[0]}_{args.intervention_position[1]}.gif") # Add pos to filename
                         imageio.mimsave(gif_filename, trial_result["frames"], fps=10)
                         tqdm.write(f"    Saved debug GIF for first trial to {gif_filename}")
                         first_trial_debug_gif_saved = True # Set flag so we don't save again
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
                    "target_acquired": trial_result.get("target_acquired", True), # Default to True if missing (conservative)
                    "initial_target_pos_y": trial_result.get("initial_target_pos")[0] if trial_result.get("initial_target_pos") else -1,
                    "initial_target_pos_x": trial_result.get("initial_target_pos")[1] if trial_result.get("initial_target_pos") else -1,
                    "final_player_pos_y": trial_result.get("final_player_pos")[0] if trial_result.get("final_player_pos") else -1,
                    "final_player_pos_x": trial_result.get("final_player_pos")[1] if trial_result.get("final_player_pos") else -1,
                    "max_activation_calibration": max_activation,
                    "intervention_value": intervention_value,
                    "intervention_radius": args.intervention_radius,
                    "calibration_factor": args.calibration_factor,
                    "intervention_pos_y": args.intervention_position[0], # Record intervention pos y (activation coords)
                    "intervention_pos_x": args.intervention_position[1], # Record intervention pos x (activation coords)
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
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            print("Seaborn-v0_8-darkgrid style not available, using default.")
            plt.style.use('default')

        # Calculate success based on new definition:
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

        # Avoid division by zero for channels with 0 valid trials
        success_rate = (success_counts_reindexed / total_valid_trials.replace(0, np.nan) * 100).fillna(0) # Use NaN then fillna(0)
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
            top_channels = non_zero_success.nlargest(num_to_visualize).index
        else:
            top_channels = pd.Index([])

        if not top_channels.empty:
            success_rate_filtered_cols = success_rate.reindex(columns=top_channels, fill_value=0)

            # Plot: Overall success rate
            plt.figure(figsize=(max(10, len(top_channels)*0.6), 6))
            overall_success_rate.loc[top_channels].sort_values(ascending=False).plot(kind='bar')
            plt.title(f"Overall Success Rate (Intervention Loc Reached, Not Acquired) @ {args.intervention_position} (Act Coords) - Top {len(top_channels)} Channels ({layer_name})") # Updated Title
            plt.xlabel("Channel Index")
            plt.ylabel("Success Rate (% of valid trials)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_filename = os.path.join(args.output_dir, f"overall_success_rate_{safe_layer_name}_{'sae' if args.is_sae else 'base'}.png")
            plt.savefig(plot_filename)
            print(f"Overall success rate plot saved to {plot_filename}")
            plt.close()

            # Plot: Success rate per target entity
            num_targets = len(args.target_entities)
            fig, axes = plt.subplots(num_targets, 1, figsize=(max(10, len(top_channels)*0.6), 4 * num_targets), sharex=True, squeeze=False)
            entity_names_sorted = sorted([ENTITY_CODE_DESCRIPTION[code] for code in args.target_entities])

            for i, target_name in enumerate(entity_names_sorted):
                ax = axes[i, 0]
                if target_name in success_rate_filtered_cols.index:
                    data_to_plot = success_rate_filtered_cols.loc[target_name]
                    data_to_plot = data_to_plot.reindex(top_channels).sort_index()
                    data_to_plot.plot(kind='bar', ax=ax)
                    ax.set_title(f"Success Rate for Target Entity: {target_name}") # Simplified title
                    ax.set_ylabel("Success Rate (% of valid trials)")
                    ax.set_ylim(0, 105)
                else:
                    ax.set_title(f"Success Rate for Target Entity: {target_name} (No data/success)")
                    ax.set_ylabel("Success Rate (% of valid trials)")
                    ax.set_ylim(0, 105)
                    pd.Series(0, index=top_channels.sort_values()).plot(kind='bar', ax=ax, color='lightgrey')

            axes[-1, 0].set_xlabel("Channel Index")
            channel_labels = sorted(top_channels.tolist())
            if channel_labels:
                axes[-1, 0].set_xticks(range(len(channel_labels)))
                axes[-1, 0].set_xticklabels(channel_labels, rotation=45, ha='right')
            else:
                 axes[-1, 0].set_xticks([])
                 axes[-1, 0].set_xticklabels([])

            fig.suptitle(f"Success Rate per Target Entity @ {args.intervention_position} (Act Coords) - Top {len(top_channels)} Channels ({layer_name})", fontsize=16, y=1.02) # Updated Title
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plot_filename = os.path.join(args.output_dir, f"per_target_success_rate_{safe_layer_name}_{'sae' if args.is_sae else 'base'}.png")
            plt.savefig(plot_filename)
            print(f"Per-target success rate plot saved to {plot_filename}")
            plt.close()
        else:
             print("Skipping visualization: No channels achieved successful outcomes or top_channels is empty.")
    else:
        print("Skipping visualization: No results were generated.")

    print("\nQuantitative experiment finished.")

if __name__ == "__main__":
    main() 