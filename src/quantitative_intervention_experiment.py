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
from utils.create_intervention_mazes import create_box_maze
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
    parser.add_argument("--target_entities", type=str, default="gem,blue_key,green_key,red_key,blue_lock,green_lock,red_lock",
                        help="Comma-separated list of entity names to target (e.g., 'gem,blue_key,blue_lock')")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials per channel")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Maximum number of steps per trial simulation")
    parser.add_argument("--intervention_position", type=str, default=None,
                        help="Intervention position as 'y,x' (e.g., '2,1'). Defaults: '4,6' for conv4a (or SAE layer 8), '8,12' for conv3a (or SAE layer 6), otherwise '4,6'.")
    parser.add_argument("--intervention_radius", type=int, default=1,
                        help="Radius for the intervention patch (0 for single point)")
    parser.add_argument("--channels", type=str, default=None,
                        help="Comma-separated list of specific channel indices to run (e.g., '0,10,20'). If None, all channels for the layer are run. Overridden by --start_channel/--end_channel if they are provided.")

    # Modified Intervention Value Argument
    parser.add_argument("--intervention_value", type=str, default="0,0.7",
                        help="Intervention value. Either a fixed float (e.g., '3.0') or a range 'min,max' (e.g., '0,3') to iterate over trials.")

    # Output Configuration
    parser.add_argument("--output_dir", type=str, default="quantitative_intervention_results",
                        help="Directory to save results (CSV and plots)")
    parser.add_argument("--num_top_channels_visualize", type=int, default=20,
                         help="Number of top channels to include in the result visualization")

    # New arguments for parallel execution by a worker
    parser.add_argument("--start_channel", type=int, default=None,
                        help="Start channel index for this worker (inclusive). Overrides --channels if set.")
    parser.add_argument("--end_channel", type=int, default=None,
                        help="End channel index for this worker (exclusive). Overrides --channels if set.")

    args = parser.parse_args()

    # --- Conditional default for intervention_position ---
    if args.intervention_position is None:
        default_intervention_position_str = "4,6" # Fallback default

        layer_spec_for_check = args.layer_spec
        is_sae_for_check = args.is_sae
        layer_number_for_check = None
        layer_name_for_check = None

        if is_sae_for_check:
            try:
                layer_number_for_check = int(layer_spec_for_check)
            except ValueError:
                # Error will be handled later if layer_spec is truly invalid for SAE
                pass
        else:
            layer_name_for_check = layer_spec_for_check

        # Apply conditional defaults
        if (is_sae_for_check and layer_number_for_check == 8) or \
           (not is_sae_for_check and layer_name_for_check == 'conv4a'):
            args.intervention_position = "4,6"
            print(f"No intervention_position specified by user. Defaulting to '4,6' for layer {layer_spec_for_check}.")
        elif (is_sae_for_check and layer_number_for_check == 6) or \
             (not is_sae_for_check and layer_name_for_check == 'conv3a'):
            args.intervention_position = "8,12"
            print(f"No intervention_position specified by user. Defaulting to '8,12' for layer {layer_spec_for_check}.")
        else:
            args.intervention_position = default_intervention_position_str
            print(f"No intervention_position specified by user. Defaulting to '{default_intervention_position_str}' for layer {layer_spec_for_check} (standard fallback).")
    # --- End conditional default ---

    # Parse intervention position string (whether user-supplied or defaulted) to a tuple
    try:
        pos_parts = args.intervention_position.split(',')
        if len(pos_parts) != 2:
            raise ValueError("Position must have two parts separated by a comma.")
        args.intervention_position = (int(pos_parts[0]), int(pos_parts[1])) # (y, x)
    except Exception as e:
        parser.error(f"Invalid format for --intervention_position '{args.intervention_position}'. Use 'y,x' format (e.g., '4,6'). Error: {e}")

    # Process target entities
    args.target_entities = [ENTITY_DESCRIPTION_CODE[name.strip()] for name in args.target_entities.split(',') if name.strip() in ENTITY_DESCRIPTION_CODE]
    if not args.target_entities:
        parser.error("No valid target entities specified.")

    # Parse specified channels
    if args.channels:
        try:
            args.channels = [int(ch.strip()) for ch in args.channels.split(',') if ch.strip()]
            if not args.channels: # Ensure list is not empty after parsing
                parser.error("If --channels is provided, it must result in a non-empty list of integers.")
            # Further validation of channel numbers against num_channels will happen in main()
            print(f"Running experiment for specific channels: {args.channels}")
        except ValueError as e:
            parser.error(f"Invalid format for --channels '{args.channels}'. Must be a comma-separated list of integers. Error: {e}")
    else:
        # If --channels is None, it signifies that all channels should be run.
        # This will be handled in the main() function after num_channels is determined.
        print("No specific channels provided via --channels. All channels for the layer will be processed.")

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


def run_simulation(experiment, venv, max_steps, target_entity_code, is_sae_run, intervention_position, intervention_config=None, collect_frames=False, intervention_reached_log_path=None, current_channel_index=None, current_intervention_value=None, current_trial_number=None, target_entity_name=None, output_dir_for_debug=None):
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
        output_dir_for_debug (str, optional): Directory to save debug frames if a target entity is not found.

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
            warning_msg = f"\nWarning: Target entity {ENTITY_CODE_DESCRIPTION.get(target_entity_code, 'Unknown')} (Type {target_type}, Theme {target_color}) not found in initial state."
            print(warning_msg)
            tqdm.write(warning_msg) # Ensure it also appears in tqdm output if running in a loop

            if output_dir_for_debug:
                try:
                    debug_frames_dir = os.path.join(output_dir_for_debug, "debug_missing_entity_frames")
                    os.makedirs(debug_frames_dir, exist_ok=True)
                    
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    entity_str = target_entity_name.replace(" ", "_") if target_entity_name else "unknown_entity"
                    channel_str = f"ch{current_channel_index}" if current_channel_index is not None else "nochannel"
                    trial_str = f"trial{current_trial_number}" if current_trial_number is not None else "notrial"
                    
                    filename = f"missing_observation_{entity_str}_{channel_str}_{trial_str}_{timestamp_str}.png"
                    filepath = os.path.join(debug_frames_dir, filename)
                    
                    # Extract the agent's observation for saving
                    agent_obs_frame = None
                    if isinstance(observation, tuple) and len(observation) > 0 and isinstance(observation[0], dict) and 'rgb' in observation[0]:
                        agent_obs_frame = observation[0]['rgb']
                    elif isinstance(observation, dict) and 'rgb' in observation:
                        agent_obs_frame = observation['rgb']
                    elif isinstance(observation, np.ndarray) and observation.ndim == 3 and observation.shape[2] == 3:
                        # Assuming it's already in a suitable HWC format if it's a raw NumPy array like this
                        agent_obs_frame = observation
                    else:
                        tqdm.write(f"    Could not determine agent observation format for debug image saving. Type: {type(observation)}")

                    if agent_obs_frame is not None:
                        # Ensure it's uint8 for image saving if it's float
                        if agent_obs_frame.dtype == np.float32 or agent_obs_frame.dtype == np.float64:
                            if np.max(agent_obs_frame) <= 1.0: # Assuming 0-1 range for float
                                agent_obs_frame = (agent_obs_frame * 255).astype(np.uint8)
                            else: # Assuming 0-255 range for float, just cast
                                agent_obs_frame = agent_obs_frame.astype(np.uint8)
                        elif agent_obs_frame.dtype != np.uint8:
                             # If it's some other type, try to convert, might fail or be incorrect
                            agent_obs_frame = agent_obs_frame.astype(np.uint8)
                            
                        imageio.imwrite(filepath, agent_obs_frame)
                        print(f"    Saved debug agent observation for missing entity to: {filepath}")
                        tqdm.write(f"    Saved debug agent observation for missing entity to: {filepath}")
                    else:
                        # Fallback to render if specific observation extraction fails
                        tqdm.write(f"    Falling back to venv.render() for debug image as agent observation was not extracted.")
                        current_frame = venv.render("rgb_array")
                        imageio.imwrite(filepath.replace("missing_observation_", "missing_render_"), current_frame)
                        print(f"    Saved debug render for missing entity to: {filepath.replace('missing_observation_', 'missing_render_')}")
                        tqdm.write(f"    Saved debug render for missing entity to: {filepath.replace('missing_observation_', 'missing_render_')}")

                except Exception as e_debug_save:
                    err_msg = f"    Error saving debug frame: {e_debug_save}"
                    print(err_msg)
                    tqdm.write(err_msg)

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

    # Determine channels to run
    channels_to_run = []
    if args.channels:
        invalid_channels = [ch for ch in args.channels if not (0 <= ch < num_channels)]
        if invalid_channels:
            raise ValueError(f"Invalid channel(s) specified: {invalid_channels}. Channels must be between 0 and {num_channels - 1} for layer '{layer_name}'.")
        channels_to_run = sorted(list(set(args.channels))) # Use sorted unique channels
        print(f"Will run experiment for specified channels: {channels_to_run}")
    else:
        channels_to_run = list(range(num_channels)) # Run all channels from 0 to num_channels-1
        print(f"Will run experiment for all {num_channels} channels in layer '{layer_name}'.")

    # Override with start_channel and end_channel if provided (for worker process)
    if args.start_channel is not None and args.end_channel is not None:
        if not (0 <= args.start_channel < num_channels and 0 < args.end_channel <= num_channels and args.start_channel < args.end_channel):
            raise ValueError(
                f"Invalid --start_channel ({args.start_channel}) or --end_channel ({args.end_channel}) for {num_channels} total channels."
                f" Ensure 0 <= start_channel < end_channel <= num_channels."
            )
        channels_to_run = list(range(args.start_channel, args.end_channel))
        print(f"Worker process: Overriding channel selection. Will run for channels {args.start_channel} to {args.end_channel -1}.")
        if args.channels:
            print(f"Note: --channels argument ('{args.channels}') was ignored due to --start_channel/--end_channel being set.")


    if not channels_to_run:
        print("Warning: No channels selected to run. Exiting.")
        return # Exit if no channels are to be processed

    # Experiment Loop
    total_experiments = len(args.target_entities) * len(channels_to_run) * args.num_trials
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

    # Modify filenames to include channel range if this is a worker
    worker_id_str = ""
    if args.start_channel is not None and args.end_channel is not None:
        worker_id_str = f"_worker_{args.start_channel}-{args.end_channel-1}"

    intervention_log_filename = os.path.join(args.output_dir, f"intervention_reached_log_{safe_layer_name}_{value_str}{worker_id_str}_{timestamp_str}.csv")
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

        # Env_args remains the same for all trials of this entity
        env_args = {"entity1": target_entity_code, "entity2": None}

        # Inner loop over channels
        channel_loop_desc = f"Channels ({target_entity_name}) - {len(channels_to_run)} specified"
        if not args.channels: # If running all channels, be more specific
            channel_loop_desc = f"Channels ({target_entity_name}) [0-{num_channels-1}]"

        for channel_index in tqdm(channels_to_run, desc=channel_loop_desc, leave=False):
            # Venv creation is now INSIDE the trial loop
            # try/finally for venv closing will also be inside the trial loop

            # ------------------------------------------------------------------
            # Intervention Trials Loop  (with retry on missing target entity)
            # ------------------------------------------------------------------
            MAX_MISSING_RETRIES = 5  # how many new envs to sample if target entity absent

            # Helper to map entity code to (type, theme)
            def _entity_type_theme(code):
                if code == 3:
                    return (9, 0)   # gem
                if code == 4:
                    return (2, 0)   # blue key
                if code == 5:
                    return (2, 1)   # green key
                if code == 6:
                    return (2, 2)   # red key
                if code == 7:
                    return (1, 0)   # blue lock
                if code == 8:
                    return (1, 1)   # green lock
                if code == 9:
                    return (1, 2)   # red lock
                raise ValueError(f"Unknown entity code: {code}")

            target_type_chk, target_theme_chk = _entity_type_theme(target_entity_code)

            for trial in range(args.num_trials):
                retries_left = MAX_MISSING_RETRIES
                venv_for_this_trial = None
                initial_target_found = False

                # Retry loop to ensure target entity exists in the initial state
                while retries_left > 0:
                    if venv_for_this_trial is not None:
                        venv_for_this_trial.close()

                    observations, venv_for_this_trial = create_box_maze(**env_args)
                    state_initial_chk = heist.state_from_venv(venv_for_this_trial, 0)

                    if state_initial_chk.get_entity_position(target_type_chk, target_theme_chk) is not None:
                        initial_target_found = True
                        break  # entity exists; proceed with trial

                    retries_left -= 1

                if not initial_target_found:
                    # Record a missing-entity outcome and continue to next trial
                    results.append({
                        "layer_name": layer_name,
                        "layer_is_sae": args.is_sae,
                        "sae_layer_number": layer_number if args.is_sae else None,
                        "channel": channel_index,
                        "trial": trial + 1,
                        "target_entity_code": target_entity_code,
                        "target_entity_name": target_entity_name,
                        "outcome": "missing_entity_initial",
                        "target_acquired": False,
                        "initial_target_pos_y": -1,
                        "initial_target_pos_x": -1,
                        "final_player_pos_y": -1,
                        "final_player_pos_x": -1,
                        "intervention_value": None,
                        "intervention_radius": args.intervention_radius,
                        "intervention_pos_y": args.intervention_position[0],
                        "intervention_pos_x": args.intervention_position[1],
                    })
                    continue  # skip run_simulation for this trial

                try:
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
                        channel_index == channels_to_run[0] and # Check against the first channel in the list
                        trial == 0
                    )

                    intervention_config = [{
                        "channel": channel_index,
                        "position": args.intervention_position,
                        "value": intervention_value,
                        "radius": args.intervention_radius,
                    }]

                    trial_result = run_simulation(
                        experiment=experiment,
                        venv=venv_for_this_trial, # Use venv_for_this_trial
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
                        target_entity_name=target_entity_name,
                        output_dir_for_debug=args.output_dir # Pass output_dir for debug saving
                    )

                    # Save Debug GIF (logic for is_first_trial_for_gif might need review if mazes change per trial)
                    # For now, it will save the GIF of the first trial of the first channel of the first entity.
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
                finally:
                    # Ensure venv_for_this_trial is closed after this specific trial is done
                    if venv_for_this_trial is not None:
                        venv_for_this_trial.close()


    # Save Results
    print("\nExperiment loop complete. Saving results...")
    results_df = pd.DataFrame(results)
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    csv_filename = os.path.join(args.output_dir, f"quantitative_results_{safe_layer_name}_{'sae' if args.is_sae else 'base'}{worker_id_str}.csv")
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
            # channel_labels = sorted(top_channels.tolist()) # Replaced below
            
            # Use channels_to_run for labeling if fewer than top_channels visualized or if specific channels were run
            if args.start_channel is not None and args.end_channel is not None:
                # If worker, plots are for its specific range
                plot_channel_subset_info = f"_channels_{args.start_channel}-{args.end_channel-1}"
                # For workers, top_channels might be misleading if it's based on global stats they don't see
                # So, we'll ensure plots are made for the channels the worker actually processed, if they made it to top_channels
                # However, the current `top_channels` is derived *from this worker's data only*.
                # So, `top_channels` is already appropriate for this worker.
                channel_labels = sorted(top_channels.tolist()) if not top_channels.empty else []
            else:
                # If not a worker, or if all channels run by worker, top_channels is fine
                plot_channel_subset_info = ""
                channel_labels = sorted(top_channels.tolist()) if not top_channels.empty else []


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
                overall_plot_filename = os.path.join(entity_plot_dir, f"overall_success_rate_{value_str}_{'sae' if args.is_sae else 'base'}_top{len(top_channels)}{plot_channel_subset_info}.png")
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
                    ax_entity.set_title(f"Success Rate for Target: {target_name} ({layer_name})\n{value_title_str} - Channels: {channel_labels if channel_labels else 'N/A'}")
                else:
                    pd.Series(0, index=channel_labels).plot(kind='bar', ax=ax_entity, color='lightgrey')
                    ax_entity.set_title(f"Success Rate for Target: {target_name} ({layer_name})\n{value_title_str} - No Success in Processed Channels")

                entity_plot_filename = os.path.join(entity_plot_dir, f"success_rate_{value_str}_{'sae' if args.is_sae else 'base'}{plot_channel_subset_info}.png")
                
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
                plt.title(f"Successful Intervention Values per Channel ({layer_name})\n{value_title_str} - Channels: {channel_labels if channel_labels else 'N/A'}")
                plt.legend(title='Target Entity', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                plt.tight_layout(rect=[0, 0, 0.9, 1])
                
                scatter_filename = os.path.join(scatter_plot_dir, f"successful_interventions_scatter_{value_str}_{'sae' if args.is_sae else 'base'}{plot_channel_subset_info}.png")
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