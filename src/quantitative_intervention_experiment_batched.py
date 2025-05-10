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
import sys # For muting stderr
from contextlib import contextmanager # For muting stderr

# Project imports
from base_model_intervention import BaseModelInterventionExperiment
from sae_spatial_intervention import SAEInterventionExperiment, ordered_layer_names
from utils.environment_modification_experiments import create_box_maze
from utils import heist
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

# Context manager to suppress stderr
@contextmanager
def suppress_stderr():
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

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
    parser.add_argument("--intervention_position", type=str, default="4,6",
                        help="Intervention position as 'y,x' (e.g., '2,1')")
    parser.add_argument("--intervention_radius", type=int, default=1,
                        help="Radius for the intervention patch (0 for single point)")
    parser.add_argument("--channels", type=str, default=None,
                        help="Comma-separated list of specific channel indices to run (e.g., '0,10,20'). If None, all channels for the layer are run.")
    parser.add_argument("--num_parallel_envs", type=int, default=16,
                        help="Number of parallel environments to run simultaneously")

    # Modified Intervention Value Argument
    parser.add_argument("--intervention_value", type=str, default="3.0",
                        help="Intervention value. Either a fixed float (e.g., '3.0') or a range 'min,max' (e.g., '0.2,2.0') to iterate over trials.")

    # Output Configuration
    parser.add_argument("--output_dir", type=str, default="quantitative_intervention_results",
                        help="Directory to save results (CSV and plots)")
    parser.add_argument("--num_top_channels_visualize", type=int, default=20,
                         help="Number of top channels to include in the result visualization")
    parser.add_argument("--save_debug_gif", action="store_true",
                        help="If set, save a GIF of the first trial of the first processed channel for visual debugging.")

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

    # Parse specified channels
    if args.channels:
        try:
            args.channels = [int(ch.strip()) for ch in args.channels.split(',') if ch.strip()]
            if not args.channels: # Ensure list is not empty after parsing
                parser.error("If --channels is provided, it must result in a non-empty list of integers.")
            print(f"Running experiment for specific channels: {args.channels}")
        except ValueError as e:
            parser.error(f"Invalid format for --channels '{args.channels}'. Must be a comma-separated list of integers. Error: {e}")
    else:
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

class BatchedInterventionExperiment:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Store intervention position from args
        self.intervention_position = args.intervention_position
        self.debug_gif_frames = None # Initialize for GIF debugging

        # Initialize model and experiment
        if args.is_sae:
            self.experiment = SAEInterventionExperiment(
                model_path=args.model_path,
                sae_checkpoint_path=args.sae_checkpoint_path,
                layer_number=int(args.layer_spec),  # Convert layer_spec to int for SAE
                device=self.device
            )
        else:
            self.experiment = BaseModelInterventionExperiment(
                model_path=args.model_path,
                layer_spec=args.layer_spec,
                device=self.device
            )

        # --- ADD THIS BLOCK TO SET self.model ---
        if hasattr(self.experiment, 'model'):
            self.model = self.experiment.model
        elif hasattr(self.experiment, 'get_model'): # Common alternative getter
            self.model = self.experiment.get_model()
        else:
            raise AttributeError(
                "The experiment object (SAEInterventionExperiment or BaseModelInterventionExperiment) "
                "does not have a .model attribute or a .get_model() method. "
                "Cannot retrieve the underlying PyTorch model."
            )
        # --- END OF ADDED BLOCK ---

        # Create output directory with descriptive name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get SAE name if it's an SAE experiment
        sae_name = None
        if args.is_sae:
            # Extract SAE name from checkpoint path if available
            if args.sae_checkpoint_path:
                checkpoint_name = os.path.basename(args.sae_checkpoint_path)
                # Try to extract SAE name from checkpoint filename
                if '_alpha_' in checkpoint_name:
                    sae_name = 'alpha'
                elif '_beta_' in checkpoint_name:
                    sae_name = 'beta'
                elif '_gamma_' in checkpoint_name:
                    sae_name = 'gamma'
                elif '_delta_' in checkpoint_name:
                    sae_name = 'delta'
                elif '_epsilon_' in checkpoint_name:
                    sae_name = 'epsilon'
                elif '_zeta_' in checkpoint_name:
                    sae_name = 'zeta'
                elif '_eta_' in checkpoint_name:
                    sae_name = 'eta'
                elif '_theta_' in checkpoint_name:
                    sae_name = 'theta'
                elif '_iota_' in checkpoint_name:
                    sae_name = 'iota'
                elif '_kappa_' in checkpoint_name:
                    sae_name = 'kappa'
                elif '_lambda_' in checkpoint_name:
                    sae_name = 'lambda'
                elif '_mu_' in checkpoint_name:
                    sae_name = 'mu'
                elif '_nu_' in checkpoint_name:
                    sae_name = 'nu'
                elif '_xi_' in checkpoint_name:
                    sae_name = 'xi'
                elif '_omicron_' in checkpoint_name:
                    sae_name = 'omicron'
                elif '_pi_' in checkpoint_name:
                    sae_name = 'pi'
                elif '_rho_' in checkpoint_name:
                    sae_name = 'rho'
                elif '_sigma_' in checkpoint_name:
                    sae_name = 'sigma'
                elif '_tau_' in checkpoint_name:
                    sae_name = 'tau'
                elif '_upsilon_' in checkpoint_name:
                    sae_name = 'upsilon'
                elif '_phi_' in checkpoint_name:
                    sae_name = 'phi'
                elif '_chi_' in checkpoint_name:
                    sae_name = 'chi'
                elif '_psi_' in checkpoint_name:
                    sae_name = 'psi'
                elif '_omega_' in checkpoint_name:
                    sae_name = 'omega'
        
        # Create a clean, descriptive directory name
        if args.is_sae:
            if sae_name:
                dir_name = f"sae_{sae_name}_l{args.layer_spec}"
            else:
                dir_name = f"sae_l{args.layer_spec}"
        else:
            dir_name = f"base_l{args.layer_spec}"
        
        # Add experiment parameters in a clean format
        dir_name += f"_n{args.num_trials}"
        if args.channels:
            dir_name += f"_c{'-'.join(map(str, args.channels))}"
        if args.intervention_is_range:
            dir_name += f"_i{args.intervention_min:.1f}-{args.intervention_max:.1f}"
        else:
            dir_name += f"_i{args.intervention_fixed_value:.1f}"
        
        self.output_dir = os.path.join(args.output_dir, dir_name, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")

        # --- Pre-calculate the box maze pattern ---
        # This pattern is constant for the entire experiment run.
        # Logic adapted from create_box_maze:
        box_pattern_template = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 3, 0, 0, 0, 4, 0], # 3 is placeholder for entity1, 4 for entity2
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 2]  # 2 is player
        ])
        self.box_maze_pattern = box_pattern_template.copy()
        loc_entity1 = (box_pattern_template == 3)
        loc_entity2 = (box_pattern_template == 4) # entity2 is None for these experiments
        self.box_maze_pattern[loc_entity1] = self.args.target_entities[0] # Use the first (and only) target entity
        self.box_maze_pattern[loc_entity2] = 1 # Replace entity2 placeholder with corridor

        # Constants for applying the pattern (from create_custom_maze_sequence)
        self.maze_size = 7 # Specific to create_box_maze
        self.entity_mapping = { # Simplified for what box_maze uses
            2: {"type": "player", "color": None}, # Player
            self.args.target_entities[0]: {"type": "gem", "color": "blue"} # Assuming target is a gem-like entity
            # This mapping needs to be more robust if target_entities[0] can be keys etc.
            # For now, assuming target_entities[0] is a code like 3 (gem), 4 (blue_key) etc.
        }
        # Refine entity_mapping based on actual entity codes from ENTITY_CODE_DESCRIPTION
        # The key 'self.args.target_entities[0]' is an ENTITY_CODE (e.g., 3 for gem)
        # We need to map this code to its type and color for heist.ENTITY_TYPES and heist.ENTITY_COLORS
        entity_code = self.args.target_entities[0]
        entity_desc_name = ENTITY_CODE_DESCRIPTION.get(entity_code)
        if entity_desc_name:
            if "key" in entity_desc_name:
                color = entity_desc_name.split('_')[0]
                self.entity_mapping[entity_code] = {"type": "key", "color": color}
            elif "gem" in entity_desc_name:
                self.entity_mapping[entity_code] = {"type": "gem", "color": None} # Gems in heist don't have color types in ENTITY_COLORS
            # Add locks if they can be target_entities[0]
        # Ensure player is in mapping
        if 2 not in self.entity_mapping: # 2 is the player's code in the pattern
             self.entity_mapping[2] =  {"type": "player", "color": None}

        self.current_target_actual_grid_pos = None # Will be updated by _reset_env_to_box_maze

        # --- Create environments ONCE and store them ---
        self.envs = []
        for _ in range(self.args.num_parallel_envs):
            # Create a base venv. The custom state will be applied by _reset_env_to_box_maze.
            # The initial call to create_box_maze is mostly to get a correctly wrapped venv object.
            # The actual maze structure from this initial call might be immediately overridden.
             _, venv = create_box_maze(
                 entity1=self.args.target_entities[0], # This sets it up initially
                 entity2=None
             )
            # Store world_dim from one of the envs, assuming it's constant
             if not hasattr(self, 'world_dim'):
                 temp_state = heist.state_from_venv(venv,0)
                 self.world_dim = temp_state.world_dim # Should be 25 for heist
            
             self.envs.append(venv)

    def _reset_env_to_box_maze(self, env, channel_idx, trial_num):
        # Get initial state bytes from the current env to correctly initialize EnvState
        # This ensures EnvState has a valid underlying structure to modify.
        initial_state_bytes_for_setup = env.env.callmethod("get_state")[0]

        # Create the state-bytes-generating function for procgen
        state_bytes_generator_fn = self._generate_procgen_state_bytes_function(
            initial_state_bytes_for_setup, 
            self.box_maze_pattern, 
            self.world_dim # self.world_dim should be correctly set from __init__
        )
        
        # Get the actual state bytes for our custom maze
        state_bytes_to_set = state_bytes_generator_fn(0) # The '0' is a dummy seed

        # Use the environment's callmethod to set its state
        env.env.callmethod("set_state", [state_bytes_to_set])

        # The target position is now directly from the intervention_position argument
        if self.intervention_position and len(self.intervention_position) == 2:
            # IMPORTANT: Ensure intervention_position is in WORLD coordinates, not pattern coordinates.
            # If intervention_position is given in pattern's intuitive coordinates (y_pat, x_pat),
            # it needs to be converted to world coordinates here.
            # Assuming self.intervention_position IS ALREADY in world coordinates for now.
            # If it's (row, col) from the activation map, it needs to map to world grid.
            # For now, direct use:
            self.current_target_actual_grid_pos = (self.intervention_position[0], self.intervention_position[1])
            # print(f"[DEBUG Target Set Directly from --intervention_position]: {self.current_target_actual_grid_pos} (World Dim: {self.world_dim}x{self.world_dim}) for env {id(env)}")
        else:
            print(f"[WARNING] intervention_position is not properly set. current_target_actual_grid_pos may be incorrect or None.", flush=True)
            self.current_target_actual_grid_pos = None


        # Reset the environment to apply the new state and get the first observation.
        # procgen.ProcgenEnv.set_state doesn't return obs, so a reset is needed.
        with suppress_stderr(): # Mute warnings from this reset if any
            obs = env.reset()
        return obs

    def _get_intervention_value(self, trial_idx):
        """Get intervention value for a specific trial."""
        if self.args.intervention_is_range:
            # Linear interpolation between min and max values
            progress = trial_idx / (self.args.num_trials - 1)
            return self.args.intervention_min + progress * (self.args.intervention_max - self.args.intervention_min)
        return self.args.intervention_fixed_value

    def run_channel_experiment(self, channel_idx):
        print(f"[GIF_DEBUG run_channel_experiment] Channel: {channel_idx}, Save GIF arg: {self.args.save_debug_gif}")
        # Initialize GIF collection structures for channel 0 if enabled
        self.gif_target_for_current_trial_reached = False # For the specific GIF trial
        self.gif_frames_after_reach_count = 0 # For the specific GIF trial

        if self.args.save_debug_gif and channel_idx == 0:
            print("[GIF_DEBUG run_channel_experiment] Initializing self.debug_gif_frames = [] for Ch 0.")
            self.debug_gif_frames = []
        else:
            self.debug_gif_frames = None

        results = []
        successful_interventions_for_channel = 0 # Renamed for clarity
        prev_game_state_missing_count = 0
        
        # Ensure the detailed_results.csv for this channel has headers
        detailed_log_path = os.path.join(self.output_dir, f"detailed_results_channel_{channel_idx}.csv")
        with open(detailed_log_path, 'w') as f_detailed:
            f_detailed.write("channel,trial,intervention_value,success,steps_taken\\n")

        # Interventions log (target reached during trial)
        intervention_reach_log_path = os.path.join(self.output_dir, f"channel_{channel_idx}_intervention_reaches.csv")
        with open(intervention_reach_log_path, 'w') as f_reach_log:
            f_reach_log.write(INTERVENTION_REACHED_LOG_HEADER)


        # Initial reset for all parallel environments before any batch starts
        current_obs_list = [None] * self.args.num_parallel_envs
        for i in range(self.args.num_parallel_envs):
            obs_from_reset = self._reset_env_to_box_maze(self.envs[i], channel_idx, i) # `i` here is env_instance_idx
            
            # Store initial observation for GIF if applicable (first trial of first batch of channel 0)
            if self.debug_gif_frames is not None and i == 0: 
                print(f"[GIF_DEBUG _reset_env_to_box_maze] Ch {channel_idx}, EnvInstance {i}. Attempting initial frame.")
                initial_frame_to_store = helpers.prepare_frame_for_gif(obs_from_reset)
                if initial_frame_to_store is not None:
                    self.debug_gif_frames.append(initial_frame_to_store)
                    print(f"[GIF_DEBUG _reset_env_to_box_maze] Initial frame added. Total frames: {len(self.debug_gif_frames)}")
                else:
                    print(f"[GIF_DEBUG _reset_env_to_box_maze] Failed to process initial frame from reset.")
            current_obs_list[i] = obs_from_reset


        # Process trials in batches
        for batch_start_trial_idx in range(0, self.args.num_trials, self.args.num_parallel_envs):
            num_envs_this_batch = min(self.args.num_parallel_envs, self.args.num_trials - batch_start_trial_idx)
            
            # For each env instance, track if its *current assigned trial for this batch* has reached the target
            reached_target_flags_this_batch = [False] * num_envs_this_batch

            # Reset GIF control flags specifically if this batch contains the designated GIF trial
            # The GIF trial is always the absolute first trial (trial_idx 0) of channel 0.
            is_gif_trial_in_this_batch_and_env_slot = (channel_idx == 0 and batch_start_trial_idx == 0)
            if is_gif_trial_in_this_batch_and_env_slot: # For env_instance_idx = 0 of this batch
                self.gif_target_for_current_trial_reached = False
                self.gif_frames_after_reach_count = 0
                print(f"[GIF_DEBUG batch_start] GIF trial flags reset for Ch0, BatchStartIdx {batch_start_trial_idx}, EnvInstance 0.")

            # Observations for the current step, for envs active in this batch
            # current_obs_list was initialized before batch loop, and is updated within step loop after env reset.

            for step in range(self.args.max_steps):
                # Prepare observations for model input from active envs in this batch
                obs_tensors_for_model_step = []
                for i in range(num_envs_this_batch): # Iterate over env instances active in this batch
                    obs_data = current_obs_list[i] 
                    processed_obs = helpers.process_observation_for_model(obs_data, self.device)
                    obs_tensors_for_model_step.append(processed_obs)
                
                if not obs_tensors_for_model_step: # Should not happen if num_envs_this_batch > 0
                    break 
                
                obs_batch_for_model = torch.cat(obs_tensors_for_model_step, dim=0)

                # Get intervention values for each trial active in this batch
                # This means current_batch_intervention_values will have num_envs_this_batch items
                current_batch_intervention_values = []
                for i in range(num_envs_this_batch):
                    overall_trial_idx_for_env_i = batch_start_trial_idx + i
                    # Cycle through self.args.intervention_value list for the specific overall trial index
                    iv_list = self.args.intervention_value
                    current_batch_intervention_values.append(iv_list[overall_trial_idx_for_env_i % len(iv_list)])
                
                # Get actions from model with intervention (Revised)
                action_logits = None
                intervention_handle = None

                try:
                    # 1. Define the intervention configuration for SAE
                    # Assuming SAEInterventionExperiment expects a config for its set_intervention method.
                    # The channel_idx in this loop is the SAE feature/dimension.
                    value_for_this_model_step = current_batch_intervention_values[i]
                    intervention_config_for_sae_step = [{
                        "sae_feature_idx": channel_idx, 
                        "value": value_for_this_model_step,
                        "intervention_type": self.args.intervention_type if hasattr(self.args, 'intervention_type') and self.args.intervention_type else "sae_activation_add",
                        # "layer_name": self.experiment.layer_name, # SAEInterventionExperiment should know its target layer_name
                        # "position": self.intervention_position, # If SAEInterventionExperiment supports spatial on its features
                    }]

                    # 2. Set the intervention via self.experiment (SAEInterventionExperiment instance)
                    # This is a guess for the method name and behavior.
                    if hasattr(self.experiment, 'set_intervention'):
                        intervention_handle = self.experiment.set_intervention(intervention_config_for_sae_step)
                    elif hasattr(self.experiment, 'apply_intervention'): # Alternative common name
                        intervention_handle = self.experiment.apply_intervention(intervention_config_for_sae_step)
                    else:
                        # Fallback: If SAEInterventionExperiment doesn't have set_intervention,
                        # we might be misinterpreting its API. For now, proceed without explicit set, 
                        # assuming model call itself might be wrapped or implicitly handles it (less likely).
                        # This will likely lead to no intervention if set_intervention is the correct pattern.
                        print(f"[WARN] SAEInterventionExperiment does not have set_intervention or apply_intervention. Intervention may not be applied.")
                        pass # Proceed to model call, intervention might not be active

                    # 3. Run the model (self.model is set in __init__ to self.experiment.model)
                    # The hook applied by set_intervention (if successful) will modify activations.
                    if hasattr(self.model, 'parameters'): # Basic check if it's a PyTorch model
                        raw_model_output = self.model(obs_batch_for_model)
                    else:
                        raise AttributeError("self.model is not a callable PyTorch model or is not properly initialized.")

                    # Extract logits: Procgen models often return a list/tuple where logits are the first element or under a .logits attribute
                    if isinstance(raw_model_output, tuple) or isinstance(raw_model_output, list):
                        if hasattr(raw_model_output[0], 'logits'):
                            action_logits = raw_model_output[0].logits
                        else: # Assuming the first element itself is logits if no .logits attr
                            action_logits = raw_model_output[0]
                    elif hasattr(raw_model_output, 'logits'):
                        action_logits = raw_model_output.logits
                    else: # Direct output is assumed to be logits
                        action_logits = raw_model_output
                        
                finally:
                    # 4. Remove the hook if a handle was obtained
                    if intervention_handle and hasattr(intervention_handle, 'remove'):
                        intervention_handle.remove()
                    elif intervention_handle and hasattr(self.experiment, 'remove_intervention'): # Alternative pattern
                        self.experiment.remove_intervention(intervention_handle)

                if action_logits is None:
                    raise ValueError("Failed to obtain action_logits from model output.")

                # 5. Convert logits to actions
                probabilities = torch.nn.functional.softmax(action_logits, dim=-1)
                actions_for_step_tensor = torch.multinomial(probabilities, num_samples=1)
                actions_for_step = actions_for_step_tensor.cpu().numpy() # .reshape(-1, 1) is implicitly handled by multinomial output for single sample
                # Ensure it's (batch_size, 1) for the environment step loop
                if actions_for_step.ndim == 1:
                    actions_for_step = actions_for_step.reshape(-1,1)

                # Step each environment instance active in this batch
                temp_next_obs_list = [None] * num_envs_this_batch
                
                for i in range(num_envs_this_batch): # Iterate over env instances 0 to num_envs_this_batch-1
                    env_instance_idx = i # This is the index into self.envs and current_obs_list for this batch
                    overall_trial_idx = batch_start_trial_idx + env_instance_idx

                    # Corrected action slicing for the step method:
                    # actions_for_step has shape (num_envs_this_batch, 1)
                    # We need to pass an action of shape (1,) for the VecEnv of size 1.
                    # actions_for_step[i] will give a (1,) shaped array.
                    action_for_this_env = actions_for_step[env_instance_idx]
                    
                    next_obs_env_i, reward_arr, done_arr, info_list_for_env_i = self.envs[env_instance_idx].step(action_for_this_env)
                    temp_next_obs_list[env_instance_idx] = next_obs_env_i
                    
                    actual_done_from_env = done_arr[0]
                    info_dict_env_i = info_list_for_env_i[0] if info_list_for_env_i else {}

                    # --- GIF Frame Collection (only for env_instance_idx 0 of batch_start_trial_idx 0 of channel_idx 0) ---
                    if self.debug_gif_frames is not None and channel_idx == 0 and batch_start_trial_idx == 0 and env_instance_idx == 0:
                        if not self.gif_target_for_current_trial_reached or self.gif_frames_after_reach_count < 5:
                            frame_to_store = helpers.prepare_frame_for_gif(temp_next_obs_list[env_instance_idx]) # Use the very latest obs
                            if frame_to_store is not None:
                                self.debug_gif_frames.append(frame_to_store)
                                print(f"[GIF_DEBUG step_loop] GIF Frame. Ch{channel_idx},Trial{overall_trial_idx},Step{step}. TargetReached: {self.gif_target_for_current_trial_reached}, AfterCount: {self.gif_frames_after_reach_count}. Frames: {len(self.debug_gif_frames)}")
                                if self.gif_target_for_current_trial_reached:
                                    self.gif_frames_after_reach_count += 1
                            else:
                                print(f"[GIF_DEBUG step_loop] Failed to process GIF step frame for Trial {overall_trial_idx}, Step {step}.")
                        elif self.gif_frames_after_reach_count >= 5 :
                            print(f"[GIF_DEBUG step_loop] GIF collection stopped for Trial {overall_trial_idx} after {self.gif_frames_after_reach_count} post-reach frames.")


                    # --- Check for reaching target IN-STEP ---
                    # This check is for the trial currently assigned to self.envs[env_instance_idx]
                    if not reached_target_flags_this_batch[env_instance_idx] and self.current_target_actual_grid_pos is not None:
                        try:
                            # heist.state_from_venv expects the VecEnv and the sub-env index (0 for our size-1 VecEnvs)
                            current_env_state = heist.state_from_venv(self.envs[env_instance_idx], 0)
                            player_pos = current_env_state.mouse_pos 
                            if player_pos and not (np.isnan(player_pos[0]) or np.isnan(player_pos[1])):
                                current_player_grid_pos = (int(player_pos[0]), int(player_pos[1]))
                                if current_player_grid_pos == self.current_target_actual_grid_pos:
                                    reached_target_flags_this_batch[env_instance_idx] = True
                                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                                    target_entity_name = ENTITY_CODE_DESCRIPTION.get(self.args.target_entities[0], 'unknown_target')
                                    
                                    log_entry_reach = (f"{timestamp},{overall_trial_idx},{step},{channel_idx},"
                                                       f"{current_batch_intervention_values[env_instance_idx]},"
                                                       f"{target_entity_name},"
                                                       f"{self.intervention_position[0]},{self.intervention_position[1]},"
                                                       f"{self.current_target_actual_grid_pos[0]},{self.current_target_actual_grid_pos[1]}\\n")
                                    with open(intervention_reach_log_path, 'a') as f_reach_log:
                                        f_reach_log.write(log_entry_reach)
                                    print(f"[REACHED TARGET] Trial {overall_trial_idx} (EnvInst {env_instance_idx}), Ch {channel_idx}, Step {step}. Player@{current_player_grid_pos}, Target@{self.current_target_actual_grid_pos}")
                                    
                                    # GIF specific: if this is the designated GIF trial and target is now reached
                                    if channel_idx == 0 and batch_start_trial_idx == 0 and env_instance_idx == 0:
                                        if not self.gif_target_for_current_trial_reached: # Ensure this is the first time
                                            self.gif_target_for_current_trial_reached = True
                                            print(f"[GIF_DEBUG target_reached_event] GIF trial (Overall {overall_trial_idx}) target reached at step {step}. Will start collecting post-reach frames.")
                        except Exception as e_heist_state:
                            print(f"Warning: Error getting heist state for Trial {overall_trial_idx} (EnvInst {env_instance_idx}), Step {step}: {e_heist_state}")                            

                    # --- Process 'done' state for the trial assigned to this environment instance ---
                    if actual_done_from_env:
                        is_success = reached_target_flags_this_batch[env_instance_idx]
                        if is_success:
                            successful_interventions_for_channel += 1
                            print(f"[TRIAL DONE - SUCCESS] Trial {overall_trial_idx} (EnvInst {env_instance_idx}), Ch {channel_idx}. Steps: {step + 1}.")
                        else:
                            print(f"[TRIAL DONE - FAILURE] Trial {overall_trial_idx} (EnvInst {env_instance_idx}), Ch {channel_idx}. Steps: {step + 1}.")
                        
                        results.append({
                            'channel': channel_idx, 'trial': overall_trial_idx, 
                            'intervention_value': current_batch_intervention_values[env_instance_idx], 
                            'success': is_success, 'steps_taken': step + 1 
                        })
                        
                        # This env instance (self.envs[env_instance_idx]) has finished its assigned trial for this batch.
                        # Reset it for its *next potential trial* in a *future batch* OR if this was its last trial overall.
                        # The obs from this reset will be stored in current_obs_list[env_instance_idx] at the end of the inner loop.
                        obs_from_reset = self._reset_env_to_box_maze(self.envs[env_instance_idx], channel_idx, 
                                                                    env_instance_idx # Pass env_instance_idx (was trial_num)
                                                                    )
                        temp_next_obs_list[env_instance_idx] = obs_from_reset # Store reset obs for next step's input
                        reached_target_flags_this_batch[env_instance_idx] = False # Reset flag for this env instance's next trial

                        # If this was the GIF trial and it's now 'actual_done_from_env'
                        if channel_idx == 0 and batch_start_trial_idx == 0 and env_instance_idx == 0:
                            print(f"[GIF_DEBUG actual_done_event] GIF trial (Overall {overall_trial_idx}) is now actual_done_from_env. Forcing GIF stop if not already.")
                            self.gif_target_for_current_trial_reached = True 
                            self.gif_frames_after_reach_count = 5 # Force stop GIF collection

                        # Check for missing prev_game_state from info_dict_env_i
                        terminal_game_state = info_dict_env_i.get('prev_game_state')
                        if terminal_game_state is None:
                            prev_game_state_missing_count +=1
                
                # Update current_obs_list with all the next_obs from the step (or resets if done)
                for i in range(num_envs_this_batch):
                    current_obs_list[i] = temp_next_obs_list[i]

                # Check if all trials that were designated for this batch have naturally finished (actual_done_from_env)
                # This is an early exit from the step loop for this batch if all its assigned trials are done.
                # The number of trials processed in results for this batch:
                num_done_this_batch_step = sum(1 for res_entry in results if res_entry['trial'] >= batch_start_trial_idx and res_entry['trial'] < batch_start_trial_idx + num_envs_this_batch)
                if num_done_this_batch_step >= num_envs_this_batch:
                    print(f"[BATCH INFO] All {num_envs_this_batch} trials for batch starting at {batch_start_trial_idx} completed by step {step}. Moving to next batch.")
                    break # Break from step loop, move to next batch_start_trial_idx
            # End of step loop for the current batch
        # End of batch_start_trial_idx loop (all trials for the channel)

        # Save debug GIF (occurs once per channel, only if channel_idx is 0)
        if self.debug_gif_frames is not None: # Implies channel_idx == 0 and self.args.save_debug_gif
            if len(self.debug_gif_frames) > 0:
                print(f"[GIF_DEBUG mimsave] Saving GIF for Ch {channel_idx} with {len(self.debug_gif_frames)} frames.")
                gif_path = os.path.join(self.output_dir, f"debug_channel_{channel_idx}_trial_0.gif") 
                try:
                    imageio.mimsave(gif_path, self.debug_gif_frames, fps=10) # Increased fps
                    print(f"Saved debug GIF to {gif_path}")
                except Exception as e:
                    print(f"Error saving debug GIF: {e}")
            else:
                print(f"[GIF_DEBUG mimsave] No frames collected for Ch {channel_idx} GIF. Skipping save.")
            
            self.debug_gif_frames = None # Clear frames for this channel

        # Final success rate calculation and logging for the channel
        final_total_trials_run_for_channel = len(results)

        if final_total_trials_run_for_channel > 0:
            # successful_interventions_for_channel already counted
            success_rate = successful_interventions_for_channel / final_total_trials_run_for_channel
        else:
            success_rate = 0
            if self.args.num_trials > 0:
                 print(f"Warning: No trials were logged to results for channel {channel_idx}, but num_trials was {self.args.num_trials}")

        print(f"CHANNEL {channel_idx} SUMMARY: Success Rate = {success_rate:.4f} ({successful_interventions_for_channel}/{final_total_trials_run_for_channel} successful trials)")
        
        # Append all results for this channel to its detailed_results.csv
        # This is slightly redundant if we write per trial, but good for a consolidated check / if per-trial write failed.
        # The current logic writes to results list, then this loop writes it all.
        for res_entry in results:
             with open(detailed_log_path, 'a') as f_detailed: # Append mode now, header written once.
                f_detailed.write(f"{res_entry['channel']},{res_entry['trial']},{res_entry['intervention_value']},{res_entry['success']},{res_entry['steps_taken']}\\n")

        if prev_game_state_missing_count > 0:
             print(f"Info: For channel {channel_idx}, 'prev_game_state' was not found in info dict for {prev_game_state_missing_count} / {final_total_trials_run_for_channel} trials where env signaled done.")
        
        return {
            "channel": channel_idx,
            "success_rate": success_rate,
            "successful_interventions": successful_interventions_for_channel,
            "total_trials": final_total_trials_run_for_channel 
        }

    def _get_target_env_pos(self, state_obj):
        """Convert intervention position from activation coordinates to environment coordinates."""
        act_h, act_w = None, None
        
        if hasattr(state_obj, 'world_dim'): # For HeistState
            env_h, env_w = state_obj.world_dim, state_obj.world_dim
        elif hasattr(state_obj, 'env_state') and hasattr(state_obj.env_state, 'world_dim_pair_y') and hasattr(state_obj.env_state, 'world_dim_pair_x'): # For procgen GameState
            env_h = state_obj.env_state.world_dim_pair_y
            env_w = state_obj.env_state.world_dim_pair_x
        else:
            print(f"Warning: _get_target_env_pos received an unknown state object type: {type(state_obj)}")
            return None

        # Map layer names/numbers to their activation dimensions
        if self.args.is_sae:
            layer_number = int(self.args.layer_spec)
            if layer_number == 0:
                act_h, act_w = 64, 64  # Input: 64×64×3
            elif layer_number == 2:
                act_h, act_w = 64, 64  # After Conv Block 1: 32×32×16
            elif layer_number == 4:
                act_h, act_w = 32, 32  # After Conv Block 2: 16×16×32
            elif layer_number == 5:
                act_h, act_w = 32, 32  # After Conv Block 2: 16×16×32
            elif layer_number == 6:
                act_h, act_w = 16, 16  # After Conv Block 3: 8×8×32
            elif layer_number == 8:
                act_h, act_w = 8, 8    # After Conv Block 4: 4×4×32

        if act_h is not None and act_w is not None and act_h > 0 and act_w > 0:
            scale_y = env_h / act_h
            scale_x = env_w / act_w
            return (int(self.intervention_position[0] * scale_y), 
                   int(self.intervention_position[1] * scale_x))
        return None

    def run_experiments(self):
        """Run experiments for all specified channels."""
        # Determine which channels to run
        if self.args.channels:
            channels_to_run = self.args.channels
        else:
            # Use hidden_channels from the SAE object
            if self.args.is_sae:
                channels_to_run = range(self.experiment.sae.hidden_channels)
            else:
                channels_to_run = range(self.experiment.num_channels)

        # Run experiments for each channel
        all_results = []
        for channel_idx in tqdm(channels_to_run, desc="Processing channels"):
            channel_results = self.run_channel_experiment(channel_idx)
            all_results.append(channel_results)

        # Save results to CSV
        self._save_results(all_results)

        # Generate visualizations
        self._generate_visualization(all_results)
        self._generate_entity_distribution_plot()

        # --- Close environments after all experiments are done ---
        self.close_envs()

    def _save_results(self, all_results):
        """Save experiment results to CSV file."""
        # Create DataFrame for summary statistics
        summary_data = []
        for result in all_results:
            summary_data.append({
                'channel': result['channel'],
                'success_rate': result['success_rate'],
                'successful_interventions': result['successful_interventions'],
                'total_trials': result['total_trials']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, 'summary_results.csv'), index=False)

        # Create DataFrame for detailed results
        detailed_data = []
        for result in all_results:
            detailed_data.extend(result['results'])
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(os.path.join(self.output_dir, 'detailed_results.csv'), index=False)

    def _generate_visualization(self, all_results):
        """Generate visualization of experiment results."""
        # Create DataFrame for visualization
        data = []
        for result in all_results:
            data.append({
                'Channel': result['channel'],
                'Success Rate': result['success_rate']
            })
        
        df = pd.DataFrame(data)

        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Channel', y='Success Rate')
        plt.title('Intervention Success Rate by Channel')
        plt.xlabel('Channel Index')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'success_rate_by_channel.png'))
        plt.close()

    def _generate_entity_distribution_plot(self):
        """Generate and save entity distribution plot."""
        try:
            detailed_csv_path = os.path.join(self.output_dir, 'detailed_results.csv')
            if not os.path.exists(detailed_csv_path) or os.path.getsize(detailed_csv_path) == 0:
                print(f"Info: Detailed results CSV '{detailed_csv_path}' is empty or not found. Skipping intervention value distribution plot.")
                return

            detailed_df = pd.read_csv(detailed_csv_path)
            
            if detailed_df.empty or 'success' not in detailed_df.columns or not detailed_df['success'].any():
                print("Info: No successful interventions found in detailed results. Skipping intervention value distribution plot.")
                return

            successful_interventions_df = detailed_df[detailed_df['success']]
            if successful_interventions_df.empty or 'intervention_value' not in successful_interventions_df.columns:
                print("Info: No 'intervention_value' column found for successful interventions. Skipping plot.")
                return

            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot distribution of intervention values for successful interventions
            sns.histplot(data=successful_interventions_df, 
                        x='intervention_value', 
                        bins=20,
                        kde=True)
            
            plt.title('Distribution of Successful Intervention Values')
            plt.xlabel('Intervention Value')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'intervention_value_distribution.png'))
            plt.close()
            
            print(f"Entity distribution plot saved to: {os.path.join(self.output_dir, 'intervention_value_distribution.png')}")
        except Exception as e:
            print(f"Warning: Could not generate entity distribution plot: {e}")

    def close_envs(self):
        """Close all environments."""
        print("Closing environments...")
        for env in self.envs:
            try:
                env.close()
            except Exception as e:
                print(f"Error closing environment: {e}")
        print("Environments closed.")

    def _generate_procgen_state_bytes_function(self, sample_initial_state_bytes, pattern_to_set, world_dim_to_use):
        """
        Returns a function that, when called, generates and returns the state_bytes
        for a Heist environment configured with the given pattern.
        Args:
            sample_initial_state_bytes: Bytes from an existing env, used to init EnvState.
            pattern_to_set: The numpy array defining the maze structure.
            world_dim_to_use: The dimension of the game world (e.g., 25 for 25x25).
        """
        # Ensure pattern_to_set is a numpy array
        if not isinstance(pattern_to_set, np.ndarray):
            pattern_to_set = np.array(pattern_to_set)

        if pattern_to_set.shape != (self.maze_size, self.maze_size):
            raise ValueError(f"Pattern shape {pattern_to_set.shape} must be ({self.maze_size},{self.maze_size})")

        # entity_mapping, maze_size, world_dim are available from self

        def actual_procgen_seq_fn(seed): # seed might be unused if pattern is fixed
            # Create a new heist.EnvState instance using the sample initial state bytes.
            # This gives us a valid EnvState object to start modifying.
            temp_state = heist.EnvState(sample_initial_state_bytes)

            # Ensure world_dim_to_use matches what temp_state believes, or prefer one.
            # For now, we trust world_dim_to_use passed in, which should match self.world_dim
            # and temp_state.world_dim if everything is consistent.
            if temp_state.world_dim != world_dim_to_use:
                print(f"Warning: world_dim_to_use ({world_dim_to_use}) mismatches temp_state.world_dim ({temp_state.world_dim}). Using world_dim_to_use.")

            # Calculate center position to place the maze pattern within the world_dim grid
            middle_y = world_dim_to_use // 2
            middle_x = world_dim_to_use // 2
            start_y_world = middle_y - self.maze_size // 2 # Top-left Y of pattern in world grid
            start_x_world = middle_x - self.maze_size // 2 # Top-left X of pattern in world grid

            # Helper to convert y-coordinate (pattern's 0 is bottom, grid's 0 is top)
            def convert_y_coord_pattern_to_world(y_pattern_intuitive, maze_size_val=self.maze_size):
                return (maze_size_val - 1) - y_pattern_intuitive

            # Create walls everywhere by default in the full world grid
            new_grid_world = np.full((world_dim_to_use, world_dim_to_use), heist.BLOCKED)

            player_world_pos = None # Store (world_y, world_x)
            entities_to_place = [] # Store {"type_code":..., "color_code":..., "position_world": (y,x)}

            for intuitive_i_pattern in range(self.maze_size): # y in pattern (0=bottom)
                for intuitive_j_pattern in range(self.maze_size): # x in pattern (0=left)
                    value_in_pattern = pattern_to_set[intuitive_i_pattern, intuitive_j_pattern]

                    # Convert pattern's intuitive Y to pattern's actual Y (0=top)
                    actual_y_pattern = convert_y_coord_pattern_to_world(intuitive_i_pattern)
                    
                    # Calculate absolute position in the world grid
                    world_grid_y = start_y_world + actual_y_pattern
                    world_grid_x = start_x_world + intuitive_j_pattern

                    if 0 <= world_grid_y < world_dim_to_use and 0 <= world_grid_x < world_dim_to_use:
                        if value_in_pattern == 1:  # Open corridor
                            new_grid_world[world_grid_y, world_grid_x] = heist.EMPTY
                        elif value_in_pattern >= 2:  # Entity or player
                            new_grid_world[world_grid_y, world_grid_x] = heist.EMPTY # Mark as walkable
                            
                            if value_in_pattern == 2:  # Player placeholder in pattern
                                player_world_pos = (world_grid_y, world_grid_x)
                            elif value_in_pattern in self.entity_mapping:
                                entity_data = self.entity_mapping[value_in_pattern]
                                entity_type_code = heist.ENTITY_TYPES.get(entity_data["type"])
                                entity_color_code = heist.ENTITY_COLORS.get(entity_data["color"]) if entity_data["color"] else None
                                if entity_type_code is not None:
                                    entities_to_place.append({
                                        "type_code": entity_type_code,
                                        "color_code": entity_color_code,
                                        "position_world": (world_grid_y, world_grid_x)
                                    })
                    else:
                        # This should not happen if world_dim >= maze_size and centered correctly
                        print(f"Warning: Calculated world position ({world_grid_y}, {world_grid_x}) is outside world_dim ({world_dim_to_use})")


            temp_state.set_grid(new_grid_world)
            temp_state.remove_all_entities() # Clear any previous entities

            if player_world_pos:
                temp_state.set_mouse_pos(player_world_pos[0], player_world_pos[1])
            else:
                # Fallback: if no player (2) in pattern, place randomly in an empty cell of the maze area
                # This part needs to be careful about coordinates (world vs pattern)
                # For simplicity, let's assume player (2) is always in the pattern for now.
                # If not, this would need a robust way to find a valid empty spot.
                print("Warning: Player (code 2) not found in pattern. Player position will be default.", flush=True)


            for entity_info in entities_to_place:
                temp_state.set_entity_position(
                    entity_info["type_code"],
                    entity_info["color_code"],
                    entity_info["position_world"][0],
                    entity_info["position_world"][1]
                )
            
            return temp_state.state_bytes
        
        return actual_procgen_seq_fn

def main():
    args = parse_args()
    experiment = BatchedInterventionExperiment(args)
    experiment.run_experiments()

if __name__ == "__main__":
    main() 