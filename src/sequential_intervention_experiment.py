import argparse
import os
import torch
import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm
import datetime
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from procgen import ProcgenEnv # Needed for venv.env
from utils import heist # Needed for state_from_venv

from sae_spatial_intervention import SAEInterventionExperiment, ordered_layer_names
from utils.environment_modification_experiments import create_custom_maze_sequence
from utils import helpers
from utils.create_intervention_mazes import create_sequential_maze

ENTITY_CODE_DESCRIPTION = {
    3: "gem",
    4: "blue_key",
    5: "green_key",
    6: "red_key",
}
ENTITY_DESCRIPTION_ORDER = ["blue_key", "green_key", "red_key", "gem"]

# Log header for visit events
VISIT_LOG_HEADER = "Timestamp,Trial,Step,VisitedEntityType,PlayerPosY,PlayerPosX,ATZGridPosY,ATZGridPosX,ActiveHotspotsCount\\n"

def parse_args():
    parser = argparse.ArgumentParser(description="Run value-driven sequential targeting experiments with SAE interventions.")

    # Model and Layer Configuration
    parser.add_argument("--model_path", type=str, default="../model_interpretable.pt", help="Path to the base model checkpoint.")
    parser.add_argument("--sae_checkpoint_path", type=str, required=True, help="Path to the SAE checkpoint.")
    parser.add_argument("--layer_number", type=int, required=True, help="SAE layer number (from sae_cnn.ordered_layer_names).")
    parser.add_argument("--target_channel", type=int, required=True, help="The specific channel index in the SAE to intervene on.")

    # Activation Values for Entities (Critical Arguments)
    parser.add_argument("--val_blue_key", type=float, required=True, help="Activation value for Blue Key hotspot.")
    parser.add_argument("--val_green_key", type=float, required=True, help="Activation value for Green Key hotspot.")
    parser.add_argument("--val_red_key", type=float, required=True, help="Activation value for Red Key hotspot.")
    parser.add_argument("--val_gem", type=float, required=True, help="Activation value for Gem hotspot.")

    # Experiment Parameters
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run.")
    parser.add_argument("--max_steps", type=int, default=150, help="Maximum number of steps per trial.")
    parser.add_argument("--intervention_radius", type=int, default=0, help="Radius for the intervention patch (0 for single point).")
    
    # ATZ Locations (Grid Coordinates)
    parser.add_argument("--atz_blue_pos", type=str, default="6,0", help="Grid position 'y,x' for Blue Key ATZ.")
    parser.add_argument("--atz_green_pos", type=str, default="6,2", help="Grid position 'y,x' for Green Key ATZ.")
    parser.add_argument("--atz_red_pos", type=str, default="6,4", help="Grid position 'y,x' for Red Key ATZ.")
    parser.add_argument("--atz_gem_pos", type=str, default="6,6", help="Grid position 'y,x' for Gem ATZ.")
    parser.add_argument("--player_start_pos", type=str, default="0,3", help="Player start position 'y,x'.")


    # Output Configuration
    parser.add_argument("--output_dir", type=str, default="sequential_targeting_results", help="Directory to save results (CSV, logs, and GIFs).")
    parser.add_argument("--save_gif_for_first_n_trials", type=int, default=1, help="Save a GIF of the first N successful/failed trials (0 to disable).")

    args = parser.parse_args()

    # Store activation values in a dictionary
    args.entity_values = {
        "blue_key": args.val_blue_key,
        "green_key": args.val_green_key,
        "red_key": args.val_red_key,
        "gem": args.val_gem,
    }

    # Parse ATZ positions
    def parse_pos(pos_str):
        parts = pos_str.split(',')
        return int(parts[0]), int(parts[1])

    args.atz_positions_grid = {
        "blue_key": parse_pos(args.atz_blue_pos),
        "green_key": parse_pos(args.atz_green_pos),
        "red_key": parse_pos(args.atz_red_pos),
        "gem": parse_pos(args.atz_gem_pos),
    }
    args.player_start_pos_grid = parse_pos(args.player_start_pos)

    return args


def map_grid_to_activation_coords(grid_pos, grid_dims, act_dims):
    """Maps coordinates from maze grid to activation map."""
    grid_h, grid_w = grid_dims
    act_h, act_w = act_dims
    
    act_y = int(round(grid_pos[0] * (act_h / float(grid_h))))
    act_x = int(round(grid_pos[1] * (act_w / float(grid_w))))
    
    # Clamp to ensure within activation map bounds
    act_y = max(0, min(act_y, act_h - 1))
    act_x = max(0, min(act_x, act_w - 1))
    return act_y, act_x

def get_activation_dimensions(sae_experiment_instance):
    """
    Gets the spatial dimensions (H', W') of the SAE hidden activations.
    This is done by performing a dummy forward pass if necessary, or inspecting the SAE.
    Assumes SAEs are 1x1 convolutions, so H', W' match the input spatial dims to the SAE.
    """
    # The hook is on the output of the base model layer.
    # Let's get the shape of this output, which is the input to the SAE.
    dummy_input_shape = None
    if sae_experiment_instance.layer_name == 'conv1a': # Output of conv_blocks.0 
        dummy_input_shape = (1, 16, 32, 32) 
    elif sae_experiment_instance.layer_name == 'conv2b': # Output of conv_blocks.1
        dummy_input_shape = (1, 32, 16, 16)
    elif sae_experiment_instance.layer_name == 'conv3a': # Output of conv_blocks.2
        dummy_input_shape = (1, 32, 8, 8) 
    elif sae_experiment_instance.layer_name == 'conv4a': # Output of conv_blocks.3
        dummy_input_shape = (1, 32, 4, 4)
    else: # Fallback: try to infer from a dummy pass if model and SAE are loaded
        try:
            # A bit hacky: create a dummy tensor that could be an output of a conv layer
            # This assumes the SAE can handle a generic tensor of some plausible shape.
            # We really need the output shape of `self.module`
            # For now, let's use known Procgen Impala CNN layer output shapes.
            # This part might need adjustment if SAEs are on other layers or architectures.
            # Example: If self.module is conv_seqs.2.res_block1.conv1 (output of conv3a)
            # The input to SAE would be (batch, channels, H, W)
            # Common shapes for Procgen Heist agent conv layers:
            # After conv_block1 (layer named 'conv_seqs.0.conv'): (B, 16, 32, 32)
            # After conv_block2 (layer named 'conv_seqs.1.conv'): (B, 32, 16, 16)
            # After conv_block3 (layer named 'conv_seqs.2.conv'): (B, 32, 8, 8)
            # After conv_block4 (layer named 'conv_seqs.3.conv'): (B, 32, 4, 4)
             pass # Covered by above specific cases

        except Exception as e:
            print(f"Warning: Could not reliably determine activation dimensions dynamically: {e}")
            # Default to a common small size if everything else fails
            return (8,8) # Fallback

    if dummy_input_shape:
        return dummy_input_shape[2], dummy_input_shape[3]
    
    # If targeting intermediate ResBlock convs, it's more complex.
    # For now, assuming SAEs are on the main block outputs mentioned above.
    # ordered_layer_names helps:
    # 2: conv_blocks.0.conv (input to SAE for "conv1a" SAEs in paper) -> 32x32 for Heist
    # 5: conv_blocks.1.conv (input to SAE for "conv2b" SAEs) -> 16x16
    # 6: conv_blocks.2.conv (input to SAE for "conv3a" SAEs) -> 8x8
    # 8: conv_blocks.3.conv (input to SAE for "conv4a" SAEs) -> 4x4
    # The SAE object itself should know its configuration.
    # However, the `position` is in the coordinate system of the SAE's feature map.
    # For ConvSAE, feature_map_size is stored.

    if hasattr(sae_experiment_instance.sae, 'cfg') and hasattr(sae_experiment_instance.sae.cfg, 'feature_map_size'):
         # Assuming feature_map_size is (H, W) or a single int if H=W
        size = sae_experiment_instance.sae.cfg.feature_map_size
        if isinstance(size, tuple) and len(size) == 2:
            return size
        elif isinstance(size, int):
            return (size, size)

    print(f"Falling back on specific known dimensions for layer {sae_experiment_instance.layer_number} ({sae_experiment_instance.layer_name})")
    if sae_experiment_instance.layer_number == 2: return (32,32) # conv1a output
    if sae_experiment_instance.layer_number == 5: return (16,16) # conv2b output
    if sae_experiment_instance.layer_number == 6: return (8,8)   # conv3a output
    if sae_experiment_instance.layer_number == 8: return (4,4)   # conv4a output
    
    raise ValueError(f"Could not determine activation dimensions for SAE layer {sae_experiment_instance.layer_number} ('{sae_experiment_instance.layer_name}'). Please check `get_activation_dimensions`.")

def create_debug_maze_visualization(venv, output_dir, timestamp_str):
    """
    Creates a debug visualization of the maze layout showing entity positions and paths.
    """
    try:
        # Get the initial state
        current_state = heist.state_from_venv(venv, 0)
        
        # Create a visualization frame
        frame = venv.render("rgb_array")
        
        # Add text annotations for entity positions
        for entity_type, entity_code in ENTITY_CODE_DESCRIPTION.items():
            entity_pos = current_state.get_entity_position(entity_code)
            if entity_pos:
                # Convert grid coordinates to pixel coordinates (approximate)
                pixel_y = entity_pos[0] * 32  # Assuming 32x32 pixels per grid cell
                pixel_x = entity_pos[1] * 32
                
                # Add text to the frame
                cv2.putText(
                    frame,
                    f"{entity_type}",
                    (pixel_x + 5, pixel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        # Save the debug visualization
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_filename = os.path.join(debug_dir, f"maze_layout_{timestamp_str}.png")
        cv2.imwrite(debug_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"\nDebug maze layout saved to: {debug_filename}")
        
        # Also save as GIF with a few frames to show the layout
        gif_frames = [frame] * 30  # 30 frames at 10fps = 3 seconds
        gif_filename = os.path.join(debug_dir, f"maze_layout_{timestamp_str}.gif")
        imageio.mimsave(gif_filename, gif_frames, fps=10)
        print(f"Debug maze layout GIF saved to: {gif_filename}")
        
    except Exception as e:
        print(f"Warning: Could not create debug visualization: {e}")

def create_atz_position_visualization(args, act_h, act_w, output_dir, timestamp_str):
    """
    Creates a visualization showing ATZ positions in both grid space and activation space.
    """
    try:
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Grid space visualization (left subplot)
        ax1.set_title('ATZ Positions in Grid Space')
        ax1.set_xlim(-0.5, 6.5)  # 7x7 grid
        ax1.set_ylim(-0.5, 6.5)
        ax1.invert_yaxis()  # Match Procgen's coordinate system
        ax1.grid(True)
        ax1.set_xticks(range(7))
        ax1.set_yticks(range(7))
        
        # Activation space visualization (right subplot)
        ax2.set_title('ATZ Positions in Activation Space')
        ax2.set_xlim(-0.5, act_w - 0.5)
        ax2.set_ylim(-0.5, act_h - 0.5)
        ax2.invert_yaxis()  # Match Procgen's coordinate system
        ax2.grid(True)
        ax2.set_xticks(range(act_w))
        ax2.set_yticks(range(act_h))
        
        # Colors for different entities
        colors = {
            'blue_key': 'blue',
            'green_key': 'green',
            'red_key': 'red',
            'gem': 'yellow'
        }
        
        # Plot positions in both spaces
        for entity_type in ENTITY_DESCRIPTION_ORDER:
            # Grid space
            grid_pos = args.atz_positions_grid[entity_type]
            grid_y, grid_x = grid_pos
            ax1.scatter(grid_x, grid_y, c=colors[entity_type], s=100, label=entity_type)
            ax1.text(grid_x + 0.1, grid_y + 0.1, entity_type, fontsize=8)
            
            # Activation space
            act_pos = map_grid_to_activation_coords(grid_pos, (7, 7), (act_h, act_w))
            act_y, act_x = act_pos
            ax2.scatter(act_x, act_y, c=colors[entity_type], s=100, label=entity_type)
            ax2.text(act_x + 0.1, act_y + 0.1, entity_type, fontsize=8)
        
        # Add player start position to grid space
        start_y, start_x = args.player_start_pos_grid
        ax1.scatter(start_x, start_y, c='black', marker='*', s=150, label='Player Start')
        ax1.text(start_x + 0.1, start_y + 0.1, 'Start', fontsize=8)
        
        # Add legends
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add activation values as text
        value_text = "Activation Values:\n"
        for entity_type in ENTITY_DESCRIPTION_ORDER:
            value_text += f"{entity_type}: {args.entity_values[entity_type]:.2f}\n"
        fig.text(0.5, 0.01, value_text, ha='center', fontsize=10)
        
        # Save the visualization
        plt.tight_layout()
        viz_dir = os.path.join(output_dir, "debug")
        os.makedirs(viz_dir, exist_ok=True)
        viz_filename = os.path.join(viz_dir, f"atz_positions_{timestamp_str}.png")
        plt.savefig(viz_filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"ATZ position visualization saved to: {viz_filename}")
        
    except Exception as e:
        print(f"Warning: Could not create ATZ position visualization: {e}")

def main():
    args = parse_args()
    print("Starting Sequential Targeting Experiment with args:", args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log file for detailed visit events
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_layer_name = ordered_layer_names[args.layer_number].replace('.', '_').replace('/', '_')
    visit_log_filename = os.path.join(args.output_dir, f"sequential_visit_log_{safe_layer_name}_ch{args.target_channel}_{timestamp_str}.csv")
    try:
        with open(visit_log_filename, 'w') as f_log:
            f_log.write(VISIT_LOG_HEADER)
        print(f"Detailed visit events will be logged to: {visit_log_filename}")
    except Exception as e_log_init:
        print(f"Warning: Could not create visit log file '{visit_log_filename}'. Logging disabled. Error: {e_log_init}")
        visit_log_filename = None

    experiment = SAEInterventionExperiment(
        model_path=args.model_path,
        sae_checkpoint_path=args.sae_checkpoint_path,
        layer_number=args.layer_number,
        device=device
    )

    act_h, act_w = get_activation_dimensions(experiment)
    print(f"SAE Activation Dimensions (H', W'): ({act_h}, {act_w})")
    
    # Create and save debug visualization of the maze layout
    print("\nCreating debug visualization of maze layout...")
    observations, venv = create_sequential_maze()
    create_debug_maze_visualization(venv, args.output_dir, timestamp_str)
    venv.close()
    
    # Create visualization of ATZ positions in both spaces
    print("\nCreating ATZ position visualization...")
    create_atz_position_visualization(args, act_h, act_w, args.output_dir, timestamp_str)
    
    # First run control experiment with all real entities
    print("\nRunning control experiment with all real entities...")
    control_results = []
    
    for trial_num in tqdm(range(args.num_trials), desc="Control Trials"):
        observations, venv = create_sequential_maze()
        observation = venv.reset()
        
        visited_entities = set()
        visit_sequence = []
        frames = []
        save_gif = trial_num < args.save_gif_for_first_n_trials

        for step_num in range(args.max_steps):
            if save_gif:
                frames.append(venv.render("rgb_array"))

            current_state = heist.state_from_venv(venv, 0)
            player_pos = None
            if current_state.mouse_pos is not None and not (np.isnan(current_state.mouse_pos[0]) or np.isnan(current_state.mouse_pos[1])):
                player_pos = (int(current_state.mouse_pos[0]), int(current_state.mouse_pos[1]))

            if player_pos:
                # Check for entity visits
                for entity_type, entity_code in ENTITY_CODE_DESCRIPTION.items():
                    if entity_type not in visited_entities:
                        entity_pos = current_state.get_entity_position(entity_code)
                        if entity_pos and player_pos == entity_pos:
                            visited_entities.add(entity_type)
                            visit_sequence.append(entity_type)
                            tqdm.write(f"  Control Trial {trial_num+1}, Step {step_num+1}: Reached {entity_type}")
                            
                            # Log the visit
                            if visit_log_filename:
                                try:
                                    log_ts = datetime.datetime.now().isoformat()
                                    log_entry = (
                                        f"{log_ts},{trial_num+1},{step_num+1},{entity_type},"
                                        f"{player_pos[0]},{player_pos[1]},"
                                        f"{entity_pos[0]},{entity_pos[1]},0\\n"  # 0 for control experiment
                                    )
                                    with open(visit_log_filename, 'a') as log_f:
                                        log_f.write(log_entry)
                                except Exception as e_log:
                                    tqdm.write(f"    [Warning] Failed to write to visit log: {e_log}")
                            break

            if len(visited_entities) == len(ENTITY_DESCRIPTION_ORDER):
                tqdm.write(f"  Control Trial {trial_num+1}: All entities visited.")
                break

            # Agent takes a step
            obs_np = observation[0]['rgb']
            obs_tensor = torch.tensor(helpers.observation_to_rgb(obs_np), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                outputs = experiment.model(obs_tensor)

            if hasattr(outputs, 'logits'): logits = outputs.logits
            elif isinstance(outputs, tuple) and hasattr(outputs[0], 'logits'): logits = outputs[0].logits
            else: raise ValueError("Could not extract logits from model output.")
            
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            action = torch.multinomial(probabilities, num_samples=1).squeeze(-1).cpu().numpy()
            observation, reward, done, info = venv.step(action)

            if done:
                tqdm.write(f"  Control Trial {trial_num+1}: Episode ended by 'done' flag at step {step_num+1}.")
                break

        control_results.append({
            "trial_num": trial_num + 1,
            "visit_sequence": list(visit_sequence),
            "num_visited": len(visited_entities),
            "completed_sequence": visit_sequence == ENTITY_DESCRIPTION_ORDER,
            "steps_taken": step_num + 1,
            "is_control": True  # Mark as control experiment
        })

        if save_gif and frames:
            try:
                gif_dir = os.path.join(args.output_dir, "gifs", "control")
                os.makedirs(gif_dir, exist_ok=True)
                status = "success" if visit_sequence == ENTITY_DESCRIPTION_ORDER else "fail"
                gif_filename = os.path.join(gif_dir, f"control_trial_{trial_num+1}_{status}_seq_{'_'.join(visit_sequence)}.gif")
                imageio.mimsave(gif_filename, frames, fps=10)
                tqdm.write(f"    Saved control GIF to {gif_filename}")
            except Exception as e_gif:
                tqdm.write(f"    Error saving control GIF for trial {trial_num+1}: {e_gif}")

        venv.close()

    # Save control results
    control_df = pd.DataFrame(control_results)
    control_filename = os.path.join(args.output_dir, f"control_results_{safe_layer_name}_{timestamp_str}.csv")
    control_df.to_csv(control_filename, index=False)
    print(f"\nControl experiment results saved to: {control_filename}")

    # Calculate control experiment statistics
    num_successful_control = control_df["completed_sequence"].sum()
    percent_successful_control = (num_successful_control / args.num_trials) * 100 if args.num_trials > 0 else 0
    
    print(f"\n--- Control Experiment Summary ---")
    print(f"Total Trials: {args.num_trials}")
    print(f"Successful Sequences: {num_successful_control}")
    print(f"Success Rate: {percent_successful_control:.2f}%")
    
    # Print most common sequences in control
    print("\nMost common sequences in control:")
    sequence_counts = control_df['visit_sequence'].astype(str).value_counts().nlargest(5)
    for seq_str, count in sequence_counts.items():
        print(f"  {seq_str}: {count} times")

    # Now run the intervention experiments
    print("\nRunning intervention experiments...")
    grid_dims = (7, 7) # Our maze is 7x7
    mapped_atz_positions = {}
    for entity_type, grid_pos in args.atz_positions_grid.items():
        mapped_atz_positions[entity_type] = map_grid_to_activation_coords(grid_pos, grid_dims, (act_h, act_w))
        print(f"ATZ {entity_type}: Grid {grid_pos} -> Activation Map {mapped_atz_positions[entity_type]}")

    trial_results = []
    gif_saved_count = 0

    for trial_num in tqdm(range(args.num_trials), desc="Trials"):
        _, venv = create_sequential_maze()
        observation = venv.reset()
        
        visited_atz_types_this_trial = set()
        visit_sequence_this_trial = []
        
        frames_this_trial = []
        save_this_trial_gif = trial_num < args.save_gif_for_first_n_trials

        handle = None # Initialize handle for hook

        try:
            # Register Hook
            if hasattr(experiment, '_hook_sae_activations'):
                hook_func = experiment._hook_sae_activations
            else:
                 raise AttributeError("SAEExperiment object missing '_hook_sae_activations' method.")
            handle = experiment.module.register_forward_hook(hook_func)

            for step_num in range(args.max_steps):
                if save_this_trial_gif:
                    frames_this_trial.append(venv.render("rgb_array"))

                # --- Determine current player position and check for ATZ visits ---
                current_state = heist.state_from_venv(venv, 0) # Assuming heist.py is in utils
                player_pos_grid = None
                if current_state.mouse_pos is not None and not (np.isnan(current_state.mouse_pos[0]) or np.isnan(current_state.mouse_pos[1])):
                    player_pos_grid = (int(current_state.mouse_pos[0]), int(current_state.mouse_pos[1]))

                if player_pos_grid:
                    for entity_type, atz_grid_pos in args.atz_positions_grid.items():
                        if entity_type not in visited_atz_types_this_trial and player_pos_grid == atz_grid_pos:
                            visited_atz_types_this_trial.add(entity_type)
                            visit_sequence_this_trial.append(entity_type)
                            tqdm.write(f"  Trial {trial_num+1}, Step {step_num+1}: Reached ATZ for {entity_type} at {atz_grid_pos}")
                            
                            if visit_log_filename:
                                try:
                                    log_ts = datetime.datetime.now().isoformat()
                                    active_hotspots_count = len(ENTITY_DESCRIPTION_ORDER) - len(visited_atz_types_this_trial)
                                    log_entry = (
                                        f"{log_ts},{trial_num+1},{step_num+1},{entity_type},"
                                        f"{player_pos_grid[0]},{player_pos_grid[1]},"
                                        f"{atz_grid_pos[0]},{atz_grid_pos[1]},{active_hotspots_count}\\n"
                                    )
                                    with open(visit_log_filename, 'a') as log_f:
                                        log_f.write(log_entry)
                                except Exception as e_log:
                                    tqdm.write(f"    [Warning] Failed to write to visit log '{visit_log_filename}': {e_log}")
                            break 

                if len(visited_atz_types_this_trial) == len(ENTITY_DESCRIPTION_ORDER):
                    tqdm.write(f"  Trial {trial_num+1}: All ATZs visited.")
                    break # End this trial

                # --- Build intervention config for unvisited ATZs ---
                current_intervention_config = []
                for entity_type_in_order in ENTITY_DESCRIPTION_ORDER:
                    if entity_type_in_order not in visited_atz_types_this_trial:
                        current_intervention_config.append({
                            "channel": args.target_channel,
                            "position": mapped_atz_positions[entity_type_in_order],
                            "value": args.entity_values[entity_type_in_order],
                            "radius": args.intervention_radius,
                        })
                
                if current_intervention_config:
                    experiment.set_intervention(current_intervention_config)
                else: # All visited or no config generated (should not happen if loop not broken)
                    experiment.disable_intervention()

                # --- Agent takes a step ---
                obs_np = observation[0]['rgb'] # Assuming Procgen observation structure
                obs_tensor = torch.tensor(helpers.observation_to_rgb(obs_np), dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = experiment.model(obs_tensor) # Hook is called here

                if hasattr(outputs, 'logits'): logits = outputs.logits
                elif isinstance(outputs, tuple) and hasattr(outputs[0], 'logits'): logits = outputs[0].logits
                else: raise ValueError("Could not extract logits from model output.")
                
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                action = torch.multinomial(probabilities, num_samples=1).squeeze(-1).cpu().numpy()
                observation, reward, done, info = venv.step(action)

                if done:
                    tqdm.write(f"  Trial {trial_num+1}: Episode ended by 'done' flag at step {step_num+1}.")
                    break
            
            # End of step loop for trial
            trial_results.append({
                "trial_num": trial_num + 1,
                "visit_sequence": list(visit_sequence_this_trial), # Ensure it's a copy
                "num_visited": len(visited_atz_types_this_trial),
                "completed_sequence": visit_sequence_this_trial == ENTITY_DESCRIPTION_ORDER,
                "steps_taken": step_num +1
            })
            
            if save_this_trial_gif and frames_this_trial:
                try:
                    gif_dir = os.path.join(args.output_dir, "gifs", f"ch{args.target_channel}")
                    os.makedirs(gif_dir, exist_ok=True)
                    status = "success" if visit_sequence_this_trial == ENTITY_DESCRIPTION_ORDER else "fail"
                    gif_filename = os.path.join(gif_dir, f"trial_{trial_num+1}_{status}_seq_{'_'.join(visit_sequence_this_trial)}.gif")
                    imageio.mimsave(gif_filename, frames_this_trial, fps=10)
                    tqdm.write(f"    Saved GIF for trial {trial_num+1} to {gif_filename}")
                except Exception as e_gif:
                    tqdm.write(f"    Error saving GIF for trial {trial_num+1}: {e_gif}")
        
        finally:
            if handle is not None:
                handle.remove()
            venv.close()
            experiment.disable_intervention() # Ensure intervention is off for next trial

    # --- Analyze and Save Results ---
    results_df = pd.DataFrame(trial_results)
    summary_filename = os.path.join(args.output_dir, f"summary_results_{safe_layer_name}_ch{args.target_channel}_{timestamp_str}.csv")
    results_df.to_csv(summary_filename, index=False)
    print(f"\nTrial results saved to: {summary_filename}")

    num_successful_sequences = results_df["completed_sequence"].sum()
    percent_successful = (num_successful_sequences / args.num_trials) * 100 if args.num_trials > 0 else 0
    
    print(f"\n--- Experiment Summary ---")
    print(f"Layer: {args.layer_number} ({ordered_layer_names[args.layer_number]})")
    print(f"Target SAE Channel: {args.target_channel}")
    print(f"Activation Values: BlueK={args.val_blue_key}, GreenK={args.val_green_key}, RedK={args.val_red_key}, Gem={args.val_gem}")
    print(f"Total Trials: {args.num_trials}")
    print(f"Successful Sequences (visited all ATZs in order {ENTITY_DESCRIPTION_ORDER}): {num_successful_sequences}")
    print(f"Success Rate: {percent_successful:.2f}%")

    avg_steps_for_successful = results_df[results_df["completed_sequence"]]["steps_taken"].mean()
    avg_visited_for_successful = results_df[results_df["completed_sequence"]]["num_visited"].mean()
    avg_visited_for_failed = results_df[~results_df["completed_sequence"]]["num_visited"].mean()

    print(f"Avg steps for successful sequences: {avg_steps_for_successful:.2f}" if not np.isnan(avg_steps_for_successful) else "N/A (no successful trials)")
    print(f"Avg ATZs visited in successful sequences: {avg_visited_for_successful:.2f}" if not np.isnan(avg_visited_for_successful) else "N/A")
    print(f"Avg ATZs visited in failed sequences: {avg_visited_for_failed:.2f}" if not np.isnan(avg_visited_for_failed) else "N/A (all trials successful or no trials)")
    
    print("\nMost common sequences:")
    sequence_counts = results_df['visit_sequence'].astype(str).value_counts().nlargest(5)
    for seq_str, count in sequence_counts.items():
        print(f"  {seq_str}: {count} times")

    print(f"\nDetailed trial data saved to {summary_filename}")
    if visit_log_filename:
        print(f"Detailed visit event log saved to {visit_log_filename}")
    print("Sequential targeting experiment finished.")

if __name__ == "__main__":
    main() 