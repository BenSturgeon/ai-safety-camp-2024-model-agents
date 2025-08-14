"""
Channel ablation sweep with bias correction.
This is a modified version of channel_ablation_sweep.py that uses bias-corrected mazes.
"""

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

# Import bias-corrected maze functions
from bias_corrected_mazes import create_bias_corrected_fork_maze, create_bias_corrected_corners_maze

from utils.helpers import run_episode_and_get_final_state
from utils.heist import (
    EnvState,
    ENTITY_TYPES,
    KEY_COLORS,
    create_venv,
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
# Need global references to sae and device for the hook
sae = None
hook_device = None


def ablation_hook(module, input, output):
    """Hook to zero out specific channels during forward pass."""
    if FEATURES_TO_ZERO:
        modified_output = output.clone()
        for channel in FEATURES_TO_ZERO:
            if channel < modified_output.shape[1]:  # Check bounds
                modified_output[:, channel, :, :] = 0.0
        return modified_output
    return output


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Channel ablation sweep with bias correction.")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the base interpretable model.")
    parser.add_argument("--sae_checkpoint_path", type=str, default=None,
                       help="Path to the SAE checkpoint (.pt file). If omitted, runs base model ablation.")
    parser.add_argument("--layer_spec", type=str, required=True,
                       help="SAE layer number (int) OR base model layer name (str).")
    
    # Experiment arguments
    parser.add_argument("--num_trials", type=int, default=5,
                       help="Number of trials per channel.")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Maximum steps per episode.")
    parser.add_argument("--output_dir", type=str, default="channel_ablation_results_bias_corrected",
                       help="Directory to save results.")
    parser.add_argument("--no_save_gifs", action="store_true",
                       help="Disable saving GIFs.")
    
    # Channel range arguments
    parser.add_argument("--start_channel", type=int, default=0,
                       help="Start channel for ablation (inclusive).")
    parser.add_argument("--end_channel", type=int, default=None,
                       help="End channel for ablation (exclusive).")
    parser.add_argument("--total_channels_for_base_layer", type=int, default=None,
                       help="Total channels for base layer (required if no SAE).")
    
    # Maze arguments
    parser.add_argument("--maze_type", type=str, default="fork", 
                       choices=["fork", "corners"],
                       help="Type of maze to use.")
    parser.add_argument("--bias_direction", type=str, default=None,
                       choices=["up", "down", "left", "right"],
                       help="Bias direction for maze correction.")
    
    return parser.parse_args()


def create_maze_environment(maze_type, bias_direction=None):
    """Create maze environment with optional bias correction."""
    if bias_direction:
        print(f"Creating {maze_type} maze with bias correction for '{bias_direction}' bias")
        if maze_type == "fork":
            return create_bias_corrected_fork_maze(bias_direction)
        elif maze_type == "corners":
            return create_bias_corrected_corners_maze(bias_direction)
        else:
            raise ValueError(f"Unknown maze type: {maze_type}")
    else:
        # Use original mazes if no bias correction
        from utils.create_intervention_mazes import create_fork_maze, create_corners_maze
        if maze_type == "fork":
            return create_fork_maze()
        elif maze_type == "corners":
            return create_corners_maze()
        else:
            raise ValueError(f"Unknown maze type: {maze_type}")


def run_channel_ablation_sweep():
    """Main function to run channel ablation sweep with bias correction."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if this is SAE or base model run
    is_sae_run = args.sae_checkpoint_path is not None
    
    # Load model
    print("Loading base model...", flush=True)
    model = helpers.load_interpretable_model(model_path=args.model_path).to(device)
    model.eval()
    
    # Load SAE if specified
    global sae, hook_device
    if is_sae_run:
        print("Loading SAE...", flush=True)
        sae = load_sae_from_checkpoint(args.sae_checkpoint_path).to(device)
        sae.eval()
        hook_device = device
        
        # Get layer details
        layer_number = int(args.layer_spec)
        layer_name = ordered_layer_names[layer_number]
        total_channels = sae.encoder.out_features
        
        print(f"SAE Mode: Layer {layer_number} ({layer_name}), {total_channels} channels")
    else:
        # Base model mode
        layer_name = args.layer_spec
        total_channels = args.total_channels_for_base_layer
        
        if total_channels is None:
            raise ValueError("Must specify --total_channels_for_base_layer for base model runs")
        
        print(f"Base Model Mode: Layer {layer_name}, {total_channels} channels")
    
    # Determine channel range
    start_channel = args.start_channel
    end_channel = args.end_channel if args.end_channel is not None else total_channels
    
    print(f"Running ablation for channels {start_channel} to {end_channel-1}")
    
    # Register hook
    if is_sae_run:
        # Hook on SAE encoder
        hook_handle = sae.encoder.register_forward_hook(ablation_hook)
    else:
        # Hook on base model layer
        target_layer = getattr(model, layer_name)
        hook_handle = target_layer.register_forward_hook(ablation_hook)
    
    # Results storage
    results = []
    
    try:
        # Run ablation for each channel
        for channel in tqdm(range(start_channel, end_channel), desc="Channels"):
            print(f"\n--- Channel {channel} ---")
            
            # Set channel to ablate
            global FEATURES_TO_ZERO
            FEATURES_TO_ZERO = [channel]
            
            # Run trials for this channel
            for trial in tqdm(range(args.num_trials), desc=f"Trials Ch {channel}", leave=False):
                
                # Create fresh maze environment for each trial
                try:
                    _, venv_trial = create_maze_environment(args.maze_type, args.bias_direction)
                    
                    # Run episode
                    total_reward, frames, last_state_bytes, ended_by_gem, ended_by_timeout = \
                        run_episode_and_get_final_state(
                            venv_trial, 
                            model, 
                            save_gif=not args.no_save_gifs and trial == 0,  # Save GIF for first trial
                            filepath=os.path.join(args.output_dir, f"ch{channel}_trial{trial}.gif"),
                            episode_timeout=args.max_steps
                        )
                    
                    # Store results
                    results.append({
                        'channel': channel,
                        'trial': trial,
                        'total_reward': total_reward,
                        'ended_by_gem': ended_by_gem,
                        'ended_by_timeout': ended_by_timeout,
                        'bias_direction': args.bias_direction,
                        'maze_type': args.maze_type
                    })
                    
                    # Close environment
                    venv_trial.close()
                    
                except Exception as e:
                    print(f"Error in channel {channel}, trial {trial}: {e}")
                    results.append({
                        'channel': channel,
                        'trial': trial,
                        'total_reward': 0.0,
                        'ended_by_gem': False,
                        'ended_by_timeout': True,
                        'bias_direction': args.bias_direction,
                        'maze_type': args.maze_type,
                        'error': str(e)
                    })
    
    finally:
        # Remove hook
        hook_handle.remove()
    
    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bias_suffix = f"_bias_{args.bias_direction}" if args.bias_direction else ""
    results_file = os.path.join(args.output_dir, f"ablation_results_{args.layer_spec}_{timestamp}{bias_suffix}.csv")
    results_df.to_csv(results_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    summary = results_df.groupby('channel').agg({
        'total_reward': 'mean',
        'ended_by_gem': 'mean',
        'ended_by_timeout': 'mean'
    }).round(3)
    
    print("\nSummary by channel:")
    print(summary)
    
    return results_file


if __name__ == "__main__":
    run_channel_ablation_sweep()