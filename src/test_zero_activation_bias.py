#!/usr/bin/env python3
"""
Test the natural bias direction of models by zeroing all channel activations
and observing which direction the agent tends to move in a cross-shaped maze.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import sys
from collections import defaultdict
import datetime
import imageio
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.create_intervention_mazes import create_cross_maze
from utils import heist
from utils.helpers import run_episode_and_get_final_state, load_interpretable_model, prepare_frame_for_gif, process_observation_for_model
import pandas as pd

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for hooks
ZERO_ALL_ACTIVATIONS = False
TARGET_LAYER = None
HOOK_HANDLE = None

def zero_activations_hook(module, input, output):
    """Hook function to zero all activations in a layer."""
    if ZERO_ALL_ACTIVATIONS:
        if isinstance(output, torch.Tensor):
            return torch.zeros_like(output)
    return output

def register_hooks_on_layer(model: nn.Module, layer_name: str):
    """Register hooks to zero activations on specified layer."""
    global HOOK_HANDLE, TARGET_LAYER
    
    # Remove existing hook if any
    if HOOK_HANDLE is not None:
        HOOK_HANDLE.remove()
        HOOK_HANDLE = None
    
    TARGET_LAYER = layer_name
    
    # Layer names map directly to model attributes
    valid_layers = ['conv1a', 'conv2a', 'conv2b', 'conv3a', 'conv4a']
    
    if layer_name not in valid_layers:
        raise ValueError(f"Unknown layer name: {layer_name}. Valid options: {valid_layers}")
    
    # Get the target module directly
    target_module = getattr(model, layer_name)
    
    # Register the hook
    HOOK_HANDLE = target_module.register_forward_hook(zero_activations_hook)
    print(f"Registered zero activation hook on layer: {layer_name}")

def get_entity_at_position(state, x, y):
    """Get entity type at given position."""
    for ent in state.state_vals["ents"]:
        if abs(ent["x"].val - x) < 0.5 and abs(ent["y"].val - y) < 0.5:
            ent_type = ent["image_type"].val
            if ent_type == 9:  # Gem
                return "gem"
            elif ent_type == 2:  # Key
                theme = ent["image_theme"].val
                color_map = {0: "blue_key", 1: "green_key", 2: "red_key"}
                return color_map.get(theme, "unknown_key")
    return None

def run_zero_activation_test(
    model_path: str,
    layer_name: str = "conv4a",
    num_trials: int = 100,
    max_steps: int = 30,
    save_gifs: bool = True,
    output_dir: str = "zero_activation_bias_results"
) -> Dict:
    """
    Run bias test with all activations zeroed.
    
    Returns:
        Dictionary containing bias statistics and results
    """
    global ZERO_ALL_ACTIVATIONS
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{layer_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_interpretable_model(model_path=model_path).to(device)
    model.eval()
    
    # Register hook to zero activations
    register_hooks_on_layer(model, layer_name)
    
    # Track results
    results = {
        "first_movements": defaultdict(int),
        "entities_reached": defaultdict(int),
        "steps_to_entity": [],
        "trials_data": []
    }
    
    # Entity positions in cross maze (from pattern)
    entity_positions = {
        "green_key": (3, 0),   # Left
        "red_key": (3, 6),     # Right  
        "blue_key": (0, 3),    # Top
        "gem": (6, 3)          # Bottom
    }
    
    print(f"Running {num_trials} trials with all {layer_name} activations zeroed...")
    
    for trial in range(num_trials):
        # Create cross maze environment
        observations, venv = create_cross_maze()
        obs = observations[0]
        
        # Enable activation zeroing
        ZERO_ALL_ACTIVATIONS = True
        
        # Track data for this trial
        trial_data = {
            "trial": trial,
            "first_movement": None,
            "entity_reached": None,
            "steps": 0,
            "final_reward": 0
        }
        
        # For GIF recording
        if save_gifs and trial < 5:  # Save first 5 trials
            # Use the helper function to prepare frame
            frame = prepare_frame_for_gif(obs)
            frames = [frame] if frame is not None else []
        
        # Run episode
        done = False
        steps = 0
        first_movement_recorded = False
        last_pos = (3, 3)  # Starting position in cross maze
        
        while not done and steps < max_steps:
            # Get action from model
            with torch.no_grad():
                obs_tensor = process_observation_for_model(obs, device)
                dist, value = model(obs_tensor)
                action = dist.sample().cpu().numpy()
            
            # Execute action
            obs, reward, done, info = venv.step(action)
            obs = obs[0]
            
            # Record first movement
            if not first_movement_recorded and action[0] in [1, 2, 3, 4]:
                action_to_dir = {1: "down", 2: "up", 3: "left", 4: "right"}
                first_movement = action_to_dir[action[0]]
                results["first_movements"][first_movement] += 1
                trial_data["first_movement"] = first_movement
                first_movement_recorded = True
            
            # Save frame for GIF
            if save_gifs and trial < 5:
                frame = prepare_frame_for_gif(obs)
                if frame is not None:
                    frames.append(frame)
            
            steps += 1
            trial_data["steps"] = steps
            
            # Check if entity was collected (reward > 0)
            if reward[0] > 0:
                # Determine which entity was reached based on last position
                state = heist.state_from_venv(venv, 0)
                
                # Check each entity position to see which one is missing
                for entity_name, (y, x) in entity_positions.items():
                    entity = get_entity_at_position(state, x, y)
                    if entity is None:  # This entity was collected
                        results["entities_reached"][entity_name] += 1
                        trial_data["entity_reached"] = entity_name
                        results["steps_to_entity"].append(steps)
                        break
                
                trial_data["final_reward"] = reward[0]
        
        # Save trial data
        results["trials_data"].append(trial_data)
        
        # Save GIF for first few trials
        if save_gifs and trial < 5:
            gif_path = os.path.join(run_dir, f"trial_{trial}_zeroed_{layer_name}.gif")
            imageio.mimsave(gif_path, frames, fps=5)
            print(f"Saved GIF: {gif_path}")
        
        venv.close()
        
        # Progress update
        if (trial + 1) % 20 == 0:
            print(f"Completed {trial + 1}/{num_trials} trials")
    
    # Disable activation zeroing
    ZERO_ALL_ACTIVATIONS = False
    
    # Remove hook
    if HOOK_HANDLE is not None:
        HOOK_HANDLE.remove()
    
    # Calculate statistics
    total_movements = sum(results["first_movements"].values())
    total_entities = sum(results["entities_reached"].values())
    
    bias_stats = {
        "layer_zeroed": layer_name,
        "num_trials": num_trials,
        "first_movement_distribution": {
            direction: count / total_movements if total_movements > 0 else 0
            for direction, count in results["first_movements"].items()
        },
        "entity_reached_distribution": {
            entity: count / total_entities if total_entities > 0 else 0
            for entity, count in results["entities_reached"].items()
        },
        "avg_steps_to_entity": np.mean(results["steps_to_entity"]) if results["steps_to_entity"] else 0,
        "entity_reach_rate": total_entities / num_trials
    }
    
    # Find dominant direction
    if results["first_movements"]:
        dominant_dir = max(results["first_movements"].items(), key=lambda x: x[1])
        bias_stats["dominant_direction"] = dominant_dir[0]
        bias_stats["dominant_direction_rate"] = dominant_dir[1] / total_movements if total_movements > 0 else 0
    
    # Save results
    save_results(results, bias_stats, run_dir)
    
    return bias_stats

def save_results(results: Dict, bias_stats: Dict, output_dir: str):
    """Save detailed results and statistics."""
    # Save raw trial data as CSV
    df = pd.DataFrame(results["trials_data"])
    df.to_csv(os.path.join(output_dir, "trial_data.csv"), index=False)
    
    # Save summary statistics
    with open(os.path.join(output_dir, "bias_statistics.txt"), 'w') as f:
        f.write("Zero Activation Bias Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Layer zeroed: {bias_stats['layer_zeroed']}\n")
        f.write(f"Number of trials: {bias_stats['num_trials']}\n")
        f.write(f"Entity reach rate: {bias_stats['entity_reach_rate']:.2%}\n")
        f.write(f"Average steps to entity: {bias_stats['avg_steps_to_entity']:.1f}\n\n")
        
        f.write("First Movement Distribution:\n")
        for direction, rate in bias_stats['first_movement_distribution'].items():
            f.write(f"  {direction}: {rate:.2%}\n")
        
        f.write(f"\nDominant direction: {bias_stats.get('dominant_direction', 'None')}\n")
        f.write(f"Dominant direction rate: {bias_stats.get('dominant_direction_rate', 0):.2%}\n\n")
        
        f.write("Entity Reached Distribution:\n")
        for entity, rate in bias_stats['entity_reached_distribution'].items():
            f.write(f"  {entity}: {rate:.2%}\n")
    
    print(f"\nResults saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Test model bias with zeroed activations")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--layer_name", type=str, default="conv4a",
                       choices=["conv1a", "conv2a", "conv2b", "conv3a", "conv4a"],
                       help="Layer to zero activations in")
    parser.add_argument("--num_trials", type=int, default=100,
                       help="Number of trials to run")
    parser.add_argument("--max_steps", type=int, default=30,
                       help="Maximum steps per episode")
    parser.add_argument("--save_gifs", action="store_true",
                       help="Save GIFs of first few trials")
    parser.add_argument("--output_dir", type=str, default="zero_activation_bias_results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    print(f"Testing bias with zeroed {args.layer_name} activations")
    print(f"Model: {args.model_path}")
    
    bias_stats = run_zero_activation_test(
        model_path=args.model_path,
        layer_name=args.layer_name,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        save_gifs=args.save_gifs,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("BIAS TEST SUMMARY")
    print("=" * 60)
    print(f"Dominant direction: {bias_stats.get('dominant_direction', 'None')}")
    print(f"Dominant direction rate: {bias_stats.get('dominant_direction_rate', 0):.2%}")
    print(f"Entity reach rate: {bias_stats['entity_reach_rate']:.2%}")
    
    print("\nFirst movement distribution:")
    for direction, rate in bias_stats['first_movement_distribution'].items():
        print(f"  {direction}: {rate:.2%}")

if __name__ == "__main__":
    main()