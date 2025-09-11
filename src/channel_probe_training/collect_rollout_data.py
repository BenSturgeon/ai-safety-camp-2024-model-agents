#!/usr/bin/env python3
"""
Collect training data from rollouts in empty maze environments.
Track what entities have been collected and label based on what the next target should be.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import pickle
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import argparse

from utils import heist
from utils.helpers import load_interpretable_model, ModelActivations, observation_to_rgb, get_device, generate_action
from utils.create_intervention_mazes import create_empty_corners_maze


def get_entity_positions(state):
    """Get current positions of all entities in the maze."""
    entities = {}
    state_vals = state.state_vals
    
    for ent in state_vals["ents"]:
        entity_type = ent["type"].val
        x = ent["x"].val
        y = ent["y"].val
        
        # Skip invalid positions
        if x < 0 or y < 0:
            continue
            
        # Player (type 0)
        if entity_type == 0:
            entities['player'] = (x, y)
        # Keys (type 2)
        elif entity_type == 2:
            color = ent["image_theme"].val
            color_names = {0: 'blue_key', 1: 'green_key', 2: 'red_key'}
            if color in color_names:
                entities[color_names[color]] = (x, y)
        # Gems (type 9)
        elif entity_type == 9:
            entities['gem'] = (x, y)
    
    return entities


def determine_next_target(collected_keys, remaining_entities):
    """
    Determine what the next target should be based on game logic.
    
    Priority order:
    1. Blue key (if not collected)
    2. Green key (if not collected)
    3. Red key (if not collected)
    4. Gem (after all keys)
    
    Returns:
        str: Label for next target ('blue_key', 'green_key', 'red_key', 'gem', or 'none')
    """
    # Check keys in priority order
    key_order = ['blue_key', 'green_key', 'red_key']
    
    for key in key_order:
        if key not in collected_keys and key in remaining_entities:
            return key
    
    # If all keys collected or no keys remain, target gem
    if 'gem' in remaining_entities:
        return 'gem'
    
    return 'none'


def run_rollout_and_collect_data(model, num_steps=100, collect_every=1, seed=None):
    """
    Run a rollout in an empty maze and collect data based on what the next target should be.
    
    Args:
        model: The trained model to use for actions
        num_steps: Maximum number of steps in the rollout
        collect_every: Collect data every N steps
        seed: Random seed for maze generation
    
    Returns:
        observations: List of observations
        activations: Dict of layer activations for each observation
        labels: List of labels (what the next target should be)
        metadata: Additional information about each sample
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create empty maze environment
    _, venv = create_empty_corners_maze(randomize_entities=True)
    
    # Get initial state
    obs = venv.reset()
    state = heist.state_from_venv(venv, 0)
    
    observations = []
    all_activations = {layer: [] for layer in ['conv1a', 'conv2a', 'conv2b', 'conv3a', 'conv4a', 'fc1', 'fc2', 'fc3']}
    labels = []
    metadata = []
    
    # Track collected keys
    collected_keys = set()
    
    # Hook storage for activations
    hook_dict = {}
    
    # Label mapping
    label_map = {
        'none': 0,
        'blue_key': 1,
        'green_key': 2,
        'red_key': 3,
        'gem': 4
    }
    
    for step in range(num_steps):
        # Get current entity positions
        current_entities = get_entity_positions(state)
        
        # Check what entities remain (not collected)
        remaining_entities = {}
        for entity_name, pos in current_entities.items():
            if entity_name != 'player' and entity_name not in collected_keys:
                remaining_entities[entity_name] = pos
        
        # Determine what the next target should be
        next_target = determine_next_target(collected_keys, remaining_entities)
        
        # Collect data at specified intervals
        if step % collect_every == 0:
            observations.append(obs[0])
            labels.append(label_map[next_target])
            
            # Store activations for this sample
            for layer_name in all_activations.keys():
                for hook_name, hook_data in hook_dict.items():
                    if layer_name in hook_name and 'output' in hook_data:
                        all_activations[layer_name].append(hook_data['output'].numpy())
                        break  # Found the layer, move to next
            
            metadata.append({
                'step': step,
                'collected_keys': list(collected_keys),
                'remaining_entities': list(remaining_entities.keys()),
                'next_target': next_target,
                'player_pos': current_entities.get('player', (-1, -1))
            })
        
        # Get action from model and extract activations
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float()
            
            # Register hooks to capture activations (only on first step)
            if step == 0:
                for name, module in model.named_modules():
                    for layer in all_activations.keys():
                        if layer == name or name.endswith('.' + layer):
                            hook_dict[name] = {}
                            def make_hook(layer_name):
                                def hook(module, input, output):
                                    hook_dict[layer_name]['output'] = output.detach().cpu()
                                return hook
                            module.register_forward_hook(make_hook(name))
            
            # Forward pass
            dist, value = model(obs_tensor)
            action = dist.sample().cpu().numpy()
        
        # Take action
        obs, reward, done, info = venv.step(action)
        
        # Update state
        state = heist.state_from_venv(venv, 0)
        
        # Check if any keys were collected this step
        new_entities = get_entity_positions(state)
        for key_name in ['blue_key', 'green_key', 'red_key']:
            if key_name in current_entities and key_name not in new_entities:
                collected_keys.add(key_name)
        
        # Check if gem was collected (would end episode)
        if 'gem' in current_entities and 'gem' not in new_entities:
            # Final sample after gem collection
            observations.append(obs[0])
            labels.append(label_map['none'])
            metadata.append({
                'step': step + 1,
                'collected_keys': list(collected_keys),
                'remaining_entities': [],
                'next_target': 'none',
                'player_pos': new_entities.get('player', (-1, -1))
            })
            break
        
        if done[0]:
            break
    
    venv.close()
    
    return observations, all_activations, labels, metadata, label_map


def collect_dataset(model, num_rollouts=100, num_steps_per_rollout=100, 
                   collect_every=2, logger=None):
    """
    Collect a dataset from multiple rollouts.
    
    Args:
        model: Trained model for generating actions
        num_rollouts: Number of rollouts to perform
        num_steps_per_rollout: Maximum steps per rollout
        collect_every: Collect data every N steps
        logger: Logger instance
    
    Returns:
        all_observations: Combined observations from all rollouts
        all_labels: Combined labels
        label_map: Mapping of labels to entity names
        all_metadata: Combined metadata
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    all_observations = []
    all_activations = {layer: [] for layer in ['conv1a', 'conv2a', 'conv2b', 'conv3a', 'conv4a', 'fc1', 'fc2', 'fc3']}
    all_labels = []
    all_metadata = []
    label_map = None
    
    logger.info(f"Collecting data from {num_rollouts} rollouts")
    logger.info(f"Max steps per rollout: {num_steps_per_rollout}")
    logger.info(f"Collecting every {collect_every} steps")
    
    for rollout_idx in tqdm(range(num_rollouts), desc="Rollouts"):
        seed = random.randint(0, 100000)
        
        observations, activations, labels, metadata, label_map = run_rollout_and_collect_data(
            model, 
            num_steps=num_steps_per_rollout,
            collect_every=collect_every,
            seed=seed
        )
        
        # Add rollout index to metadata
        for meta in metadata:
            meta['rollout_idx'] = rollout_idx
            meta['seed'] = seed
        
        all_observations.extend(observations)
        all_labels.extend(labels)
        all_metadata.extend(metadata)
        
        # Extend activations for each layer
        for layer in all_activations.keys():
            if layer in activations:
                all_activations[layer].extend(activations[layer])
    
    # Convert to numpy arrays
    all_observations = np.array(all_observations)
    all_labels = np.array(all_labels)
    
    # Print label distribution
    logger.info("\nLabel distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    reverse_label_map = {v: k for k, v in label_map.items()}
    for label_val, count in zip(unique, counts):
        logger.info(f"  {reverse_label_map[label_val]}: {count} ({100*count/len(all_labels):.1f}%)")
    
    # Print collection statistics
    total_blue_collected = sum(1 for m in all_metadata if 'blue_key' in m['collected_keys'])
    total_green_collected = sum(1 for m in all_metadata if 'green_key' in m['collected_keys'])
    total_red_collected = sum(1 for m in all_metadata if 'red_key' in m['collected_keys'])
    
    logger.info("\nCollection statistics:")
    logger.info(f"  Samples with blue key collected: {total_blue_collected}")
    logger.info(f"  Samples with green key collected: {total_green_collected}")
    logger.info(f"  Samples with red key collected: {total_red_collected}")
    
    # Convert activations to numpy arrays
    for layer in all_activations.keys():
        if all_activations[layer]:
            all_activations[layer] = np.array(all_activations[layer])
    
    return all_observations, all_activations, all_labels, label_map, all_metadata


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Collect rollout data from empty mazes')
    parser.add_argument('--num_rollouts', type=int, default=50,
                       help='Number of rollouts to perform')
    parser.add_argument('--steps_per_rollout', type=int, default=100,
                       help='Maximum steps per rollout')
    parser.add_argument('--collect_every', type=int, default=2,
                       help='Collect data every N steps')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Collecting Rollout Data from Empty Mazes")
    logger.info("="*60)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load model
    logger.info("Loading model...")
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Using model from: {args.model_path}")
        model = load_interpretable_model(model_path=args.model_path)
    else:
        # Try to use checkpoint 35001 if available
        model_path = "../../base_models/full_run/model_35001.0.pt"
        if os.path.exists(model_path):
            logger.info(f"Using checkpoint 35001: {model_path}")
            model = load_interpretable_model(model_path=model_path)
        else:
            logger.info("Using default model")
            model = load_interpretable_model()
    
    model.eval()
    
    # Collect dataset
    observations, activations, labels, label_map, metadata = collect_dataset(
        model,
        num_rollouts=args.num_rollouts,
        num_steps_per_rollout=args.steps_per_rollout,
        collect_every=args.collect_every,
        logger=logger
    )
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = os.path.join(
        os.path.dirname(__file__), 
        f'rollout_dataset_{timestamp}.pkl'
    )
    
    with open(dataset_path, 'wb') as f:
        pickle.dump({
            'observations': observations,
            'activations': activations,  # Now includes all layer activations
            'labels': labels,
            'label_map': label_map,
            'metadata': metadata,
            'num_rollouts': args.num_rollouts,
            'steps_per_rollout': args.steps_per_rollout,
            'collect_every': args.collect_every
        }, f)
    
    logger.info(f"\nDataset saved to: {dataset_path}")
    logger.info(f"Total samples: {len(observations)}")
    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()