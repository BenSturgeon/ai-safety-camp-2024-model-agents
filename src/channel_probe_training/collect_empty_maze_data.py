#!/usr/bin/env python3
"""
Collect training data from empty maze environments with only keys and gems.
This data will be used to train channel-wise probes.
"""

import sys
import argparse
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

from utils import heist
from utils.helpers import load_interpretable_model, ModelActivations, observation_to_rgb, get_device
from utils.hud_utils import verify_hud_working
from utils.create_intervention_mazes import create_empty_corners_maze


def create_empty_maze_with_entities(seed=None, entity_types='keys_and_gems'):
    """
    Create an empty maze with only specified entity types using the existing function.
    
    Args:
        seed: Random seed for reproducibility
        entity_types: 'keys_only', 'gems_only', or 'keys_and_gems'
    
    Returns:
        state: The environment state
        venv: The environment
        entities_info: Information about entities in the maze
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Use the existing create_empty_corners_maze function
    # It creates a maze with gem and 3 keys in corners, player in center
    observations, venv = create_empty_corners_maze(randomize_entities=True)
    
    # Get the state from the venv
    state = heist.state_from_venv(venv, 0)
    
    # Get entity information from the state
    entities_info = []
    state_vals = state.state_vals
    
    for ent in state_vals["ents"]:
        entity_type = ent["type"].val
        x = ent["x"].val
        y = ent["y"].val
        
        # Skip player (type 0) and HUD keys (type 11)
        if entity_type == 0 or entity_type == 11:
            continue
            
        # Keys (type 2)
        if entity_type == 2:
            color = ent["image_theme"].val
            entities_info.append({'type': 'key', 'color': color, 'x': x, 'y': y})
        # Gems (type 9)
        elif entity_type == 9:
            entities_info.append({'type': 'gem', 'x': x, 'y': y})
    
    # Filter based on entity_types parameter
    if entity_types == 'keys_only':
        # Remove gems from the maze if we only want keys
        state_vals = state.state_vals
        for ent in state_vals["ents"]:
            if ent["type"].val == 9:  # Gem
                ent["x"].val = -1
                ent["y"].val = -1
        state.state_bytes = heist._serialize_maze_state(state_vals)
        entities_info = [e for e in entities_info if e['type'] == 'key']
        
    elif entity_types == 'gems_only':
        # Remove keys from the maze if we only want gems
        state_vals = state.state_vals
        for ent in state_vals["ents"]:
            if ent["type"].val == 2:  # Key
                ent["x"].val = -1
                ent["y"].val = -1
        state.state_bytes = heist._serialize_maze_state(state_vals)
        entities_info = [e for e in entities_info if e['type'] == 'gem']
    
    # entity_types == 'keys_and_gems' keeps everything as is
    
    return state, venv, entities_info


def get_next_target_label(state, entities_info):
    """
    Determine what the next target should be in the current state.
    
    For empty mazes with keys and gems:
    - Prioritize keys over gems
    - Choose closest entity of the prioritized type
    
    Returns:
        label: Integer label for the next target
        label_info: Dictionary with label details
    """
    # mouse_pos returns (y, x) in grid coordinates
    player_y, player_x = state.mouse_pos
    
    # Separate keys and gems
    keys = [e for e in entities_info if e['type'] == 'key']
    gems = [e for e in entities_info if e['type'] == 'gem']
    
    # Label mapping for empty maze entities
    # 0: none, 1: blue_key, 2: green_key, 3: red_key, 4: gem
    # (matching heist.KEY_COLORS: {0: 'blue', 1: 'green', 2: 'red'})
    
    if not keys and not gems:
        return 0, {'type': 'none'}
    
    # Prioritize keys
    if keys:
        # Find closest key
        min_dist = float('inf')
        closest_key = None
        
        for key in keys:
            dist = abs(key['x'] - player_x) + abs(key['y'] - player_y)  # Manhattan distance
            if dist < min_dist:
                min_dist = dist
                closest_key = key
        
        if closest_key:
            # Map color index to label: 0->1 (blue), 1->2 (green), 2->3 (red)
            label = closest_key['color'] + 1
            return label, {'type': 'key', 'color': closest_key['color'], 'distance': min_dist}
    
    # If no keys, target closest gem
    if gems:
        min_dist = float('inf')
        closest_gem = None
        
        for gem in gems:
            dist = abs(gem['x'] - player_x) + abs(gem['y'] - player_y)
            if dist < min_dist:
                min_dist = dist
                closest_gem = gem
        
        if closest_gem:
            return 4, {'type': 'gem', 'distance': min_dist}
    
    return 0, {'type': 'none'}


def collect_dataset(num_samples=10000, entity_types='keys_and_gems', logger=None):
    """
    Collect a dataset from empty maze environments.
    
    Args:
        num_samples: Number of samples to collect
        entity_types: Type of entities to include
        logger: Logger instance
    
    Returns:
        observations: Array of observations
        labels: Array of target labels
        label_map: Dictionary mapping labels to entity types
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    observations = []
    labels = []
    metadata = []
    
    # Label map for empty mazes
    label_map = {
        0: 'none',
        1: 'blue_key',   # color 0
        2: 'green_key',  # color 1
        3: 'red_key',    # color 2
        4: 'gem'
    }
    
    logger.info(f"Collecting {num_samples} samples with {entity_types}")
    
    for i in tqdm(range(num_samples), desc="Collecting data"):
        # Create a new maze with random seed
        seed = random.randint(0, 100000)
        state, venv, entities_info = create_empty_maze_with_entities(
            seed=seed, 
            entity_types=entity_types
        )
        
        # Get observation
        obs = venv.reset()
        
        # Get label for next target
        label, label_info = get_next_target_label(state, entities_info)
        
        observations.append(obs[0])  # First environment
        labels.append(label)
        metadata.append({
            'seed': seed,
            'entities': entities_info,
            'label_info': label_info,
            'player_pos': state.mouse_pos  # Returns (y, x)
        })
        
        venv.close()
    
    observations = np.array(observations)
    labels = np.array(labels)
    
    # Print label distribution
    logger.info("\nLabel distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label_val, count in zip(unique, counts):
        logger.info(f"  {label_map[label_val]}: {count} ({100*count/len(labels):.1f}%)")
    
    return observations, labels, label_map, metadata


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Collect empty maze dataset')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples to collect')
    parser.add_argument('--entity_types', type=str, default='keys_and_gems',
                       choices=['keys_only', 'gems_only', 'keys_and_gems'],
                       help='Types of entities to include')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Collecting Empty Maze Dataset")
    logger.info("="*60)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Collect dataset
    observations, labels, label_map, metadata = collect_dataset(
        num_samples=args.num_samples,
        entity_types=args.entity_types,
        logger=logger
    )
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = os.path.join(
        os.path.dirname(__file__), 
        f'empty_maze_dataset_{timestamp}.pkl'
    )
    
    with open(dataset_path, 'wb') as f:
        pickle.dump({
            'observations': observations,
            'labels': labels,
            'label_map': label_map,
            'metadata': metadata,
            'entity_types': args.entity_types,
            'num_samples': args.num_samples
        }, f)
    
    logger.info(f"\nDataset saved to: {dataset_path}")
    logger.info(f"Total samples: {len(observations)}")
    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()