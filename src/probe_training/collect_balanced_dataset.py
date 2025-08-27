#!/usr/bin/env python3
"""Collect balanced dataset for training next target prediction probes."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import pickle
import random
import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime

from utils import heist
from utils.entity_collection_detector import detect_collections, get_entity_counts
from utils.helpers import load_interpretable_model, generate_action, get_device


def setup_logging(log_dir=None):
    """Setup logging to both file and console."""
    if log_dir is None:
        log_dir = os.path.dirname(__file__)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'dataset_collection_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('dataset_collection')
    logger.info(f"Logging to {log_file}")
    return logger


def determine_next_target(state: heist.EnvState, entity_counts: Dict[int, int]) -> str:
    """Determine the next target based on current game state."""
    blue_key_collected = entity_counts.get(4, 0) == 1  
    green_key_collected = entity_counts.get(5, 0) == 1
    red_key_collected = entity_counts.get(6, 0) == 1
    gem_available = entity_counts.get(3, 0) == 1
    
    blue_lock_exists = state.entity_exists(heist.ENTITY_TYPES["lock"], heist.ENTITY_COLORS["blue"])
    green_lock_exists = state.entity_exists(heist.ENTITY_TYPES["lock"], heist.ENTITY_COLORS["green"])
    red_lock_exists = state.entity_exists(heist.ENTITY_TYPES["lock"], heist.ENTITY_COLORS["red"])
    
    if not blue_key_collected and entity_counts.get(4, 0) == 2:
        return 'blue_key'
    elif blue_key_collected and blue_lock_exists:
        return 'blue_lock'
    elif blue_key_collected and not blue_lock_exists:
        if not green_key_collected and entity_counts.get(5, 0) == 2:
            return 'green_key'
        elif green_key_collected and green_lock_exists:
            return 'green_lock'
        elif green_key_collected and not green_lock_exists:
            if not red_key_collected and entity_counts.get(6, 0) == 2:
                return 'red_key'
            elif red_key_collected and red_lock_exists:
                return 'red_lock'
            elif red_key_collected and not red_lock_exists and gem_available:
                return 'gem'
    
    if entity_counts.get(4, 0) == 2:
        return 'blue_key'
    elif entity_counts.get(5, 0) == 2:
        return 'green_key' 
    elif entity_counts.get(6, 0) == 2:
        return 'red_key'
    elif gem_available:
        return 'gem'
    
    return 'unknown'


def collect_balanced_dataset(
    model,
    samples_per_category=2000,
    max_episodes=1000,
    max_steps_per_episode=300,
    logger=None
):
    """Collect balanced dataset with equal distribution of all entity types."""
    if logger is None:
        logger = logging.getLogger('dataset_collection')
    
    target_categories = [
        'blue_key', 'blue_lock',
        'green_key', 'green_lock', 
        'red_key', 'red_lock',
        'gem'
    ]
    
    logger.info(f"Collecting {samples_per_category} samples per category")
    
    seeds_file = os.path.join(os.path.dirname(__file__), '../../viable_seeds.txt')
    viable_seeds = []
    if os.path.exists(seeds_file):
        with open(seeds_file, 'r') as f:
            viable_seeds = [int(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(viable_seeds)} viable seeds with red keys")
    
    if not viable_seeds:
        logger.warning("No viable seeds found! Using random seeds...")
        viable_seeds = list(range(10000, 20000))
    
    label_map = {name: i for i, name in enumerate(target_categories)}
    reverse_map = {i: name for name, i in label_map.items()}
    
    episode_data = []
    category_totals = {cat: 0 for cat in target_categories}
    
    device = get_device()
    model = model.to(device)
    
    episodes_run = 0
    successful_episodes = 0
    
    while episodes_run < max_episodes:
        if all(count >= samples_per_category * 2 for count in category_totals.values()):
            logger.info("Collected enough total samples for all categories!")
            break
        
        red_key_deficit = samples_per_category - category_totals.get('red_key', 0)
        
        if red_key_deficit > 0 and episodes_run % 2 == 0 and viable_seeds:
            seed = random.choice(viable_seeds)
        elif episodes_run % 5 == 0 and viable_seeds:
            seed = random.choice(viable_seeds)
        elif episodes_run % 5 == 1:
            seed = random.randint(0, 5000)
        elif episodes_run % 5 == 2:
            seed = random.randint(50000, 100000)
        else:
            seed = random.randint(0, 100000)
        
        venv = heist.create_venv(num=1, start_level=seed, num_levels=1)
        
        try:
            obs = venv.reset()
            state = heist.state_from_venv(venv, 0)
            
            episode_observations = []
            episode_targets = []
            
            previous_counts = None
            collection_order = []
            
            for step in range(max_steps_per_episode):
                state = heist.state_from_venv(venv, 0)
                current_counts = get_entity_counts(state)
                
                if previous_counts is not None:
                    _, collected = detect_collections(state, previous_counts)
                    if collected:
                        for entity in collected:
                            collection_order.append((step, entity))
                
                next_target = determine_next_target(state, current_counts)
                
                if next_target in target_categories:
                    if obs.ndim == 4:
                        obs_sample = obs[0]
                    elif obs.ndim == 3 and obs.shape[-1] == 3:
                        obs_sample = obs.transpose(2, 0, 1)
                    else:
                        obs_sample = obs
                    
                    episode_observations.append(obs_sample)
                    episode_targets.append(next_target)
                    category_totals[next_target] += 1
                
                action = generate_action(model, obs, is_procgen_env=True)
                obs, reward, done, info = venv.step(action)
                
                previous_counts = current_counts
                
                if done[0]:
                    if reward[0] > 0:
                        successful_episodes += 1
                    break
            
            if episode_observations:
                episode_data.append({
                    'observations': episode_observations,
                    'targets': episode_targets,
                    'seed': seed,
                    'successful': reward[0] > 0 if done[0] else False,
                    'collection_order': collection_order
                })
            
            episodes_run += 1
            
            if episodes_run % 50 == 0:
                logger.info(f"Progress: {episodes_run} episodes, {successful_episodes} successful")
                
        finally:
            venv.close()
    
    logger.info("\nCreating balanced dataset from collected episodes...")
    
    category_observations = {cat: [] for cat in target_categories}
    
    for episode in episode_data:
        for obs, target in zip(episode['observations'], episode['targets']):
            if target in target_categories:
                category_observations[target].append(obs)
    
    min_available = min(len(category_observations[cat]) for cat in target_categories)
    actual_samples_per_category = min(min_available, samples_per_category)
    
    logger.info(f"Using {actual_samples_per_category} samples per category")
    
    observations = []
    labels = []
    
    for cat in target_categories:
        cat_obs = category_observations[cat]
        if len(cat_obs) > actual_samples_per_category:
            indices = np.random.choice(len(cat_obs), actual_samples_per_category, replace=False)
            sampled_obs = [cat_obs[i] for i in indices]
        else:
            sampled_obs = cat_obs
        
        observations.extend(sampled_obs)
        labels.extend([label_map[cat]] * len(sampled_obs))
    
    observations = np.array(observations, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    indices = np.random.permutation(len(observations))
    observations = observations[indices]
    labels = labels[indices]
    
    logger.info(f"Final dataset: {len(observations)} total samples")
    
    stats = {
        'total_samples': len(observations),
        'episodes_run': episodes_run,
        'samples_per_category': actual_samples_per_category,
        'category_counts': {reverse_map[i]: int(c) for i, c in enumerate(np.bincount(labels))},
        'timestamp': datetime.now().isoformat()
    }
    
    return observations, labels, reverse_map, stats


def main():
    logger = setup_logging()
    logger.info("Starting balanced dataset collection")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger.info("Loading model...")
    model = load_interpretable_model()
    model.eval()
    
    observations, labels, label_map, stats = collect_balanced_dataset(
        model,
        samples_per_category=2000,
        max_episodes=1000,
        max_steps_per_episode=300,
        logger=logger
    )
    
    dataset_path = os.path.join(os.path.dirname(__file__), 'balanced_dataset.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump({
            'observations': observations,
            'labels': labels,
            'label_map': label_map
        }, f)
    logger.info(f"Saved dataset to {dataset_path}")
    
    stats_file = os.path.join(os.path.dirname(__file__), 'collection_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")


if __name__ == "__main__":
    main()