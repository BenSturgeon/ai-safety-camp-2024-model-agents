#!/usr/bin/env python3
"""
Test entity-specific ablations using computed activation spans.
Runs rollouts with ablations targeting specific entities and tracks collection rates.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import imageio

from utils.create_intervention_mazes import create_empty_corners_maze
from utils import heist
from utils.heist import EnvState
from utils import helpers
from utils.simple_entity_tracker import EntityTracker
from utils.entity_collection_detector import detect_collections, get_entity_counts, ENTITY_CODES
from compute_entity_activation_spans import compute_spans_for_directory
from check_disjoint_spans import find_disjoint_spans


class NextTargetProbe(nn.Module):
    """Probe network to predict next target from model activations."""
    
    def __init__(self, input_dim, num_classes=5, hidden_dim=256):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.probe(x)


def determine_next_target(entity_counts: Dict[int, int]) -> str:
    """
    Determine what the next target should be based on current entity states.
    
    Logic:
    - Keys (codes 4,5,6): count of 2 means on board + in HUD, 1 means only in HUD (collected)
    - Gem (code 3): count of 1 means on board, 0 means collected
    - Locks: disappear when opened (not tracked in counts but we check if keys are collected)
    """
    # Check collection status
    blue_key_collected = entity_counts.get(4, 0) == 1  # Only HUD copy remains
    green_key_collected = entity_counts.get(5, 0) == 1
    red_key_collected = entity_counts.get(6, 0) == 1
    gem_on_board = entity_counts.get(3, 0) == 1
    
    # Determine next target based on collection order
    if not blue_key_collected and entity_counts.get(4, 0) == 2:
        return 'blue_key'
    elif blue_key_collected:
        # After blue key, target blue lock (we assume it exists if we have all entities)
        # We can't directly check lock status from counts, so we move to next key
        if not green_key_collected and entity_counts.get(5, 0) == 2:
            return 'green_key'
        elif green_key_collected:
            if not red_key_collected and entity_counts.get(6, 0) == 2:
                return 'red_key'
            elif red_key_collected and gem_on_board:
                return 'gem'
    
    # Default fallback - find first available entity
    if entity_counts.get(4, 0) == 2:
        return 'blue_key'
    elif entity_counts.get(5, 0) == 2:
        return 'green_key'
    elif entity_counts.get(6, 0) == 2:
        return 'red_key'
    elif entity_counts.get(3, 0) == 1:
        return 'gem'
    
    return 'unknown'


def load_probe(probe_path: str, device: str = "cuda") -> Tuple[NextTargetProbe, Dict]:
    """Load the trained probe."""
    if not os.path.exists(probe_path):
        return None, None
    
    checkpoint = torch.load(probe_path, map_location=device)
    
    probe = NextTargetProbe(
        input_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes']
    ).to(device)
    
    probe.load_state_dict(checkpoint['probe_state_dict'])
    probe.eval()
    
    return probe, checkpoint['label_map']


def extract_conv3a_features(observation, model, device: str = "cuda"):
    """Extract conv3a features from a single observation."""
    if observation.ndim == 3:
        observation = observation.unsqueeze(0)
    
    model_activations = helpers.ModelActivations(model)
    
    # Convert to RGB format expected by model
    obs_rgb = helpers.observation_to_rgb(observation)
    
    # Get activations
    _, activations = model_activations.run_with_cache(obs_rgb, ['conv3a'])
    
    # Extract and flatten features
    if 'conv3a' in activations:
        feat = activations['conv3a']
        if isinstance(feat, tuple):
            feat = feat[0]
        
        # Flatten spatial dimensions
        if feat.ndim == 4:  # (batch, channels, height, width)
            feat = feat.reshape(feat.shape[0], -1)
        elif feat.ndim == 3:  # Single sample
            feat = feat.reshape(1, -1)
        
        model_activations.clear_hooks()
        return feat.detach()
    
    model_activations.clear_hooks()
    return None


def load_model(model_path: str, device: str = "cuda"):
    """Load the base model."""
    model = helpers.load_interpretable_model(model_path=model_path).to(device)
    model.eval()
    return model

def setup_ablation_hook(model, layer_name: str, activation_spans: Dict, target_entity: str, device: str = "cuda", 
                       inverse: bool = False, disjoint_only: bool = False, buffer_multiplier: float = 1.0):
    """
    Setup a hook that ablates channels based on activation spans for a target entity.
    
    Args:
        model: The model to hook
        layer_name: Name of the layer to ablate (e.g., "conv4a")
        activation_spans: Dict containing spans for each entity
        target_entity: Name of the entity to target (e.g., "gem", "blue_key")
        device: Device to run on
        inverse: If True, ablate everything EXCEPT the target entity spans
        disjoint_only: If True, use only non-overlapping spans (requires inverse=True)
        buffer_multiplier: Multiplier for activation range buffer (1.0 = no buffer, 1.25 = 25% buffer)
    """
    handle = None
    
    def ablation_hook(module, input, output):
        """Ablate based on the mode (normal, inverse, or disjoint)."""
        if target_entity not in activation_spans:
            return output
        
        # If disjoint_only mode, compute disjoint spans first
        if disjoint_only and inverse:
            # Use disjoint spans instead of full spans
            all_disjoint_spans = find_disjoint_spans(activation_spans)
            entity_channels = all_disjoint_spans.get(target_entity, {})
            
            if not entity_channels:
                # No disjoint spans for this entity - zero everything
                return torch.zeros_like(output)
        else:
            entity_channels = activation_spans[target_entity]
        
        if not inverse:
            # Normal mode: ablate only the target entity's spans
            # For each channel that responds to this entity
            for channel_str, ranges in entity_channels.items():
                channel = int(channel_str)
                
                # Only zero out spatial locations within the banned ranges
                if output.ndim == 4:  # (batch, channels, height, width)
                    channel_activations = output[:, channel, :, :]  # Shape: (batch, height, width)
                    
                    # Create mask for locations to ablate
                    ablation_mask = torch.zeros_like(channel_activations, dtype=torch.bool)
                    
                    for range_min, range_max in ranges:
                        # Apply buffer multiplier if specified
                        if buffer_multiplier != 1.0:
                            range_width = range_max - range_min
                            expanded_width = range_width * buffer_multiplier
                            center = (range_min + range_max) / 2
                            range_min = center - expanded_width / 2
                            range_max = center + expanded_width / 2
                        
                        # Mark spatial locations within the range for ablation
                        in_range = (channel_activations >= range_min) & (channel_activations <= range_max)
                        ablation_mask = ablation_mask | in_range
                    
                    # Zero out only the spatial locations that fall within the ranges
                    output[:, channel, :, :][ablation_mask] = 0
        else:
            # Inverse mode: ablate everything EXCEPT the target entity's spans
            # This means:
            # 1. ALL channels without spans for this entity are completely zeroed
            # 2. For channels WITH spans, ONLY preserve values within the span ranges
            # entity_channels already set above based on disjoint_only flag
            
            if output.ndim == 4:  # (batch, channels, height, width)
                num_channels = output.shape[1]
                
                # Start with everything zeroed
                output_clone = torch.zeros_like(output)
                
                # Then restore ONLY the spatial locations within spans for channels that have them
                for channel_str, ranges in entity_channels.items():
                    channel = int(channel_str)
                    if channel < num_channels:  # Safety check
                        channel_activations = output[:, channel, :, :]  # Original activations
                        
                        # For this channel, only preserve values that fall within the activation ranges
                        for range_min, range_max in ranges:
                            # Apply buffer multiplier if specified (but NOT for disjoint mode)
                            if buffer_multiplier != 1.0 and not disjoint_only:
                                range_width = range_max - range_min
                                expanded_width = range_width * buffer_multiplier
                                center = (range_min + range_max) / 2
                                range_min = center - expanded_width / 2
                                range_max = center + expanded_width / 2
                            
                            # Find spatial locations where activation values fall within this range
                            in_range_mask = (channel_activations >= range_min) & (channel_activations <= range_max)
                            
                            # Restore only these specific locations
                            output_clone[:, channel, :, :][in_range_mask] = channel_activations[in_range_mask]
                
                output = output_clone
        
        return output
    
    # Find the layer to hook
    for name, module in model.named_modules():
        if layer_name in name:
            handle = module.register_forward_hook(ablation_hook)
            break
    
    return handle

def run_rollout_with_ablation(model, venv, activation_spans: Dict, target_entity: str = None, 
                             layer_name: str = "conv4a", max_steps: int = 100, device: str = "cuda",
                             save_gif: bool = False, gif_path: str = None, inverse: bool = False,
                             disjoint_only: bool = False, buffer_multiplier: float = 1.0,
                             probe=None, probe_label_map: Dict = None):
    """
    Run a single rollout with optional entity-specific ablation and probe analysis.
    
    Args:
        model: The model to use
        venv: The environment
        activation_spans: Activation spans for each entity
        target_entity: Entity to ablate (None for no ablation)
        layer_name: Layer to apply ablation to
        max_steps: Maximum steps per rollout
        device: Device to run on
        save_gif: Whether to save a GIF of this rollout
        gif_path: Path to save the GIF (if save_gif is True)
        inverse: If True, ablate everything EXCEPT the target entity spans
        disjoint_only: If True, use only non-overlapping spans
        buffer_multiplier: Multiplier for activation range buffer
        probe: Optional trained probe for next target prediction
        probe_label_map: Label mapping for probe predictions
        
    Returns:
        Dict with rollout results including collected entities and probe analysis
    """
    # Get initial state
    observation = venv.reset()
    initial_state_bytes = venv.env.callmethod("get_state")[0]
    initial_state = EnvState(initial_state_bytes)
    
    # Initialize the simple tracker
    tracker = EntityTracker(initial_state)
    
    if save_gif:
        print(f"    Initial entities on board:")
    
    # Setup ablation hook if targeting an entity
    hook_handle = None
    if target_entity and activation_spans:
        hook_handle = setup_ablation_hook(model, layer_name, activation_spans, target_entity, device, inverse, disjoint_only, buffer_multiplier)
    
    # Track probe predictions and actual behavior
    step_data = []
    probe_predictions = []
    actual_next_targets = []
    
    # Run rollout
    done = False
    steps = 0
    obs_list = []
    
    while not done and steps < max_steps:
        # Get current state for ground truth next target (do this before action)
        current_state_bytes = venv.env.callmethod("get_state")[0]
        current_state = EnvState(current_state_bytes)
        entity_counts = get_entity_counts(current_state)
        actual_next_target = determine_next_target(entity_counts)
        
        # Process observation
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device)
        if obs_tensor.ndim == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        elif obs_tensor.ndim == 4 and obs_tensor.shape[0] != 1:
            obs_tensor = obs_tensor[0].unsqueeze(0)
        
        # Extract conv3a features and get probe prediction if probe is available
        probe_prediction = None
        probe_confidence = None
        
        if probe is not None and probe_label_map is not None:
            conv3a_features = extract_conv3a_features(obs_tensor, model, device)
            if conv3a_features is not None:
                with torch.no_grad():
                    logits = probe(conv3a_features)
                    probs = torch.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    confidence = torch.max(probs, dim=1)[0].item()
                    
                    # Map back to entity name
                    reverse_label_map = {v: k for k, v in probe_label_map.items()}
                    probe_prediction = reverse_label_map.get(predicted_class, 'unknown')
                    probe_confidence = confidence
        
        # Store step data for probe analysis
        if probe is not None:
            step_data.append({
                'step': steps,
                'actual_next_target': actual_next_target,
                'probe_prediction': probe_prediction,
                'probe_confidence': probe_confidence,
                'probe_correct': probe_prediction == actual_next_target,
                'entity_counts': dict(entity_counts)
            })
        
        # Get action from model
        with torch.no_grad():
            action = helpers.generate_action(model, obs_tensor, is_procgen_env=True)
        
        # Step environment
        observation, reward, done, info = venv.step(action)
        done = done[0]
        steps += 1
        
        # Update tracker - this handles all collection detection
        current_state_bytes = venv.env.callmethod("get_state")[0]
        current_state = EnvState(current_state_bytes)
        collected_this_step = tracker.update(current_state, steps, done)
        
        if collected_this_step and save_gif:
            print(f"    Step {steps}: Collected {', '.join(collected_this_step)}!")
        
        # Store observation
        if observation is not None and observation.shape[0] > 0:
            current_frame = observation[0]
            if current_frame.ndim == 3 and current_frame.shape[0] in [1, 3, 4]:
                current_frame = (current_frame * 255.0).astype(np.uint8)
                current_frame = np.transpose(current_frame, (1, 2, 0))
                obs_list.append(current_frame)
    
    # Clean up hook
    if hook_handle:
        hook_handle.remove()
    
    # Get final state
    final_state_bytes = venv.env.callmethod("get_state")[0]
    final_state = EnvState(final_state_bytes)
    final_player_pos = final_state.mouse_pos
    
    # Get collection data from tracker
    collection_order = tracker.get_collection_order()
    collected_entity_names = tracker.get_collected_entities()
    
    if save_gif:
        print(f"    Final summary: Collected {len(collected_entity_names)} entities in {steps} steps")
    
    # Save GIF if requested
    if save_gif and gif_path and obs_list:
        try:
            imageio.mimsave(gif_path, obs_list, fps=10)
            print(f"  Saved GIF to {gif_path}")
        except Exception as e:
            print(f"  Failed to save GIF: {e}")
    
    # Calculate probe accuracy for this rollout
    probe_accuracy = 0.0
    num_valid_predictions = 0
    
    if probe is not None and step_data:
        valid_predictions = [s for s in step_data if s['probe_prediction'] is not None and s['actual_next_target'] != 'unknown']
        if valid_predictions:
            probe_accuracy = sum(1 for s in valid_predictions if s['probe_correct']) / len(valid_predictions)
            num_valid_predictions = len(valid_predictions)
    
    result = {
        "final_player_pos": final_player_pos,
        "steps": steps,
        "collected_entity_names": collected_entity_names,
        "collection_order": collection_order,  # List of (step, entity_name) tuples
        "gem_collected": 'gem' in collected_entity_names,
        "num_keys_collected": len([name for name in collected_entity_names if 'key' in name]),
        "observations": obs_list
    }
    
    # Add probe analysis if probe was used
    if probe is not None:
        result.update({
            "step_data": step_data,
            "probe_accuracy": probe_accuracy,
            "num_valid_predictions": num_valid_predictions
        })
    
    return result

def get_model_path_from_config(run_dir: str) -> str:
    """Extract model path from the parallel_run_config.txt file."""
    config_path = os.path.join(run_dir, "parallel_run_config.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Find model_path in the content
    if "--model_path:" in content:
        # Split by spaces to get individual arguments
        parts = content.split()
        for i, part in enumerate(parts):
            if part == "--model_path:":
                if i + 1 < len(parts):
                    model_path = parts[i + 1]
                    # Convert relative path to absolute if needed
                    if model_path.startswith("../"):
                        # The config assumes we're running from src directory
                        # So ../base_models means root/base_models
                        src_dir = os.path.dirname(os.path.abspath(__file__))
                        model_path = os.path.join(src_dir, model_path)
                        model_path = os.path.normpath(model_path)
                    return model_path
    
    raise ValueError(f"Model path not found in config file: {config_path}")

def run_ablation_experiment(run_dir: str, num_trials: int = 100, 
                           layer_name: str = "conv4a", device: str = "cuda", inverse: bool = False,
                           disjoint_only: bool = False, buffer_multiplier: float = 1.0,
                           probe_path: str = None):
    """
    Run ablation experiments for each entity using the run directory.
    
    Args:
        run_dir: Directory containing intervention results and config
        num_trials: Number of trials per condition
        layer_name: Layer to ablate
        device: Device to use
        inverse: If True, run inverse ablation experiments
        disjoint_only: If True, use only non-overlapping spans (requires inverse=True)
        buffer_multiplier: Multiplier for activation range buffer
        probe_path: Optional path to trained probe for next target prediction analysis
        
    Returns:
        Dict with results including probe analysis if probe_path provided
    """
    # Get model path from config
    model_path = get_model_path_from_config(run_dir)
    print(f"Loading model from config: {model_path}")
    model = load_model(model_path, device)
    
    # Load probe if path provided
    probe = None
    probe_label_map = None
    if probe_path:
        print(f"Loading probe from: {probe_path}")
        probe, probe_label_map = load_probe(probe_path, device)
        if probe is not None:
            print(f"Probe loaded successfully. Label mapping: {probe_label_map}")
        else:
            print(f"Warning: Could not load probe from {probe_path}")
    
    # Load activation spans
    print(f"Computing activation spans from {run_dir}")
    spans_data = compute_spans_for_directory(run_dir, use_cache=True)
    activation_spans = spans_data["activation_spans"]
    
    # Entities to test - start with baseline (no ablation)
    test_entities = [None, "gem", "blue_key", "green_key", "red_key"]  # None = no ablation baseline
    
    results = []
    
    # Create directory for GIFs - check environment variable first
    gif_dir = os.environ.get('ABLATION_GIF_DIR', 'ablation_gifs')
    os.makedirs(gif_dir, exist_ok=True)
    
    for target_entity in test_entities:
        entity_name = target_entity if target_entity else "baseline"
        if disjoint_only:
            ablation_mode_str = "disjoint_ablation"
        elif inverse:
            ablation_mode_str = "inverse_ablation"
        else:
            ablation_mode_str = "ablation"
        print(f"\nTesting {entity_name} {ablation_mode_str} ({num_trials} trials)...")
        
        # Track collection rates
        collection_counts = defaultdict(int)
        
        for trial in range(num_trials):
            # Create new maze
            _, venv = create_empty_corners_maze(randomize_entities=True)
            
            # Save GIF for first trial of each condition
            save_gif = (trial == 0)
            gif_path = os.path.join(gif_dir, f"{entity_name}_{ablation_mode_str}.gif") if save_gif else None
            
            # Run rollout with ablation
            result = run_rollout_with_ablation(
                model, venv, activation_spans, target_entity, 
                layer_name, max_steps=200, device=device,  # Increased max_steps
                save_gif=save_gif, gif_path=gif_path, inverse=inverse, disjoint_only=disjoint_only,
                buffer_multiplier=buffer_multiplier, probe=probe, probe_label_map=probe_label_map
            )
            
            # Track what was collected
            for entity_name_collected in result["collected_entity_names"]:
                collection_counts[entity_name_collected] += 1
            
            # Store result with collection order
            trial_result = {
                "trial": trial,
                "ablation_target": entity_name,
                "gem_collected": result["gem_collected"],
                "num_keys_collected": result["num_keys_collected"],
                "total_steps": result["steps"],
                "entities_collected": result["collected_entity_names"],
                "collection_order": result["collection_order"]  # List of (step, entity) tuples
            }
            
            # Add probe analysis if available
            if probe is not None:
                trial_result.update({
                    "probe_accuracy": result.get("probe_accuracy", 0.0),
                    "num_valid_predictions": result.get("num_valid_predictions", 0),
                    "step_data": result.get("step_data", [])
                })
            
            results.append(trial_result)
            
            venv.close()
            
            if (trial + 1) % 10 == 0:
                print(f"  Completed {trial + 1}/{num_trials} trials")
        
        # Print summary for this condition
        print(f"\nSummary for {entity_name} {ablation_mode_str}:")
        for entity, count in collection_counts.items():
            rate = count / num_trials * 100
            print(f"  {entity}: {count}/{num_trials} ({rate:.1f}%)")
    
    # Calculate aggregate statistics for display
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    
    # Group results by ablation target for statistics
    aggregate_stats = {}
    target_order = ["baseline", "gem", "blue_key", "green_key", "red_key"]
    
    for target in target_order:
        target_results = [r for r in results if r["ablation_target"] == target]
        if target_results:
            # Calculate statistics
            gem_rate = sum(1 for r in target_results if r["gem_collected"]) / len(target_results)
            avg_keys = sum(r["num_keys_collected"] for r in target_results) / len(target_results)
            avg_steps = sum(r["total_steps"] for r in target_results) / len(target_results)
            
            # Calculate average collection times for each entity
            entity_step_times = defaultdict(list)
            for r in target_results:
                for step, entity in r["collection_order"]:
                    entity_step_times[entity].append(step)
            
            avg_collection_steps = {}
            for entity, steps in entity_step_times.items():
                avg_collection_steps[entity] = sum(steps) / len(steps) if steps else None
            
            # Calculate entity-specific collection rates
            entity_collection_rates = {}
            for entity in ["blue_key", "green_key", "red_key", "gem"]:
                collected_count = sum(1 for r in target_results if entity in r["entities_collected"])
                entity_collection_rates[entity] = collected_count / len(target_results)
            
            # Calculate probe statistics if available
            probe_stats = {}
            if probe is not None:
                probe_accuracies = [r.get("probe_accuracy", 0) for r in target_results if r.get("probe_accuracy") is not None]
                probe_valid_predictions = [r.get("num_valid_predictions", 0) for r in target_results]
                
                if probe_accuracies:
                    probe_stats = {
                        "avg_probe_accuracy": sum(probe_accuracies) / len(probe_accuracies),
                        "std_probe_accuracy": np.std(probe_accuracies),
                        "avg_valid_predictions": sum(probe_valid_predictions) / len(probe_valid_predictions)
                    }
                    
                    # Calculate confusion matrix and entity-specific accuracy
                    all_step_data = []
                    for r in target_results:
                        if r.get("step_data"):
                            all_step_data.extend(r["step_data"])
                    
                    if all_step_data:
                        # Entity-specific probe accuracy
                        entity_probe_accuracy = {}
                        probe_prediction_counts = defaultdict(int)
                        correct_predictions_by_target = defaultdict(lambda: {'correct': 0, 'total': 0})
                        
                        for step_info in all_step_data:
                            if (step_info.get("probe_prediction") is not None and 
                                step_info.get("actual_next_target") != "unknown"):
                                
                                actual = step_info["actual_next_target"]
                                predicted = step_info["probe_prediction"]
                                
                                probe_prediction_counts[predicted] += 1
                                correct_predictions_by_target[actual]['total'] += 1
                                if step_info.get("probe_correct", False):
                                    correct_predictions_by_target[actual]['correct'] += 1
                        
                        for entity, stats in correct_predictions_by_target.items():
                            if stats['total'] > 0:
                                entity_probe_accuracy[entity] = stats['correct'] / stats['total']
                        
                        probe_stats.update({
                            "probe_accuracy_by_target": entity_probe_accuracy,
                            "probe_prediction_counts": dict(probe_prediction_counts)
                        })
            
            aggregate_stats[target] = {
                "gem_collection_rate": gem_rate,
                "avg_keys_collected": avg_keys,
                "avg_total_steps": avg_steps,
                "avg_collection_steps": avg_collection_steps,
                "entity_collection_rates": entity_collection_rates,
                "num_trials": len(target_results)
            }
            
            # Add probe statistics if available
            if probe_stats:
                aggregate_stats[target]["probe_analysis"] = probe_stats
            
            # Display
            ablation_type = "INVERSE ABLATION" if inverse else "ABLATION"
            label = "NO ABLATION (BASELINE)" if target == "baseline" else f"{target.upper()} {ablation_type}"
            print(f"\n{label}:")
            
            # Show targeted vs non-targeted effects
            if target != "baseline":
                target_rate = entity_collection_rates.get(target, 0)
                mode_desc = "PRESERVED ENTITY" if inverse else "TARGET ENTITY"
                print(f"  {mode_desc} ({target}):")
                print(f"    Collection rate: {target_rate:.2%}")
                if target in avg_collection_steps and avg_collection_steps[target]:
                    print(f"    Avg collection step: {avg_collection_steps[target]:.1f}")
                
                print(f"  OTHER ENTITIES:")
                for entity in ["blue_key", "green_key", "red_key", "gem"]:
                    if entity != target:
                        rate = entity_collection_rates.get(entity, 0)
                        print(f"    {entity}: {rate:.2%}", end="")
                        if entity in avg_collection_steps and avg_collection_steps[entity]:
                            print(f" (avg step: {avg_collection_steps[entity]:.1f})")
                        else:
                            print()
            else:
                # Baseline - show all entities
                print(f"  Collection rates:")
                for entity in ["blue_key", "green_key", "red_key", "gem"]:
                    rate = entity_collection_rates.get(entity, 0)
                    print(f"    {entity}: {rate:.2%}", end="")
                    if entity in avg_collection_steps and avg_collection_steps[entity]:
                        print(f" (avg step: {avg_collection_steps[entity]:.1f})")
                    else:
                        print()
            
            print(f"  Overall stats:")
            print(f"    Gem collection rate: {gem_rate:.2%}")
            print(f"    Avg keys collected: {avg_keys:.2f}")
            print(f"    Avg total steps: {avg_steps:.1f}")
            
            # Display probe analysis if available
            if probe_stats:
                print(f"  Probe analysis:")
                print(f"    Avg probe accuracy: {probe_stats['avg_probe_accuracy']:.3f} ± {probe_stats['std_probe_accuracy']:.3f}")
                print(f"    Avg valid predictions per trial: {probe_stats['avg_valid_predictions']:.1f}")
                
                if 'probe_accuracy_by_target' in probe_stats:
                    print(f"    Probe accuracy by target entity:")
                    for entity, acc in probe_stats['probe_accuracy_by_target'].items():
                        print(f"      {entity}: {acc:.3f}")
                
                if 'probe_prediction_counts' in probe_stats:
                    total_predictions = sum(probe_stats['probe_prediction_counts'].values())
                    if total_predictions > 0:
                        print(f"    Probe prediction distribution:")
                        for entity, count in probe_stats['probe_prediction_counts'].items():
                            pct = count / total_predictions * 100
                            print(f"      {entity}: {count} ({pct:.1f}%)")
    
    # Create comprehensive output structure
    mode_desc = "Ablate only non-overlapping spans" if disjoint_only else ("Ablate everything EXCEPT target entity spans" if inverse else "Ablate only target entity spans")
    output_data = {
        "experiment_info": {
            "run_dir": run_dir,
            "num_trials": num_trials,
            "layer_name": layer_name,
            "device": device,
            "inverse_mode": inverse,
            "disjoint_only": disjoint_only,
            "buffer_multiplier": buffer_multiplier,
            "mode_description": mode_desc,
            "probe_path": probe_path,
            "probe_enabled": probe is not None,
            "probe_label_map": probe_label_map
        },
        "aggregate_statistics": aggregate_stats,
        "detailed_results": results
    }
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description="Test entity-specific ablations")
    parser.add_argument("--run_dir", type=str,
                       default="quantitative_intervention_runs/quantitative_interventions_base/base_serial_sweep_64_256_20250709_204312/run_base_conv4a_val_0-17p406613_ents_gem_blue_key_green_key_red_key_blue_lock_green_lock_red_lock_20250713_123949",
                       help="Directory containing intervention results and config")
    parser.add_argument("--num_trials", type=int, default=100,
                       help="Number of trials per condition")
    parser.add_argument("--layer", type=str, default="conv4a",
                       help="Layer to ablate")
    parser.add_argument("--output_file", type=str, default="entity_ablation_results.json",
                       help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--inverse", action="store_true",
                       help="If set, ablate everything EXCEPT the target entity spans")
    parser.add_argument("--disjoint", action="store_true",
                       help="If set, use only non-overlapping spans (requires --inverse)")
    parser.add_argument("--buffer", type=float, default=1.0,
                       help="Buffer multiplier for activation ranges (1.0 = no buffer, 1.25 = 25% buffer)")
    parser.add_argument("--probe_path", type=str, default=None,
                       help="Path to trained probe for next target prediction analysis (optional)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.disjoint and not args.inverse:
        parser.error("--disjoint requires --inverse to be set")
    
    # Run experiment
    output_data = run_ablation_experiment(
        args.run_dir,
        args.num_trials,
        args.layer,
        args.device,
        args.inverse,
        args.disjoint,
        args.buffer,
        args.probe_path
    )
    
    # Save results as JSON
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()