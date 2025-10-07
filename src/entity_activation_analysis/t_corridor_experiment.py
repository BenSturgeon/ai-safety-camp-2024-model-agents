#!/usr/bin/env python3
"""
T-Corridor Entity Activation Analysis

This script creates a T-shaped corridor environment and conducts activation analysis
by swapping entities at each step of a rollout to compare channel responses.

Setup:
1. Agent starts at bottom of T-corridor
2. Agent navigates up the vertical corridor, then right to the target entity
3. At each step, we create parallel environments with different entities
4. Collect and compare activations across all entity variants
5. Analyze which channels show differential responses to specific entities
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict

from src.utils.helpers import (
    load_interpretable_model,
    ModelActivations,
    observation_to_rgb,
    get_device,
    generate_action
)
from src.utils.create_intervention_mazes import create_t_corridor_maze
from src.utils import heist
from src.utils.entity_collection_detector import detect_collections

# Entity codes for swapping
ENTITY_CODES = {
    'gem': 3,
    'blue_key': 4,
    'green_key': 5,
    'red_key': 6
}

# For gem variant (gem on left), we swap only keys
KEY_ENTITY_CODES = {
    'blue_key': 4,
    'green_key': 5,
    'red_key': 6
}


def swap_entity_in_state(venv, entity_code, keep_gem=False):
    """
    Swap the target entity in the environment while preserving everything else.

    Args:
        venv: The vectorized environment
        entity_code: New entity code to place at target location
        keep_gem: If True, preserve gem on left while swapping key on right

    Returns:
        Modified observation
    """
    state = heist.state_from_venv(venv, 0)
    entities = state.state_vals.get("ents", [])

    if not entities:
        return venv.reset()

    # Identify player and all non-HUD, non-player entities
    player_row_cell = None
    player_col_cell = None
    world_entities = []  # All non-HUD, non-player entities with their positions

    for ent in entities:
        t = ent["image_type"].val
        if t == 0:  # player
            ex = ent["x"].val
            ey = ent["y"].val
            if ex is not None and ey is not None:
                if np.isfinite(ex) and np.isfinite(ey):
                    player_col_cell = int(round(ex - 0.5))
                    player_row_cell = int(round(ey - 0.5))
            continue

        if heist._is_hud_entity(ent):
            continue
        ex = ent["x"].val
        ey = ent["y"].val
        if ex is None or ey is None:
            continue
        if not (np.isfinite(ex) and np.isfinite(ey)):
            continue
        if ex < 0 or ey < 0:
            continue

        col_cell = int(round(ex - 0.5))
        row_cell = int(round(ey - 0.5))
        world_entities.append({
            'type': t,
            'col': col_cell,
            'row': row_cell
        })

    if not world_entities:
        return venv.reset()

    # Determine which entity to swap based on variant
    entity_col_cell = None
    entity_row_cell = None
    gem_col_cell = None
    gem_row_cell = None

    if keep_gem:
        # Gem variant: find gem and key
        for ent in world_entities:
            if ent['type'] == 9:  # gem
                gem_col_cell = ent['col']
                gem_row_cell = ent['row']
            elif ent['type'] == 2:  # key (to be swapped)
                entity_col_cell = ent['col']
                entity_row_cell = ent['row']

        # If we didn't find both, fall back to any entity
        if entity_col_cell is None and world_entities:
            entity_col_cell = world_entities[0]['col']
            entity_row_cell = world_entities[0]['row']
    else:
        # Standard variant: swap the single entity (should be only one)
        entity_col_cell = world_entities[0]['col']
        entity_row_cell = world_entities[0]['row']

    if entity_col_cell is None or entity_row_cell is None:
        return venv.reset()

    # Remove all (non-HUD) entities and restore player
    state.remove_all_entities()
    if player_row_cell is not None and player_col_cell is not None:
        state.set_mouse_pos(player_row_cell, player_col_cell)

    # In gem variant, always place gem at gem position and key at key position
    # In standard variant, just place the requested entity at entity position
    if keep_gem:
        # Always place gem at its position
        if gem_col_cell is not None and gem_row_cell is not None:
            state.set_entity_position(9, None, gem_col_cell, gem_row_cell)
        # Place the swapped key at the key position (only keys, never gem)
        if entity_col_cell is not None and entity_row_cell is not None and entity_code != 3:
            color_map = {4: 0, 5: 1, 6: 2}  # blue, green, red
            state.set_entity_position(2, color_map[entity_code], entity_col_cell, entity_row_cell)
    else:
        # Standard variant: place the requested entity
        if entity_col_cell is not None and entity_row_cell is not None:
            if entity_code == 3:  # Gem (type 9)
                state.set_entity_position(9, None, entity_col_cell, entity_row_cell)
            else:  # Keys (type 2, theme by color)
                color_map = {4: 0, 5: 1, 6: 2}  # blue, green, red
                state.set_entity_position(2, color_map[entity_code], entity_col_cell, entity_row_cell)

    # Apply state
    venv.env.callmethod("set_state", [state.state_bytes])
    obs = venv.reset()

    return obs


def run_rollout_with_entity_swapping(base_entity_code=4, gem_on_left=False,
                                     max_steps=12, layer='conv4a'):
    """
    Run a single rollout and at each step, swap entities to collect activations.

    Args:
        base_entity_code: Starting entity code
        gem_on_left: If True, use gem variant (left arm), else standard (right arm)
        max_steps: Maximum steps for rollout
        layer: Which layer to extract activations from

    Returns:
        Dictionary with step-by-step activation comparisons
    """
    print(f"Running T-corridor experiment (gem_on_left={gem_on_left})...")

    # Create base environment
    obs_list, venv = create_t_corridor_maze(entity_code=base_entity_code, gem_on_left=gem_on_left)
    base_obs = obs_list[0]

    # Load model - check for checkpoint argument
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--10k':
        model_path = "base_models/full_run/model_10001.0.pt"
        checkpoint = "10k"
    elif len(sys.argv) > 1 and sys.argv[1] == '--30k':
        model_path = "base_models/full_run/model_30001.0.pt"
        checkpoint = "30k"
    elif len(sys.argv) > 1 and sys.argv[1] == '--35k':
        model_path = "base_models/full_run/model_35001.0.pt"
        checkpoint = "35k"
    elif len(sys.argv) > 1 and sys.argv[1] == '--50k':
        model_path = "base_models/full_run/model_50001.0.pt"
        checkpoint = "50k"
    else:
        model_path = "base_models/full_run/model_60001.0.pt"
        checkpoint = "60k"

    device = get_device()
    model = load_interpretable_model(model_path=model_path).to(device)
    model.eval()
    print(f"Using checkpoint: {checkpoint} ({model_path})")
    model_activations = ModelActivations(model)

    # Determine which entities to swap
    if gem_on_left:
        # Gem variant: only swap keys (6 comparisons)
        entity_codes = KEY_ENTITY_CODES
    else:
        # Standard variant: swap all entities (7 comparisons including gem)
        entity_codes = ENTITY_CODES

    # Store results for each step
    results = {
        'config': {
            'base_entity': base_entity_code,
            'gem_on_left': gem_on_left,
            'max_steps': max_steps,
            'layer': layer
        },
        'steps': []
    }

    # Run rollout
    obs = base_obs
    for step in tqdm(range(max_steps), desc="Rollout steps"):
        # Get state for this step
        step_data = {
            'step': step,
            'entity_activations': {},
            'activation_diffs': {}
        }

        # For each entity variant, collect activations
        for entity_name, entity_code in entity_codes.items():
            # Swap entity
            swapped_obs = swap_entity_in_state(venv, entity_code, keep_gem=gem_on_left)

            # Extract activations
            obs_rgb = observation_to_rgb(swapped_obs)
            _, activations = model_activations.run_with_cache(obs_rgb, [layer])

            if layer in activations:
                feat = activations[layer]
                if isinstance(feat, tuple):
                    feat = feat[0]

                # Pool spatially to get per-channel activations
                # Handle both 4D (batch, ch, h, w) and 3D (ch, h, w) tensors
                if len(feat.shape) == 4:
                    pooled = feat.mean(dim=(2, 3)).detach().cpu().numpy()
                    step_data['entity_activations'][entity_name] = pooled[0].tolist()
                elif len(feat.shape) == 3:
                    pooled = feat.mean(dim=(1, 2)).detach().cpu().numpy()
                    step_data['entity_activations'][entity_name] = pooled.tolist()
                else:
                    print(f"Warning: Unexpected activation shape {feat.shape}")
                    step_data['entity_activations'][entity_name] = feat.detach().cpu().numpy().flatten().tolist()

        # Compute activation differences between entities
        entity_names = list(entity_codes.keys())
        for i, entity1 in enumerate(entity_names):
            for entity2 in entity_names[i+1:]:
                act1 = np.array(step_data['entity_activations'][entity1])
                act2 = np.array(step_data['entity_activations'][entity2])

                diff = np.abs(act1 - act2)
                diff_key = f"{entity1}_vs_{entity2}"
                step_data['activation_diffs'][diff_key] = diff.tolist()

        results['steps'].append(step_data)

        # Take action with model to continue rollout
        action = generate_action(model, obs, is_procgen_env=True)
        obs, _, done, _ = venv.step(action)

        if done[0]:
            print(f"Rollout finished at step {step}")
            break

    model_activations.clear_hooks()
    venv.close()

    return results


def analyze_activation_ranges(results):
    """
    Analyze which channels show consistent differences across entity types.

    Args:
        results: Results dictionary from run_rollout_with_entity_swapping

    Returns:
        Analysis dictionary with per-channel statistics
    """
    print("\nAnalyzing activation differences...")

    # Aggregate across all steps
    num_channels = len(results['steps'][0]['entity_activations'][
        list(results['steps'][0]['entity_activations'].keys())[0]
    ])

    channel_stats = defaultdict(lambda: {
        'max_diff': 0.0,
        'avg_diff': 0.0,
        'diff_variance': 0.0,
        'entity_pairs': defaultdict(list)
    })

    # Collect all diffs per channel
    for step_data in results['steps']:
        for diff_key, diffs in step_data['activation_diffs'].items():
            for ch_idx, diff_val in enumerate(diffs):
                channel_stats[ch_idx]['entity_pairs'][diff_key].append(diff_val)

    # Compute statistics
    for ch_idx in range(num_channels):
        all_diffs = []
        for diff_list in channel_stats[ch_idx]['entity_pairs'].values():
            all_diffs.extend(diff_list)

        if all_diffs:
            channel_stats[ch_idx]['max_diff'] = float(np.max(all_diffs))
            channel_stats[ch_idx]['avg_diff'] = float(np.mean(all_diffs))
            channel_stats[ch_idx]['diff_variance'] = float(np.var(all_diffs))

    # Sort channels by max difference
    sorted_channels = sorted(
        channel_stats.items(),
        key=lambda x: x[1]['max_diff'],
        reverse=True
    )

    analysis = {
        'channel_stats': dict(channel_stats),
        'top_discriminative_channels': [
            {
                'channel': ch_idx,
                'max_diff': stats['max_diff'],
                'avg_diff': stats['avg_diff'],
                'diff_variance': stats['diff_variance']
            }
            for ch_idx, stats in sorted_channels[:20]  # Top 20
        ]
    }

    return analysis


def main():
    """Run both variants of the experiment."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--10k':
        checkpoint = "10k"
    elif len(sys.argv) > 1 and sys.argv[1] == '--30k':
        checkpoint = "30k"
    elif len(sys.argv) > 1 and sys.argv[1] == '--35k':
        checkpoint = "35k"
    elif len(sys.argv) > 1 and sys.argv[1] == '--50k':
        checkpoint = "50k"
    else:
        checkpoint = "60k"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "src/entity_activation_analysis/results"
    os.makedirs(output_dir, exist_ok=True)

    # Variant 1: Standard T (entity on right arm)
    print("\n=== Running Standard T-Corridor (entity on right) ===")
    results_standard = run_rollout_with_entity_swapping(
        base_entity_code=4,  # Start with blue key
        gem_on_left=False,
        max_steps=12,
        layer='conv4a'
    )

    analysis_standard = analyze_activation_ranges(results_standard)
    results_standard['analysis'] = analysis_standard

    # Save results
    output_file_standard = f"{output_dir}/t_corridor_standard_{checkpoint}_{timestamp}.json"
    with open(output_file_standard, 'w') as f:
        json.dump(results_standard, f, indent=2)
    print(f"\nStandard variant results saved to: {output_file_standard}")

    # Variant 2: Gem variant (gem on left arm, swap keys on right)
    print("\n=== Running Gem Variant T-Corridor (gem on left, keys on right) ===")
    results_gem_variant = run_rollout_with_entity_swapping(
        base_entity_code=4,  # Start with blue key on right, gem is on left
        gem_on_left=True,
        max_steps=12,
        layer='conv4a'
    )

    analysis_gem_variant = analyze_activation_ranges(results_gem_variant)
    results_gem_variant['analysis'] = analysis_gem_variant

    # Save results
    output_file_gem = f"{output_dir}/t_corridor_gem_variant_{checkpoint}_{timestamp}.json"
    with open(output_file_gem, 'w') as f:
        json.dump(results_gem_variant, f, indent=2)
    print(f"\nGem variant results saved to: {output_file_gem}")

    # Print summary
    print("\n=== SUMMARY ===")
    print("\nStandard variant - Top 5 discriminative channels:")
    for item in analysis_standard['top_discriminative_channels'][:5]:
        print(f"  Channel {item['channel']}: max_diff={item['max_diff']:.3f}, "
              f"avg_diff={item['avg_diff']:.3f}")

    print("\nGem variant - Top 5 discriminative channels:")
    for item in analysis_gem_variant['top_discriminative_channels'][:5]:
        print(f"  Channel {item['channel']}: max_diff={item['max_diff']:.3f}, "
              f"avg_diff={item['avg_diff']:.3f}")


if __name__ == "__main__":
    main()
