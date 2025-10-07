#!/usr/bin/env python3
"""
Capture activation patterns during a rollout at the moment when the agent
is targeting red key (after collecting blue and green), and use per-seed offsets.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch

from src.utils.helpers import (
    load_interpretable_model,
    ModelActivations,
    get_device,
    generate_action,
    observation_to_rgb
)
from src.utils.create_intervention_mazes import create_cross_maze
from src.utils.entity_collection_detector import get_entity_counts, detect_collections
from src.utils import heist

def capture_red_targeting_pattern(model, model_activations, seed):
    """
    Run a rollout and capture activations during all three targeting phases:
    1. Blue targeting (from start until blue collected)
    2. Green targeting (after blue collected, before green collected)
    3. Red targeting (after blue and green collected, before red collected)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_list, venv = create_cross_maze(include_locks=False)
    obs = obs_list[0]

    state = heist.state_from_venv(venv, 0)
    entity_counts = get_entity_counts(state)

    blue_collected = False
    green_collected = False
    red_collected = False
    blue_targeting_activations = []
    green_targeting_activations = []
    red_targeting_activations = []
    collections = []

    for step in range(100):
        # Get current activations
        if len(obs.shape) == 4:
            obs_rgb = observation_to_rgb(obs[0])
        else:
            obs_rgb = observation_to_rgb(obs)

        obs_tensor = torch.tensor(obs_rgb, dtype=torch.float32).unsqueeze(0).to(get_device())
        _, activations = model_activations.run_with_cache(obs_tensor, ['conv4a'])

        if 'conv4a' in activations:
            feat = activations['conv4a']
            pooled = torch.mean(feat, dim=(1, 2)).cpu().numpy()

            # Capture activations based on current targeting phase
            if not blue_collected:
                # Targeting blue (initial phase)
                blue_targeting_activations.append(pooled)
            elif blue_collected and not green_collected:
                # Targeting green (after blue)
                green_targeting_activations.append(pooled)
            elif blue_collected and green_collected and not red_collected:
                # Targeting red (after blue and green)
                red_targeting_activations.append(pooled)

        # Generate action and step
        action = generate_action(model, obs)
        obs, reward, done, info = venv.step(action)
        obs = obs[0]

        # Check for collections
        state = heist.state_from_venv(venv, 0)
        entity_counts, collected_this_step = detect_collections(state, entity_counts)

        if collected_this_step:
            for item in collected_this_step:
                collections.append(item)
                print(f"  Step {step}: Collected {item}")

                if 'blue_key' in item:
                    blue_collected = True
                    print(f"    -> Blue collected")
                elif 'green_key' in item:
                    green_collected = True
                    print(f"    -> Green collected")
                    if blue_collected:
                        print(f"      Now targeting red key, capturing activations...")
                elif 'red_key' in item:
                    red_collected = True
                    if blue_collected and green_collected:
                        print(f"    -> Red collected. Captured {len(red_targeting_activations)} frames of red-targeting")

        if done[0]:
            break

    venv.close()

    # Average activations for each targeting phase
    avg_blue_targeting = None
    avg_green_targeting = None
    avg_red_targeting = None

    if len(blue_targeting_activations) > 0:
        avg_blue_targeting = np.mean(blue_targeting_activations, axis=0)

    if len(green_targeting_activations) > 0:
        avg_green_targeting = np.mean(green_targeting_activations, axis=0)

    if len(red_targeting_activations) > 0:
        avg_red_targeting = np.mean(red_targeting_activations, axis=0)

    # Return all three patterns
    return {
        'blue': avg_blue_targeting,
        'green': avg_green_targeting,
        'red': avg_red_targeting,
        'collections': collections
    }

def test_red_targeting_offset(model, model_activations, test_seed, offset):
    """
    Test if applying the red-targeting offset changes agent behavior.
    """
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)

    obs_list, venv = create_cross_maze(include_locks=False)
    obs = obs_list[0]

    state = heist.state_from_venv(venv, 0)
    entity_counts = get_entity_counts(state)

    # Add hook to apply offset
    def modify_conv4a(module, input, output):
        for ch in range(32):
            output[:, ch] += offset[ch]
        return output

    hook_handle = model.conv4a.register_forward_hook(modify_conv4a)

    first_collected = None
    all_collected = []

    for step in range(100):
        action = generate_action(model, obs)
        obs, reward, done, info = venv.step(action)
        obs = obs[0]

        state = heist.state_from_venv(venv, 0)
        entity_counts, collected_this_step = detect_collections(state, entity_counts)

        if collected_this_step:
            for item in collected_this_step:
                all_collected.append(item)
                if first_collected is None:
                    first_collected = item

        if done[0] or len(all_collected) >= 3:
            break

    hook_handle.remove()
    venv.close()

    return first_collected, all_collected

def run_with_intervention(model, seed, offset=None):
    """Run a single rollout with optional offset intervention."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_list, venv = create_cross_maze(include_locks=False)
    obs = obs_list[0]

    state = heist.state_from_venv(venv, 0)
    entity_counts = get_entity_counts(state)

    hook_handle = None
    if offset is not None:
        def modify_conv4a(module, input, output):
            for ch in range(32):
                output[:, ch] += offset[ch]
            return output
        hook_handle = model.conv4a.register_forward_hook(modify_conv4a)

    first_collected = None

    for step in range(100):
        action = generate_action(model, obs)
        obs, reward, done, info = venv.step(action)
        obs = obs[0]

        state = heist.state_from_venv(venv, 0)
        entity_counts, collected_this_step = detect_collections(state, entity_counts)

        if collected_this_step and first_collected is None:
            first_collected = collected_this_step[0]
            break

        if done[0]:
            break

    if hook_handle is not None:
        hook_handle.remove()

    venv.close()
    return first_collected

def main():
    # Load model
    model_path = "base_models/full_run/model_35001.0.pt"
    device = get_device()
    model = load_interpretable_model(model_path=model_path).to(device)
    model.eval()
    model_activations = ModelActivations(model)

    print("=== STEP 1: Collecting per-seed red-targeting patterns ===")

    # Target 10 successful pattern captures (test mode)
    target_count = 10
    seed_start = 200
    max_seed_attempts = 100

    # First get baseline behavior - keep trying until we have enough
    print(f"\nGetting baseline behaviors (target: {target_count})...")
    baselines = {}
    seed = seed_start

    while len(baselines) < max_seed_attempts:
        result = run_with_intervention(model, seed, offset=None)
        baselines[seed] = result
        seed += 1

        if len(baselines) % 100 == 0:
            print(f"  Progress: {len(baselines)} baselines collected")

        if len(baselines) >= max_seed_attempts:
            break

    test_seeds = list(baselines.keys())

    # Count baseline behaviors
    blue_count = sum(1 for e in baselines.values() if e and 'blue' in e)
    green_count = sum(1 for e in baselines.values() if e and 'green' in e)
    red_count = sum(1 for e in baselines.values() if e and 'red' in e)

    print(f"\nBaseline behavior ({len(test_seeds)} seeds):")
    print(f"  Blue first: {blue_count} ({blue_count/len(test_seeds)*100:.1f}%)")
    print(f"  Green first: {green_count} ({green_count/len(test_seeds)*100:.1f}%)")
    print(f"  Red first: {red_count} ({red_count/len(test_seeds)*100:.1f}%)")

    print(f"\n=== STEP 2: Capturing per-seed targeting patterns (target: {target_count}) ===")

    # For each seed, capture blue, green, and red targeting patterns
    # Keep going until we get 400 successful captures
    seed_patterns = {}
    successful_captures = 0

    seed = seed_start
    attempts = 0

    while successful_captures < target_count and attempts < max_seed_attempts:
        # Try to capture all targeting patterns
        patterns = capture_red_targeting_pattern(model, model_activations, seed)
        collections = patterns['collections']

        if (patterns['blue'] is not None and
            patterns['green'] is not None and
            patterns['red'] is not None and
            len(collections) >= 3):
            # Check if it actually went blue->green->red
            if ('blue_key' in collections[0] and 'green_key' in collections[1] and
                'red_key' in collections[2]):
                seed_patterns[seed] = patterns
                successful_captures += 1
                if successful_captures <= 3:  # Print first few for verification
                    print(f"  Seed {seed}: Captured all targeting patterns (blue->green->red)")

        seed += 1
        attempts += 1

        if attempts % 100 == 0:
            print(f"  Progress: {attempts} seeds tried, {successful_captures} patterns captured")

    print(f"\nSuccessfully captured {successful_captures} red-targeting patterns from {attempts} seeds")

    if successful_captures == 0:
        print("No red-targeting patterns captured! Cannot proceed.")
        return

    print("\n=== STEP 3: Testing per-seed red-targeting offsets ===")

    # Apply per-seed offsets to all seeds with captured patterns
    red_first_results = 0
    green_first_results = 0
    blue_first_results = 0
    none_results = 0

    tested_seeds = list(seed_patterns.keys())
    for i, seed in enumerate(tested_seeds):
        # Calculate per-seed offset: red_targeting - blue_targeting
        # This should steer from "going for blue" to "going for red"
        offset = seed_patterns[seed]['red'] - seed_patterns[seed]['blue']

        if i < 3:  # Print first few for debugging
            print(f"  Seed {seed} offset stats: mean={offset.mean():.3f}, std={offset.std():.3f}, range=[{offset.min():.3f}, {offset.max():.3f}]")

        # Test with offset
        new_behavior = run_with_intervention(model, seed, offset=offset)

        if new_behavior:
            if 'red' in new_behavior:
                red_first_results += 1
            elif 'green' in new_behavior:
                green_first_results += 1
            elif 'blue' in new_behavior:
                blue_first_results += 1
        else:
            none_results += 1

        if i < 5:  # Print first few results
            baseline = baselines.get(seed, 'unknown')
            print(f"  Seed {seed}: {baseline} -> {new_behavior}")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(tested_seeds)} tested")

    tested_count = len(tested_seeds)
    print(f"\n=== RESULTS: Per-Seed Red-Targeting Offsets ===")
    print(f"Tested on {tested_count} seeds with captured red patterns:")
    print(f"  Red first: {red_first_results} ({red_first_results/tested_count*100:.1f}%)")
    print(f"  Green first: {green_first_results} ({green_first_results/tested_count*100:.1f}%)")
    print(f"  Blue first: {blue_first_results} ({blue_first_results/tested_count*100:.1f}%)")
    print(f"  None: {none_results}")

    # Get baseline stats for same seeds
    baseline_blue = sum(1 for s in tested_seeds if baselines.get(s) and 'blue' in baselines.get(s))
    baseline_green = sum(1 for s in tested_seeds if baselines.get(s) and 'green' in baselines.get(s))
    baseline_red = sum(1 for s in tested_seeds if baselines.get(s) and 'red' in baselines.get(s))

    print(f"\nCompare to baseline (same {tested_count} seeds):")
    print(f"  Blue first: {baseline_blue/tested_count*100:.1f}% -> {blue_first_results/tested_count*100:.1f}%")
    print(f"  Green first: {baseline_green/tested_count*100:.1f}% -> {green_first_results/tested_count*100:.1f}%")
    print(f"  Red first: {baseline_red/tested_count*100:.1f}% -> {red_first_results/tested_count*100:.1f}%")

if __name__ == "__main__":
    main()