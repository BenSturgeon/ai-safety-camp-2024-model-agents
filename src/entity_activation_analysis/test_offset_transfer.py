#!/usr/bin/env python3
"""
Test if learned offsets can transform one entity's activations to another's.
Place green key on left, blue key on right, then apply blue->green offset
to see if the model behaves as if seeing green key instead of blue.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.helpers import (
    load_interpretable_model,
    ModelActivations,
    observation_to_rgb,
    get_device,
    generate_action
)
from src.utils.create_intervention_mazes import create_t_corridor_maze
from src.utils.entity_collection_detector import detect_collections
from src.utils import heist

def calculate_entity_offsets(checkpoint='30k'):
    """
    Calculate the average offset between entity activations from existing data.
    Returns offsets for converting blue key -> other entities.
    """
    # Load the T-corridor results to get entity-specific activations
    results_dir = "src/entity_activation_analysis/results"

    # Find the most recent standard T-corridor result for this checkpoint
    import glob
    pattern = f"{results_dir}/t_corridor_standard_{checkpoint}_*.json"
    files = glob.glob(pattern)

    if not files:
        print(f"No T-corridor results found for checkpoint {checkpoint}")
        return None

    latest_file = sorted(files)[-1]
    print(f"Loading offsets from: {latest_file}")

    with open(latest_file, 'r') as f:
        data = json.load(f)

    # Calculate average activations for each entity across all steps
    entity_means = {}
    for entity in ['blue_key', 'green_key', 'red_key', 'gem']:
        activations_list = []
        for step_data in data['steps']:
            if entity in step_data['entity_activations']:
                activations_list.append(step_data['entity_activations'][entity])

        if activations_list:
            entity_means[entity] = np.mean(activations_list, axis=0)

    # Calculate offsets from blue key to other entities
    offsets = {}
    if 'blue_key' in entity_means:
        for entity in ['green_key', 'red_key', 'gem']:
            if entity in entity_means:
                # Offset = target - source
                offsets[f'blue_to_{entity}'] = entity_means[entity] - entity_means['blue_key']

    return offsets, entity_means

def run_offset_transfer_experiment(checkpoint='30k'):
    """
    Main experiment: Apply offsets to see if we can make the model
    treat blue key as if it were green key.
    """
    print(f"\n{'='*60}")
    print(f"OFFSET TRANSFER EXPERIMENT - Checkpoint {checkpoint}")
    print(f"{'='*60}")

    # Load model
    checkpoint_map = {
        '10k': "base_models/full_run/model_10001.0.pt",
        '30k': "base_models/full_run/model_30001.0.pt",
        '35k': "base_models/full_run/model_35001.0.pt",
        '60k': "base_models/full_run/model_60001.0.pt"
    }

    model_path = checkpoint_map.get(checkpoint, checkpoint_map['30k'])
    device = get_device()
    model = load_interpretable_model(model_path=model_path).to(device)
    model.eval()
    print(f"Loaded model: {model_path}")

    # Calculate offsets from previous experiment
    offsets, entity_means = calculate_entity_offsets(checkpoint)
    if not offsets:
        print("Failed to calculate offsets")
        return

    print(f"\nCalculated offsets from blue key:")
    for key, offset in offsets.items():
        mean_abs_offset = np.mean(np.abs(offset))
        print(f"  {key}: mean absolute offset = {mean_abs_offset:.3f}")

    # Create test environment with two keys
    print("\nCreating T-corridor with green key (left) and blue key (right)...")
    obs_list, venv = create_t_corridor_maze(
        two_keys=True,
        left_key=5,   # Green key on left
        right_key=4   # Blue key on right
    )
    obs = obs_list[0]

    # Create model with hooks for intervention
    model_activations = ModelActivations(model)

    # Store results
    results = {
        'checkpoint': checkpoint,
        'offsets': {k: v.tolist() for k, v in offsets.items()},
        'rollouts': {}
    }

    # Function to apply offset during forward pass
    def apply_offset_hook(offset_vector):
        def hook_fn(module, input, output):
            # Add offset to all spatial positions
            if isinstance(output, tuple):
                feat = output[0]
            else:
                feat = output

            # Apply offset (broadcast across spatial dimensions)
            offset_tensor = torch.tensor(offset_vector, device=feat.device, dtype=feat.dtype)

            # Handle different tensor shapes
            if len(feat.shape) == 4:  # (batch, channels, height, width)
                offset_tensor = offset_tensor.view(1, -1, 1, 1)
            elif len(feat.shape) == 3:  # (channels, height, width)
                offset_tensor = offset_tensor.view(-1, 1, 1)

            modified = feat + offset_tensor

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return hook_fn

    # Run different scenarios
    scenarios = [
        ('baseline', None),
        ('blue_to_green', offsets.get('blue_to_green_key')),
        ('blue_to_red', offsets.get('blue_to_red_key')),
    ]

    layer_name = 'conv4a'

    for scenario_name, offset in scenarios:
        print(f"\n--- Running {scenario_name} scenario ---")

        # Reset environment
        obs_list, venv = create_t_corridor_maze(
            two_keys=True,
            left_key=5,   # Green key on left
            right_key=4   # Blue key on right
        )
        obs = obs_list[0]

        # Track initial counts for collection detection
        prev_counts = None

        # Clear any existing hooks
        for module in model.modules():
            module._forward_hooks.clear()

        # Apply offset if specified
        hook_handle = None
        if offset is not None:
            # Register hook on conv4a layer
            for name, module in model.named_modules():
                if name == 'conv4a':
                    hook_handle = module.register_forward_hook(apply_offset_hook(offset))
                    print(f"    Applied offset hook to {name}")
                    break

        # Run a short rollout
        trajectory = []
        max_steps = 20
        collected_entity = None

        for step in range(max_steps):
            # Generate action
            action = generate_action(model, obs, is_procgen_env=True)

            # Extract scalar if it's a list
            if isinstance(action, list) or isinstance(action, np.ndarray):
                action = int(action[0]) if len(action) > 0 else 0

            # Only print every 5 steps to reduce output
            if step % 5 == 0 or step < 2:
                print(f"    Step {step}: action={action}")

            trajectory.append({
                'step': step,
                'action': action
            })

            # Take action
            obs, reward, done, info = venv.step(np.array([action]))

            # Check for collections
            curr_state = heist.state_from_venv(venv, 0)
            curr_counts, collected_list = detect_collections(curr_state, prev_counts)

            if collected_list:
                collected_entity = collected_list[0]
                print(f"  Step {step}: Collected {collected_entity}!")
                trajectory[-1]['collected'] = collected_entity
                break

            prev_counts = curr_counts

            if done[0]:
                print(f"  Episode ended at step {step}")
                break

        # Track which direction the agent primarily went
        left_actions = sum(1 for t in trajectory if t['action'] == 1)  # Action 1 = left
        right_actions = sum(1 for t in trajectory if t['action'] == 2)  # Action 2 = right

        results['rollouts'][scenario_name] = {
            'trajectory': trajectory,
            'collected': collected_entity,
            'left_actions': left_actions,
            'right_actions': right_actions,
            'primary_direction': 'left' if left_actions > right_actions else 'right' if right_actions > left_actions else 'neutral'
        }

        # Remove hook if it was added
        if hook_handle is not None:
            hook_handle.remove()

    # Analyze results
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    for scenario_name, data in results['rollouts'].items():
        collected = data['collected']
        direction = data['primary_direction']

        if collected:
            print(f"{scenario_name:15} -> Collected {collected:10} (went {direction}, L:{data['left_actions']} R:{data['right_actions']})")
        else:
            print(f"{scenario_name:15} -> No collection (went {direction}, L:{data['left_actions']} R:{data['right_actions']})")

    # Check if offset successfully changed behavior
    baseline_result = results['rollouts']['baseline']
    offset_result = results['rollouts']['blue_to_green']

    success = False
    if baseline_result['collected'] == 'blue_key' and offset_result['collected'] == 'green_key':
        success = True
        print(f"\n🎉 SUCCESS! Offset changed behavior from blue key to green key!")
    elif baseline_result['primary_direction'] != offset_result['primary_direction']:
        print(f"\n⚠️  PARTIAL SUCCESS: Direction changed from {baseline_result['primary_direction']} to {offset_result['primary_direction']}")
    else:
        print(f"\n❌ No significant behavior change detected")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"src/entity_activation_analysis/results/offset_transfer_{checkpoint}_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Create visualization
    visualize_offset_results(results, checkpoint)

    model_activations.clear_hooks()
    venv.close()

    return results, success

def visualize_offset_results(results, checkpoint):
    """Create visualization showing which key the agent pursues in each scenario."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scenarios = ['baseline', 'blue_to_green', 'blue_to_red']
    scenario_titles = [
        'Baseline (No Offset)',
        'Blue→Green Offset Applied',
        'Blue→Red Offset Applied'
    ]
    colors_map = {
        'baseline': 'gray',
        'blue_to_green': 'green',
        'blue_to_red': 'red'
    }

    for idx, (scenario, title) in enumerate(zip(scenarios, scenario_titles)):
        ax = axes[idx]

        if scenario not in results['rollouts']:
            continue

        data = results['rollouts'][scenario]

        # Extract collection info
        collected = data['collected']
        left_actions = data['left_actions']
        right_actions = data['right_actions']

        # Create simple bar chart showing direction preference
        directions = ['Left\n(Green)', 'Right\n(Blue)']
        counts = [left_actions, right_actions]

        bars = ax.bar(directions, counts, color=['green', 'blue'], alpha=0.6)

        # Highlight which was collected
        if collected == 'green_key':
            bars[0].set_edgecolor('black')
            bars[0].set_linewidth(3)
        elif collected == 'blue_key':
            bars[1].set_edgecolor('black')
            bars[1].set_linewidth(3)

        ax.set_ylabel('Action Count')
        ax.set_title(title)
        ax.set_ylim(0, max(10, max(counts) + 2))

        # Add collection result
        if collected:
            result_text = f"Collected: {collected.replace('_', ' ').title()}"
            color = 'green' if 'green' in collected else 'blue' if 'blue' in collected else 'red'
        else:
            result_text = f"No collection"
            color = 'gray'

        ax.text(0.5, 0.95, result_text,
                transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                fontweight='bold', fontsize=11)

    plt.suptitle(f'Offset Transfer Experiment - {checkpoint} Checkpoint\n'
                 f'Testing if activation offsets can redirect agent behavior',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = f"src/entity_activation_analysis/plots/offset_transfer_{checkpoint}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

def main():
    """Run offset transfer experiments for different checkpoints."""

    # Parse command line arguments
    checkpoint = '30k'  # Default
    if len(sys.argv) > 1:
        if sys.argv[1] == '--30k':
            checkpoint = '30k'
        elif sys.argv[1] == '--35k':
            checkpoint = '35k'
        elif sys.argv[1] == '--60k':
            checkpoint = '60k'

    # Run the experiment
    results, success = run_offset_transfer_experiment(checkpoint)

    print(f"\n{'='*60}")
    print("KEY FINDING:")
    if success:
        print("✅ The network uses static offsets to encode entity identity!")
        print("   Adding the blue→green offset made the agent pursue the green key.")
    else:
        print("⚠️  Offset transfer did not produce clear behavior change.")
        print("   This checkpoint may use more complex encoding strategies.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()