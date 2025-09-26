"""
Rollout experiments for testing ablation effects on agent behavior.
"""

import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import os

from src.utils.helpers import save_observations_as_gif
from src.utils.entity_collection_detector import detect_collections
from src.utils.create_intervention_mazes import create_cross_maze
import src.utils.heist as heist


def test_green_key_first(ablator, max_steps=100, save_gif=True):
    """
    Test if ablations successfully steer agent to collect green_key first.
    Success = green_key collected before any other entity.

    Args:
        ablator: Ablator instance with model and probes
        max_steps: Maximum steps in rollout
        save_gif: Whether to save rollout as GIF

    Returns:
        dict: Results including success status and collection order
    """
    print(f"\n{'='*60}")
    print("GREEN KEY FIRST TEST")
    print("Success = Collect green_key before any other entity")
    print('='*60)

    target_entity = 'green_key'

    # Create cross maze
    _, venv = create_cross_maze(include_locks=False)
    obs = venv.reset()

    # Initialize entity tracking
    state = heist.state_from_venv(venv, 0)
    entity_counts = None

    # Store observations for GIF
    observations_for_gif = []

    # Track collections
    collections = []
    first_entity_collected = None

    # Process first observation
    if isinstance(obs, np.ndarray):
        obs_array = obs
    else:
        obs_array = np.asarray(obs)

    if len(obs_array.shape) == 4:
        obs_array = obs_array[0]

    # Initialize ablation mask
    obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)

    with torch.no_grad():
        _, conv3a = ablator.get_conv3a_with_hook(obs_tensor)
        if conv3a is not None:
            ablator.ablation_mask = torch.ones_like(conv3a[0])
            ablator.total_neurons = ablator.ablation_mask.numel()
            ablator.ablated_count = 0

    # Run rollout
    for step in range(max_steps):
        if save_gif:
            observations_for_gif.append(obs_array.copy())

        # Convert observation
        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        print(f"\nStep {step}:")

        # Get initial probe prediction
        with torch.no_grad():
            output, conv3a = ablator.get_conv3a_with_hook(obs_tensor)

            if conv3a is not None:
                flat_conv3a = conv3a.view(1, -1)
                probe_out = ablator.probe(flat_conv3a)
                probe_probs = F.softmax(probe_out, dim=-1)
                pred_idx = probe_probs.argmax().item()
                predicted_entity = ablator.label_map.get(pred_idx, 'unknown')

                # Get confidence for target entity
                target_idx = ablator.reverse_map.get(target_entity)
                target_confidence = probe_probs[0, target_idx].item() if target_idx is not None else 0.0
                max_confidence = probe_probs.max().item()
            else:
                predicted_entity = 'unknown'
                target_confidence = 0.0
                max_confidence = 0.0

            # Get action
            if isinstance(output, tuple):
                logits = output[0].logits
            else:
                logits = output.logits
            action_probs = F.softmax(logits, dim=-1)
            action = action_probs.argmax().item()

        print(f"  Initial: Probe={predicted_entity} ({max_confidence:.1%}), Target_conf={target_confidence:.1%}, Action={action}")

        # If not already predicting target with high confidence, optimize ablations
        if target_confidence < 0.9:
            # Store initial mask state
            initial_ablated = (ablator.ablation_mask == 0).sum().item()

            # Use random search to find optimal ablations
            ablator.ablation_mask = ablator.find_optimal_ablations_random(obs_tensor, target_entity, max_trials=20)
            ablator.ablated_count = (ablator.ablation_mask == 0).sum().item()

            # Report what changed
            new_ablations = ablator.ablated_count - initial_ablated
            if new_ablations > 0:
                print(f"  >>> Added {new_ablations} new ablations this step")

            # Get final prediction after optimization
            with torch.no_grad():
                output, conv3a = ablator.get_conv3a_with_hook(obs_tensor)
                if conv3a is not None:
                    flat_conv3a = conv3a.view(1, -1)
                    probe_out = ablator.probe(flat_conv3a)
                    probe_probs = F.softmax(probe_out, dim=-1)
                    pred_idx = probe_probs.argmax().item()
                    predicted_entity = ablator.label_map.get(pred_idx, 'unknown')
                    target_confidence = probe_probs[0, target_idx].item() if target_idx is not None else 0.0
                    max_confidence = probe_probs.max().item()

                    # Get updated action
                    if isinstance(output, tuple):
                        logits = output[0].logits
                    else:
                        logits = output.logits
                    action_probs = F.softmax(logits, dim=-1)
                    action = action_probs.argmax().item()
        else:
            print(f"  ✓ Already predicting {target_entity} with {target_confidence:.1%} confidence")

        # Log final state
        print(f"  Final: Probe={predicted_entity} ({max_confidence:.1%}), Target_conf={target_confidence:.1%}, Action={action}, Ablated={ablator.ablated_count}/{ablator.total_neurons}")

        # Take action
        obs, reward, done, info = venv.step(np.array([action]))

        # Check for collections
        state = heist.state_from_venv(venv, 0)
        entity_counts, collected_this_step = detect_collections(state, entity_counts)

        # Track collections
        if collected_this_step:
            for entity in collected_this_step:
                collections.append((step, entity))
                if first_entity_collected is None:
                    first_entity_collected = entity
                print(f"  >>> COLLECTED: {entity} <<<")

                # Check success/failure
                if entity == 'green_key':
                    print(f"\n🎉 SUCCESS! Green key collected first at step {step}")
                    done = [True]  # End episode
                else:
                    print(f"\n❌ FAILURE: {entity} collected before green_key at step {step}")
                    done = [True]  # End episode

        # Update observation
        if isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = np.asarray(obs)

        if len(obs_array.shape) == 4:
            obs_array = obs_array[0]

        # Check if done
        is_done = done[0] if isinstance(done, np.ndarray) else done
        if is_done:
            break

    # Save GIF
    gif_path = None
    if save_gif and observations_for_gif:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = f'src/probe_guided_intervention/results/green_key_test_{timestamp}.gif'
        os.makedirs('src/probe_guided_intervention/results', exist_ok=True)

        save_observations_as_gif(
            observations_for_gif,
            filepath=gif_path,
            fps=10,
            enhance_size=True
        )
        print(f"\n📹 Saved test GIF to: {gif_path}")

    # Clean up
    venv.close()

    # Report results
    success = first_entity_collected == 'green_key'
    results = {
        'success': success,
        'first_collected': first_entity_collected,
        'collections': collections,
        'total_steps': step + 1,
        'total_ablated': ablator.ablated_count,
        'ablation_percentage': ablator.ablated_count / ablator.total_neurons * 100 if ablator.total_neurons > 0 else 0,
        'gif_path': gif_path
    }

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print('='*60)
    print(f"Success: {'✓' if success else '✗'}")
    print(f"First collected: {first_entity_collected or 'Nothing'}")
    print(f"All collections: {collections}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Ablations: {results['total_ablated']}/{ablator.total_neurons} ({results['ablation_percentage']:.1f}%)")

    return results


def test_conv4a_optimization_rollout(ablator, max_steps=30, reoptimize_each_step=True):
    """
    Test conv4a optimization during rollout with re-optimization at each step.
    """
    print(f"\n{'='*60}")
    print("CONV4A RE-OPTIMIZATION ROLLOUT TEST")
    print(f"Re-optimize each step: {reoptimize_each_step}")
    print('='*60)

    # Create cross maze
    _, venv = create_cross_maze(include_locks=False)
    obs = venv.reset()

    # Convert observation
    if isinstance(obs, np.ndarray):
        obs_array = obs
    else:
        obs_array = np.asarray(obs)
    if len(obs_array.shape) == 4:
        obs_array = obs_array[0]

    # Initialize tracking
    state = heist.state_from_venv(venv, 0)
    entity_counts = None
    observations_for_gif = []
    collections = []
    first_entity_collected = None

    # Initialize ablation mask
    obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)

    with torch.no_grad():
        _, conv3a, conv4a, _ = ablator.get_conv_layers_with_hook(obs_tensor)
        if conv3a is not None:
            ablator.ablation_mask = torch.ones_like(conv3a[0])
            ablator.total_neurons = ablator.ablation_mask.numel()

    # Run rollout
    for step in range(max_steps):
        observations_for_gif.append(obs_array.copy())

        # Convert observation
        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        print(f"\nStep {step}:")

        # Optimize ablations for conv4a prediction
        if step == 0 or reoptimize_each_step:
            print(f"  Re-optimizing ablations for conv4a...")
            ablator.ablation_mask = ablator.find_optimal_ablations_for_conv4a(
                obs_tensor, 'green_key', max_trials=30
            )
            ablator.ablated_count = (ablator.ablation_mask == 0).sum().item()

        # Get predictions with current ablations
        with torch.no_grad():
            output, conv3a, conv4a, _ = ablator.get_conv_layers_with_hook(obs_tensor)

            # Check conv3a probe
            if conv3a is not None:
                conv3a_flat = conv3a.view(1, -1)
                conv3a_out = ablator.probe(conv3a_flat)
                conv3a_probs = F.softmax(conv3a_out, dim=-1)
                conv3a_pred_idx = conv3a_probs.argmax().item()
                conv3a_entity = ablator.label_map.get(conv3a_pred_idx, 'unknown')
                conv3a_conf = conv3a_probs.max().item()

                # Get green_key confidence
                green_idx = ablator.reverse_map.get('green_key')
                conv3a_green_conf = conv3a_probs[0, green_idx].item() if green_idx is not None else 0.0

            # Check conv4a probe
            if conv4a is not None:
                conv4a_flat = conv4a.view(1, -1)
                conv4a_out = ablator.conv4a_probe(conv4a_flat)
                conv4a_probs = F.softmax(conv4a_out, dim=-1)
                conv4a_pred_idx = conv4a_probs.argmax().item()
                conv4a_entity = ablator.label_map.get(conv4a_pred_idx, 'unknown')
                conv4a_conf = conv4a_probs.max().item()

                # Get green_key confidence
                conv4a_green_conf = conv4a_probs[0, green_idx].item() if green_idx is not None else 0.0

            # Get action
            if isinstance(output, tuple):
                logits = output[0].logits
            else:
                logits = output.logits
            action = F.softmax(logits, dim=-1).argmax().item()

        print(f"  Conv3a: {conv3a_entity} ({conv3a_conf:.1%}), green_key={conv3a_green_conf:.1%}")
        print(f"  Conv4a: {conv4a_entity} ({conv4a_conf:.1%}), green_key={conv4a_green_conf:.1%}")
        print(f"  Action: {action}, Ablated: {ablator.ablated_count}/{ablator.total_neurons}")

        # Take action
        obs, reward, done, info = venv.step(np.array([action]))

        # Check for collections
        state = heist.state_from_venv(venv, 0)
        entity_counts, collected_this_step = detect_collections(state, entity_counts)

        if collected_this_step:
            for entity in collected_this_step:
                collections.append((step, entity))
                if first_entity_collected is None:
                    first_entity_collected = entity
                print(f"  >>> COLLECTED: {entity} <<<")

                if entity == 'green_key':
                    print(f"\n🎉 SUCCESS! Green key collected at step {step}")
                    done = [True]
                else:
                    print(f"\n❌ FAILURE: {entity} collected before green_key")
                    done = [True]

        # Update observation
        if isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = np.asarray(obs)
        if len(obs_array.shape) == 4:
            obs_array = obs_array[0]

        # Check if done
        is_done = done[0] if isinstance(done, np.ndarray) else done
        if is_done:
            break

    # Save GIF
    if observations_for_gif:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = f'src/probe_guided_intervention/results/conv4a_reoptimize_{timestamp}.gif'
        os.makedirs('src/probe_guided_intervention/results', exist_ok=True)

        save_observations_as_gif(
            observations_for_gif,
            filepath=gif_path,
            fps=10,
            enhance_size=True
        )
        print(f"\n📹 Saved GIF to: {gif_path}")

    venv.close()

    # Report results
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print('='*60)
    print(f"First entity collected: {first_entity_collected}")
    print(f"Collections: {collections}")
    print(f"Success: {'✓' if first_entity_collected == 'green_key' else '✗'}")

    return {
        'success': first_entity_collected == 'green_key',
        'first_collected': first_entity_collected,
        'collections': collections,
        'gif_path': gif_path if observations_for_gif else None
    }


def test_fc1_optimization_rollout(ablator, max_steps=30, reoptimize_each_step=True):
    """
    Test fc1 optimization during rollout with optional re-optimization at each step.
    """
    print(f"\n{'='*60}")
    print("FC1 OPTIMIZATION ROLLOUT TEST")
    print(f"Re-optimize each step: {reoptimize_each_step}")
    print('='*60)

    # Create cross maze
    _, venv = create_cross_maze(include_locks=False)
    obs = venv.reset()

    # Convert observation
    if isinstance(obs, np.ndarray):
        obs_array = obs
    else:
        obs_array = np.asarray(obs)
    if len(obs_array.shape) == 4:
        obs_array = obs_array[0]

    # Initialize tracking
    state = heist.state_from_venv(venv, 0)
    entity_counts = None
    observations_for_gif = []
    collections = []
    first_entity_collected = None

    # Initialize ablation mask
    obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)

    with torch.no_grad():
        _, conv3a, _, fc1 = ablator.get_conv_layers_with_hook(obs_tensor)
        if conv3a is not None:
            ablator.ablation_mask = torch.ones_like(conv3a[0])
            ablator.total_neurons = ablator.ablation_mask.numel()

    # Run rollout
    for step in range(max_steps):
        observations_for_gif.append(obs_array.copy())

        # Convert observation
        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        print(f"\nStep {step}:")

        # Optimize ablations for fc1 prediction
        if step == 0 or reoptimize_each_step:
            print(f"  Re-optimizing ablations for fc1...")
            ablator.ablation_mask = ablator.find_optimal_ablations_for_fc1(
                obs_tensor, 'green_key', max_trials=30
            )
            ablator.ablated_count = (ablator.ablation_mask == 0).sum().item()

        # Get predictions with current ablations
        with torch.no_grad():
            output, conv3a, conv4a, fc1 = ablator.get_conv_layers_with_hook(obs_tensor)

            # Check fc1 probes for all entities
            if fc1 is not None:
                fc1_flat = fc1.view(1, -1)

                print(f"  FC1 predictions:")
                for entity, probe in ablator.fc1_probes.items():
                    out = probe(fc1_flat)
                    probs = F.softmax(out, dim=-1)
                    conf = probs[0, 1].item()
                    print(f"    {entity}: {conf:.1%}")

            # Get action
            if isinstance(output, tuple):
                logits = output[0].logits
            else:
                logits = output.logits
            action = F.softmax(logits, dim=-1).argmax().item()

        print(f"  Action: {action}, Ablated: {ablator.ablated_count}/{ablator.total_neurons}")

        # Take action
        obs, reward, done, info = venv.step(np.array([action]))

        # Check for collections
        state = heist.state_from_venv(venv, 0)
        entity_counts, collected_this_step = detect_collections(state, entity_counts)

        if collected_this_step:
            for entity in collected_this_step:
                collections.append((step, entity))
                if first_entity_collected is None:
                    first_entity_collected = entity
                print(f"  >>> COLLECTED: {entity} <<<")

                if entity == 'green_key':
                    print(f"\n🎉 SUCCESS! Green key collected at step {step}")
                    done = [True]
                else:
                    print(f"\n❌ FAILURE: {entity} collected before green_key")
                    done = [True]

        # Update observation
        if isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = np.asarray(obs)
        if len(obs_array.shape) == 4:
            obs_array = obs_array[0]

        # Check if done
        is_done = done[0] if isinstance(done, np.ndarray) else done
        if is_done:
            break

    # Save GIF
    if observations_for_gif:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = f'src/probe_guided_intervention/results/fc1_reoptimize_{timestamp}.gif'
        os.makedirs('src/probe_guided_intervention/results', exist_ok=True)

        save_observations_as_gif(
            observations_for_gif,
            filepath=gif_path,
            fps=10,
            enhance_size=True
        )
        print(f"\n📹 Saved GIF to: {gif_path}")

    venv.close()

    # Report results
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print('='*60)
    print(f"First entity collected: {first_entity_collected}")
    print(f"Collections: {collections}")
    print(f"Success: {'✓' if first_entity_collected == 'green_key' else '✗'}")

    return {
        'success': first_entity_collected == 'green_key',
        'first_collected': first_entity_collected,
        'collections': collections,
        'gif_path': gif_path if observations_for_gif else None
    }