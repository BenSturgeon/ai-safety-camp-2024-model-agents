"""
Unified ablation optimization for all layers.
Supports continuous masks with amplification [0, 2] where:
  0 = fully ablate
  1 = normal (no change)
  2 = double activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_optimization_score(target_conf, other_confs, target_entity, action_changed=False, mask=None, allow_amplification=False):
    """
    Unified scoring function for all optimization methods.

    Score = 2.0*target - 0.5*others_sum - 0.3*others_max + 0.1*action_changed - regularization
    """
    # Get other entity confidences
    others = {e: c for e, c in other_confs.items() if e != target_entity}
    others_sum = sum(others.values())
    others_max = max(others.values()) if others else 0

    # Base score: maximize target, minimize others
    score = (
        2.0 * target_conf           # Strong weight on target
        - 0.5 * others_sum           # Penalize sum of others
        - 0.3 * others_max           # Extra penalty for strongest competitor
        + 0.1 * int(action_changed)  # Bonus for changing action
    )

    # Regularization penalty for extreme modulations
    if mask is not None and allow_amplification:
        modulation_penalty = (mask - 1.0).abs().mean().item()
        score -= 0.05 * modulation_penalty  # Small penalty for deviation from normal

    return score


def generate_continuous_mask(shape, strategy='random', trial=0, allow_amplification=True):
    """
    Generate continuous mask values for any layer shape.

    Strategies:
    - random: Fully random values
    - sparse_amplify: Mostly normal, some amplified
    - sparse_ablate: Mostly normal, some ablated
    - mixed: Various patterns (beta, normal, gradient)
    - targeted: Strong changes on few neurons
    """
    if allow_amplification:
        # Values in [0, 2] where 1 = normal
        if strategy == 'random':
            mask = torch.rand(shape) * 2.0

        elif strategy == 'sparse_amplify':
            mask = 0.8 + torch.rand(shape) * 0.4  # Base: 0.8-1.2
            if len(shape) == 1:  # FC layer
                num_amplify = min(5, shape[0] // 3)
                indices = torch.randperm(shape[0])[:num_amplify]
                mask[indices] = 1.3 + torch.rand(num_amplify) * 0.7
            else:  # Conv layer
                num_amplify = min(20, shape[0] // 4)
                channels = torch.randperm(shape[0])[:num_amplify]
                for c in channels:
                    mask[c] = 1.3 + torch.rand_like(mask[c]) * 0.7

        elif strategy == 'sparse_ablate':
            mask = 0.8 + torch.rand(shape) * 0.4  # Base: 0.8-1.2
            if len(shape) == 1:  # FC layer
                num_ablate = min(5, shape[0] // 3)
                indices = torch.randperm(shape[0])[:num_ablate]
                mask[indices] = torch.rand(num_ablate) * 0.7
            else:  # Conv layer
                num_ablate = min(20, shape[0] // 4)
                channels = torch.randperm(shape[0])[:num_ablate]
                for c in channels:
                    mask[c] = torch.rand_like(mask[c]) * 0.7

        elif strategy == 'mixed':
            if trial % 3 == 0:
                # Beta distribution for smooth patterns
                alpha, beta = 2.0, 2.0
                mask = torch.distributions.Beta(alpha, beta).sample(shape) * 2.0
            elif trial % 3 == 1:
                # Normal distribution centered at 1.0
                mask = torch.normal(mean=1.0, std=0.4, size=shape)
                mask = torch.clamp(mask, 0.0, 2.0)
            else:
                # Gradient patterns
                if len(shape) == 1:
                    mask = torch.linspace(0.3, 1.7, shape[0])
                    mask = mask[torch.randperm(shape[0])]
                else:
                    mask = 0.5 + torch.rand(shape) * 1.0

        else:  # 'targeted'
            mask = 0.9 + torch.rand(shape) * 0.2  # Base: 0.9-1.1
            if len(shape) == 1:
                num_strong = min(3, shape[0] // 5)
                indices = torch.randperm(shape[0])[:num_strong]
                if trial % 2 == 0:
                    mask[indices] = torch.rand(num_strong) * 0.3
                else:
                    mask[indices] = 1.7 + torch.rand(num_strong) * 0.3
            else:
                num_strong = min(10, shape[0] // 8)
                channels = torch.randperm(shape[0])[:num_strong]
                for c in channels:
                    if trial % 2 == 0:
                        mask[c] = torch.rand_like(mask[c]) * 0.3
                    else:
                        mask[c] = 1.7 + torch.rand_like(mask[c]) * 0.3
    else:
        # Traditional ablation only [0, 1]
        if strategy == 'random':
            mask = torch.rand(shape)
        elif strategy == 'sparse':
            mask = torch.ones(shape)
            sparsity = np.random.uniform(0.05, 0.3)
            ablate = torch.rand(shape) < sparsity
            mask[ablate] = 0
        else:
            mask = torch.rand(shape)

    return mask


def unified_layer_optimization(controller, obs, target_entity, layer_name='conv3a',
                               max_trials=1000, allow_amplification=True, verbose=True):
    """
    Unified optimization function for any layer.

    Args:
        controller: AblationController instance
        obs: Observation tensor
        target_entity: Target entity to optimize for
        layer_name: Which layer to optimize ('conv3a', 'conv4a', 'fc1', 'fc2', 'fc3')
        max_trials: Number of optimization trials
        allow_amplification: Whether to allow values > 1 (up to 2)
        verbose: Whether to print progress

    Returns:
        best_mask: Optimized mask for the specified layer
    """
    # Set controller to correct layer
    controller.ablation_layer = layer_name

    # Get layer dimensions and setup
    with torch.no_grad():
        _, conv3a, conv4a, fc1, fc2, fc3 = controller.get_conv_layers_with_hook(obs)

        layer_map = {
            'conv3a': (conv3a, controller.probe if hasattr(controller, 'probe') else None, True),
            'conv4a': (conv4a, controller.conv4a_probe if hasattr(controller, 'conv4a_probe') else None, True),
            'fc1': (fc1, controller.fc1_probes if hasattr(controller, 'fc1_probes') else {}, False),
            'fc2': (fc2, controller.fc2_probes if hasattr(controller, 'fc2_probes') else {}, False),
            'fc3': (fc3, controller.fc3_probes if hasattr(controller, 'fc3_probes') else {}, False),
        }

        if layer_name not in layer_map:
            raise ValueError(f"Unknown layer: {layer_name}")

        layer_output, probes, is_multiclass = layer_map[layer_name]

        if layer_output is None:
            print(f"ERROR: Could not get {layer_name} activations")
            return None

        shape = layer_output[0].shape

    # Initialize mask on controller
    mask_attr = {
        'conv3a': 'ablation_mask',
        'conv4a': 'conv4a_ablation_mask',
        'fc1': 'fc1_ablation_mask',
        'fc2': 'fc2_ablation_mask',
        'fc3': 'fc3_ablation_mask'
    }[layer_name]

    setattr(controller, mask_attr, torch.ones(shape).float())

    # Get baseline
    baseline_confs = _get_layer_predictions(controller, obs, layer_name, probes, is_multiclass)
    baseline_action = _get_action(controller, obs)

    if verbose:
        print(f"\nOptimizing {layer_name} for {target_entity}")
        print(f"  Shape: {shape}, Baseline: {baseline_confs.get(target_entity, 0):.1%}")
        print(f"  Mode: {'Amplification [0,2]' if allow_amplification else 'Ablation [0,1]'}")

    # Optimization loop
    best_mask = torch.ones(shape).float()
    best_score = -float('inf')
    best_target_conf = baseline_confs.get(target_entity, 0.0)

    strategies = ['random', 'sparse_amplify', 'sparse_ablate', 'mixed', 'targeted']
    trials_per_strategy = max_trials // len(strategies)

    for strategy in strategies:
        if verbose and trials_per_strategy > 50:
            print(f"  Strategy: {strategy}")

        for trial in range(trials_per_strategy):
            # Generate and apply mask
            test_mask = generate_continuous_mask(shape, strategy, trial, allow_amplification)
            setattr(controller, mask_attr, test_mask)

            # Evaluate
            test_confs = _get_layer_predictions(controller, obs, layer_name, probes, is_multiclass)
            new_action = _get_action(controller, obs)
            action_changed = (new_action != baseline_action)

            # Score
            target_conf = test_confs.get(target_entity, 0.0)
            score = compute_optimization_score(
                target_conf, test_confs, target_entity,
                action_changed, test_mask, allow_amplification
            )

            # Update best
            if score > best_score or (target_conf > best_target_conf + 0.1):
                best_score = score
                best_target_conf = target_conf
                best_mask = test_mask.clone()

                if verbose and (trial % 100 == 0 or target_conf > 0.7):
                    others_max = max((c for e, c in test_confs.items() if e != target_entity), default=0)
                    print(f"    New best: {target_entity}={target_conf:.1%}, others_max={others_max:.1%}, score={score:.2f}")

                # Early exit
                if target_conf > 0.9:
                    if verbose:
                        print(f"  ✓ Achieved >90% confidence!")
                    return best_mask

    if verbose:
        print(f"  Final: {target_entity}={best_target_conf:.1%}")

    return best_mask


def _get_layer_predictions(controller, obs, layer_name, probes, is_multiclass):
    """Helper to get predictions for a layer."""
    with torch.no_grad():
        _, conv3a, conv4a, fc1, fc2, fc3 = controller.get_conv_layers_with_hook(obs)

        layer_outputs = {
            'conv3a': conv3a,
            'conv4a': conv4a,
            'fc1': fc1,
            'fc2': fc2,
            'fc3': fc3
        }

        output = layer_outputs[layer_name]
        if output is None:
            return {}

        output_flat = output.view(1, -1)
        confs = {}

        if is_multiclass:
            # Single multi-class probe
            if probes is not None:
                logits = probes(output_flat)
                probs = F.softmax(logits, dim=-1)
                # Get all entity confidences from label map
                if hasattr(controller, 'label_map'):
                    for idx, entity_name in controller.label_map.items():
                        confs[entity_name] = probs[0, idx].item()
        else:
            # Binary probes
            for entity, probe in probes.items():
                logits = probe(output_flat)
                probs = F.softmax(logits, dim=-1)
                confs[entity] = probs[0, 1].item()

    return confs


def _get_action(controller, obs):
    """Helper to get action from model."""
    with torch.no_grad():
        model_output = controller.model(obs)
        if isinstance(model_output, tuple):
            logits = model_output[0].logits
        else:
            logits = model_output.logits
        return F.softmax(logits, dim=-1)[0].argmax().item()


# ============================================================================
# DEPRECATED: Old functions kept for backwards compatibility
# Use unified_layer_optimization instead!
# ============================================================================

def find_optimal_ablations_random(model, probe, ablation_mask, obs, target_entity,
                                   get_conv3a_hook_fn, label_map, reverse_map, max_trials=30):
    """DEPRECATED: Use unified_layer_optimization instead."""
    # Minimal implementation for compatibility
    print("WARNING: find_optimal_ablations_random is deprecated. Use unified_layer_optimization.")
    return ablation_mask


def find_optimal_ablations_for_conv4a(ablation_mask, obs, target_entity, get_conv_layers_hook_fn,
                                      probe, conv4a_probe, label_map, reverse_map, max_trials=50):
    """DEPRECATED: Use unified_layer_optimization instead."""
    print("WARNING: find_optimal_ablations_for_conv4a is deprecated. Use unified_layer_optimization.")
    return ablation_mask


def find_optimal_ablations_for_fc1(ablation_mask, obs, fc1_probes, get_conv_layers_hook_fn,
                                   target_entity='green_key', max_trials=100):
    """DEPRECATED: Use unified_layer_optimization instead."""
    print("WARNING: find_optimal_ablations_for_fc1 is deprecated. Use unified_layer_optimization.")
    return ablation_mask


def find_optimal_ablations_all_layers(ablation_mask, obs, get_all_layers_hook_fn, conv4a_probe,
                                      fc1_probes, fc2_probes, label_map, target_entity='green_key',
                                      max_trials=30):
    """DEPRECATED: Use unified_layer_optimization instead."""
    print("WARNING: find_optimal_ablations_all_layers is deprecated. Use unified_layer_optimization.")
    return ablation_mask


def find_optimal_ablations_for_fc3_probes(ablation_mask, obs, fc3_probes, get_conv_layers_hook_fn,
                                          target_entity='green_key', max_trials=500):
    """DEPRECATED: Use unified_layer_optimization instead."""
    print("WARNING: find_optimal_ablations_for_fc3_probes is deprecated. Use unified_layer_optimization.")
    return ablation_mask


def find_optimal_ablations_for_fc2_to_fc3(fc2_shape, obs, fc3_probes, controller,
                                          target_entity='green_key', max_trials=2000, continuous=True):
    """DEPRECATED: Use unified_layer_optimization instead."""
    print("WARNING: find_optimal_ablations_for_fc2_to_fc3 is deprecated. Use unified_layer_optimization.")
    return torch.ones(fc2_shape)


def find_optimal_ablations_for_fc3_direct(fc3_shape, obs, fc3_probes, controller,
                                          target_entity='green_key', max_trials=2000, allow_amplification=True):
    """DEPRECATED: Use unified_layer_optimization instead."""
    print("WARNING: find_optimal_ablations_for_fc3_direct is deprecated. Use unified_layer_optimization.")
    return torch.ones(fc3_shape)