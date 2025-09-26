"""
Ablation optimization methods for different layers and probe types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def find_optimal_ablations_random(model, probe, ablation_mask, obs, target_entity,
                                   get_conv3a_hook_fn, label_map, reverse_map, max_trials=30):
    """
    Use random search to find ablations that make probe predict target entity.
    Random ablation patterns work better than gradient-based selection!

    Args:
        model: The neural network model
        probe: The probe to optimize for
        ablation_mask: Current ablation mask
        obs: Single observation tensor [1, 3, 64, 64]
        target_entity: Entity we want the probe to predict
        get_conv3a_hook_fn: Function to get conv3a with hook
        label_map: Entity to index mapping
        reverse_map: Index to entity mapping
        max_trials: Number of random patterns to try

    Returns:
        improved_mask: Boolean mask of neurons to keep (1) or ablate (0)
    """
    obs.requires_grad = False

    # Get baseline prediction
    with torch.no_grad():
        _, conv3a = get_conv3a_hook_fn(obs)
        if conv3a is None:
            raise RuntimeError("Failed to get conv3a activations")

        flat_conv3a = conv3a.view(1, -1)
        probe_outputs = probe(flat_conv3a)
        probe_probs = F.softmax(probe_outputs, dim=-1)

        target_idx = reverse_map.get(target_entity)
        if target_idx is None:
            raise ValueError(f"Target entity {target_entity} not in label map")

        baseline_target_prob = probe_probs[0, target_idx].item()

    # Start with current mask
    best_mask = ablation_mask.clone()
    best_prob = baseline_target_prob
    best_is_target = False

    print(f"    Starting random search. Baseline P({target_entity})={baseline_target_prob:.1%}")

    # Random search: try different ablation patterns
    for trial in range(max_trials):
        # Create random ablation pattern
        test_mask = torch.ones_like(ablation_mask)

        # Random sparsity between 5% and 30%
        sparsity = np.random.uniform(0.05, 0.3)
        ablate_positions = torch.rand_like(test_mask) < sparsity
        test_mask[ablate_positions] = 0

        # Apply test mask and check improvement
        # Note: This needs to be integrated with the model's forward hook
        # For now, returning the mask for the main class to apply
        with torch.no_grad():
            # Temporarily apply mask (this would be done via hook in main class)
            old_mask = ablation_mask.clone()
            ablation_mask.copy_(test_mask)

            _, conv3a_test = get_conv3a_hook_fn(obs)
            flat_test = conv3a_test.view(1, -1)
            probe_out_test = probe(flat_test)
            probs_test = F.softmax(probe_out_test, dim=-1)
            test_prob = probs_test[0, target_idx].item()

            # Restore old mask
            ablation_mask.copy_(old_mask)

        # Check if this helps make target the argmax
        test_pred_idx = probs_test.argmax().item()
        test_is_target = (test_pred_idx == target_idx)

        # Accept if target becomes argmax or probability improves significantly
        if test_is_target and (not best_is_target or test_prob > best_prob):
            # Found a pattern that makes target the argmax!
            best_mask = test_mask.clone()
            best_prob = test_prob
            best_is_target = True
            num_ablated = (test_mask == 0).sum().item()
            print(f"    Trial {trial}: SUCCESS! Predicts {target_entity}, P={test_prob:.1%}, Ablated={num_ablated}")

            # If confidence is high enough, stop
            if test_prob > 0.9:
                print(f"    Achieved >90% confidence!")
                break
        elif test_prob > best_prob + 0.1 and not best_is_target:
            # Significant improvement but not argmax yet
            best_mask = test_mask.clone()
            best_prob = test_prob
            num_ablated = (test_mask == 0).sum().item()
            print(f"    Trial {trial}: Improved P({target_entity})={test_prob:.1%}, Ablated={num_ablated}")

    # Final summary
    final_ablated = (best_mask == 0).sum().item()
    print(f"    Random search complete: Final P({target_entity})={best_prob:.1%}, Total ablated={final_ablated}")

    return best_mask


def find_optimal_ablations_for_conv4a(ablation_mask, obs, target_entity,
                                       get_conv_layers_hook_fn, probe, conv4a_probe,
                                       label_map, reverse_map, max_trials=50):
    """
    Find conv3a ablations that make CONV4A probe predict target entity.
    This is the key insight - we ablate conv3a to affect conv4a predictions!

    Args:
        ablation_mask: Current conv3a ablation mask
        obs: Single observation tensor [1, 3, 64, 64]
        target_entity: Entity we want conv4a probe to predict
        get_conv_layers_hook_fn: Function to get conv layers with hook
        probe: Conv3a probe (for reference)
        conv4a_probe: Conv4a probe to optimize
        label_map: Entity to index mapping
        reverse_map: Index to entity mapping
        max_trials: Number of random patterns to try

    Returns:
        improved_mask: Boolean mask of neurons to keep (1) or ablate (0)
    """
    obs.requires_grad = False

    # Get baseline predictions from BOTH probes
    with torch.no_grad():
        _, conv3a, conv4a, _ = get_conv_layers_hook_fn(obs)
        if conv3a is None or conv4a is None:
            raise RuntimeError("Failed to get conv activations")

        # Check conv4a probe baseline
        conv4a_flat = conv4a.view(1, -1)
        conv4a_outputs = conv4a_probe(conv4a_flat)
        conv4a_probs = F.softmax(conv4a_outputs, dim=-1)

        target_idx = reverse_map.get(target_entity)
        if target_idx is None:
            raise ValueError(f"Target entity {target_entity} not in label map")

        baseline_conv4a_prob = conv4a_probs[0, target_idx].item()

    # Start with current mask
    best_mask = ablation_mask.clone()
    best_conv4a_prob = baseline_conv4a_prob
    best_is_target = False

    print(f"    Conv4a optimization: Baseline P({target_entity})={baseline_conv4a_prob:.1%}")

    # Random search: try different ablation patterns
    for trial in range(max_trials):
        # Create random ablation pattern
        test_mask = torch.ones_like(ablation_mask)

        # Try wider range of sparsity for conv4a effect
        sparsity = np.random.uniform(0.1, 0.5)
        ablate_positions = torch.rand_like(test_mask) < sparsity
        test_mask[ablate_positions] = 0

        # Apply test mask and check conv4a predictions
        # Note: This needs integration with hooks in main class
        with torch.no_grad():
            old_mask = ablation_mask.clone()
            ablation_mask.copy_(test_mask)

            _, conv3a_test, conv4a_test, _ = get_conv_layers_hook_fn(obs)

            # Check conv4a probe (this is what we're optimizing)
            conv4a_flat = conv4a_test.view(1, -1)
            conv4a_out = conv4a_probe(conv4a_flat)
            conv4a_probs = F.softmax(conv4a_out, dim=-1)
            conv4a_target_prob = conv4a_probs[0, target_idx].item()

            # Restore old mask
            ablation_mask.copy_(old_mask)

        # Check if this helps make target the argmax in conv4a
        conv4a_pred_idx = conv4a_probs.argmax().item()
        conv4a_is_target = (conv4a_pred_idx == target_idx)

        # Accept if conv4a probe improves
        if conv4a_is_target and (not best_is_target or conv4a_target_prob > best_conv4a_prob):
            # Found a pattern that makes conv4a predict target!
            best_mask = test_mask.clone()
            best_conv4a_prob = conv4a_target_prob
            best_is_target = True
            num_ablated = (test_mask == 0).sum().item()

            # Also check conv3a for reference
            conv3a_flat = conv3a_test.view(1, -1)
            conv3a_out = probe(conv3a_flat)
            conv3a_probs = F.softmax(conv3a_out, dim=-1)
            conv3a_prob = conv3a_probs[0, target_idx].item()

            print(f"    Trial {trial}: CONV4A SUCCESS! Conv4a={conv4a_target_prob:.1%}, Conv3a={conv3a_prob:.1%}, Ablated={num_ablated}")

            # If conv4a confidence is high enough, stop
            if conv4a_target_prob > 0.7:  # Lower threshold since conv4a is harder
                print(f"    Achieved >70% conv4a confidence!")
                break
        elif conv4a_target_prob > best_conv4a_prob + 0.05 and not best_is_target:
            # Improvement in conv4a probability
            best_mask = test_mask.clone()
            best_conv4a_prob = conv4a_target_prob
            num_ablated = (test_mask == 0).sum().item()
            print(f"    Trial {trial}: Conv4a improved to {conv4a_target_prob:.1%}, Ablated={num_ablated}")

    # Final summary
    final_ablated = (best_mask == 0).sum().item()
    print(f"    Conv4a optimization complete: Final P({target_entity})={best_conv4a_prob:.1%}, Total ablated={final_ablated}")

    return best_mask


def find_optimal_ablations_for_fc1(ablation_mask, obs, fc1_probes, get_conv_layers_hook_fn,
                                    target_entity='green_key', max_trials=50):
    """
    Find conv3a ablations that maximize fc1 probe confidence for target entity
    while minimizing confidence for all other entities.

    Args:
        ablation_mask: Current conv3a ablation mask
        obs: Single observation tensor [1, 3, 64, 64]
        fc1_probes: Dictionary of binary probes for each entity
        get_conv_layers_hook_fn: Function to get conv layers with hook
        target_entity: Entity we want fc1 probes to predict (default: green_key)
        max_trials: Number of random patterns to try

    Returns:
        improved_mask: Boolean mask of neurons to keep (1) or ablate (0)
    """
    obs.requires_grad = False

    # Get baseline predictions from fc1 probes
    with torch.no_grad():
        _, conv3a, _, fc1 = get_conv_layers_hook_fn(obs)
        if conv3a is None or fc1 is None:
            raise RuntimeError("Failed to get activations")

        # Get baseline confidences for all entities
        fc1_flat = fc1.view(1, -1)
        baseline_confidences = {}
        for entity, probe in fc1_probes.items():
            output = probe(fc1_flat)
            # Binary probe outputs [neg_logit, pos_logit], we want P(entity present)
            probs = F.softmax(output, dim=-1)
            confidence = probs[0, 1].item()  # Probability of class 1 (entity present)
            baseline_confidences[entity] = confidence

        print(f"    FC1 baseline confidences:")
        for entity, conf in baseline_confidences.items():
            print(f"      {entity}: {conf:.1%}")

    # Start with current mask
    best_mask = ablation_mask.clone()
    best_target_conf = baseline_confidences.get(target_entity, 0.0)
    best_others_sum = sum(conf for entity, conf in baseline_confidences.items() if entity != target_entity)
    best_score = best_target_conf - 0.5 * best_others_sum  # Reward target, penalize others

    print(f"    Starting optimization for {target_entity}")
    print(f"    Baseline score: {best_score:.3f} (target={best_target_conf:.1%}, others_sum={best_others_sum:.1%})")

    # Random search: try different ablation patterns
    for trial in range(max_trials):
        # Create random ablation pattern
        test_mask = torch.ones_like(ablation_mask)

        # Try wider range of sparsity for fc1 effect
        sparsity = np.random.uniform(0.2, 0.6)
        ablate_positions = torch.rand_like(test_mask) < sparsity
        test_mask[ablate_positions] = 0

        # Apply test mask and check fc1 predictions
        with torch.no_grad():
            old_mask = ablation_mask.clone()
            ablation_mask.copy_(test_mask)

            _, _, _, fc1_test = get_conv_layers_hook_fn(obs)

            # Check all fc1 probes
            fc1_flat = fc1_test.view(1, -1)
            test_confidences = {}
            for entity, probe in fc1_probes.items():
                output = probe(fc1_flat)
                probs = F.softmax(output, dim=-1)
                confidence = probs[0, 1].item()  # Probability of class 1 (entity present)
                test_confidences[entity] = confidence

            # Restore old mask
            ablation_mask.copy_(old_mask)

        # Calculate score: maximize target, minimize others
        target_conf = test_confidences.get(target_entity, 0.0)
        others_sum = sum(conf for entity, conf in test_confidences.items() if entity != target_entity)
        score = target_conf - 0.5 * others_sum

        # Accept if score improves
        if score > best_score:
            best_mask = test_mask.clone()
            best_target_conf = target_conf
            best_others_sum = others_sum
            best_score = score
            num_ablated = (test_mask == 0).sum().item()

            print(f"    Trial {trial}: IMPROVED! Target={target_conf:.1%}, Others_sum={best_others_sum:.1%}, Score={score:.3f}, Ablated={num_ablated}")

            # If target is very high and others are low, stop
            if target_conf > 0.8 and others_sum < 1.0:
                print(f"    Achieved excellent separation!")
                break

    # Final summary
    final_ablated = (best_mask == 0).sum().item()
    print(f"    FC1 optimization complete:")
    print(f"      Target ({target_entity}): {best_target_conf:.1%}")
    print(f"      Others sum: {best_others_sum:.1%}")
    print(f"      Final score: {best_score:.3f}")
    print(f"      Total ablated: {final_ablated}")

    return best_mask