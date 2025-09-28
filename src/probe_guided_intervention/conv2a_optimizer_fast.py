#!/usr/bin/env python
"""
Fast Conv2a optimization for FC3 probe control.
Optimized version that doesn't repeatedly add/remove hooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv2a_activations(controller, obs_tensor):
    """Get conv2a activations from observation."""
    activations = {}

    def hook(name):
        def fn(module, input, output):
            activations[name] = output
        return fn

    handles = []
    for name, module in controller.model.named_modules():
        if 'conv2a' in name and isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(hook('conv2a'))
            handles.append(handle)
            break

    with torch.no_grad():
        _ = controller.model(obs_tensor)

    for handle in handles:
        handle.remove()

    return activations.get('conv2a')


def optimize_conv2a_multilayer_fast(controller, obs_tensor, target_entity='green_key',
                                    n_steps=1000, lr=0.5, threshold=0.8, fc3_weight=3.0):
    """
    Fast version: Register hooks ONCE, not every iteration.

    Args:
        controller: AblationController
        obs_tensor: Input observation
        target_entity: Target entity to optimize for
        n_steps: Number of optimization steps
        lr: Learning rate
        threshold: Early stopping threshold for FC3
        fc3_weight: Weight for FC3 vs other layers

    Returns:
        optimized_conv2a, best_fc3_prob
    """

    # Ensure tensors are on the same device as the model (should be CPU)
    device = next(controller.model.parameters()).device
    obs_tensor = obs_tensor.to(device)

    # Get original conv2a
    conv2a_original = get_conv2a_activations(controller, obs_tensor)
    conv2a_original = conv2a_original.to(device)

    # Initialize optimization
    conv2a_modified = nn.Parameter(conv2a_original.clone().requires_grad_(True))
    optimizer = torch.optim.Adam([conv2a_modified], lr=lr)

    # Pre-check which probes exist (do this ONCE before loop)
    has_fc1 = hasattr(controller, 'fc1_probes') and target_entity in controller.fc1_probes
    has_fc2 = hasattr(controller, 'fc2_probes') and target_entity in controller.fc2_probes
    has_fc3 = hasattr(controller, 'fc3_probes') and target_entity in controller.fc3_probes

    # Cache the probe models themselves to avoid dict lookups
    fc1_probe = controller.fc1_probes[target_entity] if has_fc1 else None
    fc2_probe = controller.fc2_probes[target_entity] if has_fc2 else None
    fc3_probe = controller.fc3_probes[target_entity] if has_fc3 else None

    # Storage for activations captured by hooks
    activations = {}

    # Register ALL hooks ONCE (conv2a replacement + layer captures)
    handles = []

    def replace_conv2a_hook(module, input, output):
        # Directly return the current conv2a_modified tensor
        return conv2a_modified

    def capture_hook(layer_name):
        def fn(module, input, output):
            activations[layer_name] = output
        return fn

    for name, module in controller.model.named_modules():
        if 'conv2a' in name and isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(replace_conv2a_hook)
            handles.append(handle)
        elif name == 'fc1' and isinstance(module, nn.Linear):
            handle = module.register_forward_hook(capture_hook('fc1'))
            handles.append(handle)
        elif name == 'fc2' and isinstance(module, nn.Linear):
            handle = module.register_forward_hook(capture_hook('fc2'))
            handles.append(handle)
        elif name == 'fc3' and isinstance(module, nn.Linear):
            handle = module.register_forward_hook(capture_hook('fc3'))
            handles.append(handle)

    best_fc3_prob = 0
    best_conv2a = None

    try:
        for step in range(n_steps):
            # Clear previous activations and do a single forward pass
            activations.clear()
            with torch.enable_grad():  # Ensure gradients are enabled
                output = controller.model(obs_tensor)

            # Extract captured activations
            fc1 = activations.get('fc1')
            fc2 = activations.get('fc2')
            fc3 = activations.get('fc3')

            # Compute losses
            losses = {}
            probs = {}

            # FC1 probe (using cached probe)
            if has_fc1 and fc1 is not None:
                fc1_logit = fc1_probe(fc1.flatten(1))
                fc1_prob = F.softmax(fc1_logit, dim=-1)[:, 1]
                losses['fc1'] = -torch.log(fc1_prob + 1e-8)
                probs['fc1'] = fc1_prob.item()

            # FC2 probe (using cached probe)
            if has_fc2 and fc2 is not None:
                fc2_logit = fc2_probe(fc2.flatten(1))
                fc2_prob = F.softmax(fc2_logit, dim=-1)[:, 1]
                losses['fc2'] = -torch.log(fc2_prob + 1e-8)
                probs['fc2'] = fc2_prob.item()

            # FC3 probe (weighted more, using cached probe)
            if has_fc3 and fc3 is not None:
                fc3_logit = fc3_probe(fc3.flatten(1))
                fc3_prob = F.softmax(fc3_logit, dim=-1)[:, 1]
                losses['fc3'] = -torch.log(fc3_prob + 1e-8) * fc3_weight
                probs['fc3'] = fc3_prob.item()

            # Combine losses (weighted average accounting for fc3_weight)
            if losses:
                total_weight = len(losses) - 1 + fc3_weight  # Account for FC3's extra weight
                total_loss = sum(losses.values()) / total_weight
            else:
                total_loss = torch.tensor(0.0, device=device)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([conv2a_modified], max_norm=1.0)
            optimizer.step()

            fc3_prob = probs.get('fc3', 0)

            if fc3_prob > best_fc3_prob:
                best_fc3_prob = fc3_prob
                best_conv2a = conv2a_modified.detach().clone()

            # Log less frequently for speed
            if step % 10 == 0:
                print(f"    Opt iter {step:3d}: FC3={fc3_prob:.1%}, FC2={probs.get('fc2', 0):.1%}, "
                      f"FC1={probs.get('fc1', 0):.1%}, loss={total_loss.item():.3f}")

            # More aggressive early stopping for speed
            if fc3_prob >= max(threshold, 0.5):  # Stop early even at 50% if threshold is lower
                print(f"    Early stop at iter {step}: FC3={fc3_prob:.1%}")
                break

    finally:
        # Clean up: remove all hooks
        for handle in handles:
            handle.remove()

    return best_conv2a if best_conv2a is not None else conv2a_modified.detach(), best_fc3_prob


# For compatibility, import the fast version as the main one
optimize_conv2a_multilayer = optimize_conv2a_multilayer_fast