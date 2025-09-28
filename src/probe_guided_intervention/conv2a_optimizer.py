#!/usr/bin/env python
"""
Conv2a optimization for FC3 probe control.
Optimizing earlier layers (conv2a) provides better control over FC3 predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_conv2a_fc3_loss(conv2a_modified, controller, obs_tensor, target='green_key', threshold=0.8):
    """
    Compute loss for conv2a modifications to control FC3 probes.

    Args:
        conv2a_modified: Modified conv2a activations
        controller: AblationController with FC3 probes
        obs_tensor: Input observation
        target: Target entity to optimize for

    Returns:
        loss, metrics dict
    """

    # Set up hook to replace conv2a with modified version
    def replace_conv2a_hook(module, input, output):
        return conv2a_modified

    # Register hooks
    handles = []
    for name, module in controller.model.named_modules():
        if 'conv2a' in name and isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(replace_conv2a_hook)
            handles.append(handle)

    # Get FC3 activations with hook
    fc3_activations = {}
    def get_fc3_hook(module, input, output):
        fc3_activations['fc3'] = output

    for name, module in controller.model.named_modules():
        if name == 'fc3' and isinstance(module, nn.Linear):
            fc3_handle = module.register_forward_hook(get_fc3_hook)
            handles.append(fc3_handle)

    # Forward pass
    _ = controller.model(obs_tensor)
    fc3 = fc3_activations.get('fc3')

    # Remove all hooks
    for handle in handles:
        handle.remove()

    # Compute FC3 probe predictions
    key_probs = {}
    if fc3 is not None and hasattr(controller, 'fc3_probes'):
        for entity in ['green_key', 'blue_key', 'red_key']:
            if entity in controller.fc3_probes:
                logit = controller.fc3_probes[entity](fc3.flatten(1))
                prob = F.softmax(logit, dim=-1)[:, 1]
                key_probs[entity] = prob

    # Create competitive distribution - treat as logits and apply softmax
    if key_probs:
        # Stack all probabilities and treat them as logits for softmax
        all_probs = torch.stack(list(key_probs.values()))
        # Apply softmax to create distribution that sums to 1
        entity_distribution = F.softmax(all_probs * 5.0, dim=0)  # Scale up for sharper distribution

        # Map back to entities
        entities = list(key_probs.keys())
        target_idx = entities.index(target) if target in entities else None

        if target_idx is not None:
            target_dist_prob = entity_distribution[target_idx]
            # Loss to maximize target while others compete
            loss = -torch.log(target_dist_prob + 1e-8)

            # Additional penalty if target is below threshold
            if target_dist_prob < threshold:
                loss += 5.0 * (threshold - target_dist_prob) ** 2
        else:
            loss = torch.tensor(10.0)  # High loss if target not found
    else:
        loss = torch.tensor(10.0)

    # Small change penalty
    conv2a_original = getattr(controller, 'conv2a_original', conv2a_modified)
    change_penalty = 0.001 * ((conv2a_modified - conv2a_original) ** 2).mean()
    loss += change_penalty

    target_prob = key_probs.get(target, torch.tensor(0.0))

    metrics = {
        'target_prob': target_prob.item() if torch.is_tensor(target_prob) else 0,
        'other_probs': {k: v.item() for k, v in key_probs.items() if k != target}
    }

    return loss, metrics


def get_conv2a_activations(controller, obs_tensor):
    """Extract conv2a activations from the model."""
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().clone()
        return hook

    handles = []
    for name, module in controller.model.named_modules():
        if 'conv2a' in name and isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(get_activation('conv2a'))
            handles.append(handle)

    with torch.no_grad():
        _ = controller.model(obs_tensor)
        conv2a = activations.get('conv2a')

    for handle in handles:
        handle.remove()

    return conv2a


def compute_conv2a_multilayer_loss(conv2a_modified, controller, obs_tensor, target='green_key', fc3_weight=3.0):
    """
    Compute loss using all layer probes with extra weighting for FC3.

    Args:
        conv2a_modified: Modified conv2a activations
        controller: AblationController with probes
        obs_tensor: Input observation
        target: Target entity to optimize for
        fc3_weight: How much more to weight FC3 vs other layers

    Returns:
        loss, metrics dict
    """
    # Set up hook to replace conv2a with modified version
    def replace_conv2a_hook(module, input, output):
        return conv2a_modified

    # Register hooks
    handles = []
    for name, module in controller.model.named_modules():
        if 'conv2a' in name and isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(replace_conv2a_hook)
            handles.append(handle)

    # Get all layer activations
    output, conv3a, conv4a, fc1, fc2, fc3 = controller.get_conv_layers_with_hook(obs_tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Compute losses for all layers
    losses = {}
    probs = {}

    # FC1 probe (weight = 1)
    if hasattr(controller, 'fc1_probes') and target in controller.fc1_probes:
        fc1_logit = controller.fc1_probes[target](fc1.flatten(1))
        fc1_prob = F.softmax(fc1_logit, dim=-1)[:, 1]
        losses['fc1'] = -torch.log(fc1_prob + 1e-8)
        probs['fc1'] = fc1_prob.item()

    # FC2 probe (weight = 1)
    if hasattr(controller, 'fc2_probes') and target in controller.fc2_probes:
        fc2_logit = controller.fc2_probes[target](fc2.flatten(1))
        fc2_prob = F.softmax(fc2_logit, dim=-1)[:, 1]
        losses['fc2'] = -torch.log(fc2_prob + 1e-8)
        probs['fc2'] = fc2_prob.item()

    # FC3 probe (weight = fc3_weight)
    if hasattr(controller, 'fc3_probes') and target in controller.fc3_probes:
        fc3_logit = controller.fc3_probes[target](fc3.flatten(1))
        fc3_prob = F.softmax(fc3_logit, dim=-1)[:, 1]
        losses['fc3'] = -torch.log(fc3_prob + 1e-8) * fc3_weight
        probs['fc3'] = fc3_prob.item()

    # Weighted average loss
    if losses:
        total_weight = len(losses) - 1 + fc3_weight  # Account for FC3's extra weight
        total_loss = sum(losses.values()) / total_weight
    else:
        total_loss = torch.tensor(10.0)

    # Small change penalty
    conv2a_original = getattr(controller, 'conv2a_original', conv2a_modified)
    change_penalty = 0.001 * ((conv2a_modified - conv2a_original) ** 2).mean()
    total_loss += change_penalty

    # Get other entity probs for FC3 (for logging)
    other_probs = {}
    if fc3 is not None and hasattr(controller, 'fc3_probes'):
        for entity in ['green_key', 'blue_key', 'red_key']:
            if entity != target and entity in controller.fc3_probes:
                logit = controller.fc3_probes[entity](fc3.flatten(1))
                prob = F.softmax(logit, dim=-1)[:, 1]
                other_probs[entity] = prob.item()

    metrics = {
        'target_prob': probs.get('fc3', 0),  # Main metric is FC3
        'fc1_prob': probs.get('fc1', 0),
        'fc2_prob': probs.get('fc2', 0),
        'fc3_prob': probs.get('fc3', 0),
        'other_probs': other_probs
    }

    return total_loss, metrics


def compute_conv2a_multilayer_loss_with_suppression(conv2a_modified, controller, obs_tensor, target='green_key', fc3_weight=3.0):
    """
    Compute loss using all layer probes with competitive suppression.

    This version uses softmax across all entity probabilities to create
    competition - maximizing target while suppressing others.

    Args:
        conv2a_modified: Modified conv2a activations
        controller: AblationController with probes
        obs_tensor: Input observation
        target: Target entity to optimize for
        fc3_weight: How much more to weight FC3 vs other layers

    Returns:
        loss, metrics dict
    """
    # Set up hook to replace conv2a with modified version
    def replace_conv2a_hook(module, input, output):
        return conv2a_modified

    # Register hooks
    handles = []
    for name, module in controller.model.named_modules():
        if 'conv2a' in name and isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(replace_conv2a_hook)
            handles.append(handle)

    # Get all layer activations
    output, conv3a, conv4a, fc1, fc2, fc3 = controller.get_conv_layers_with_hook(obs_tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Compute losses for all layers
    losses = {}
    probs = {}

    # FC1 probe (weight = 1) - with suppression
    if hasattr(controller, 'fc1_probes'):
        fc1_key_probs = {}
        for entity in ['green_key', 'blue_key', 'red_key']:
            if entity in controller.fc1_probes:
                logit = controller.fc1_probes[entity](fc1.flatten(1))
                prob = F.softmax(logit, dim=-1)[:, 1]
                fc1_key_probs[entity] = prob

        if target in fc1_key_probs and len(fc1_key_probs) > 1:
            # Apply competitive softmax
            all_probs = torch.stack(list(fc1_key_probs.values()))
            entity_distribution = F.softmax(all_probs * 5.0, dim=0)

            # Find target index
            target_idx = list(fc1_key_probs.keys()).index(target)
            losses['fc1'] = -torch.log(entity_distribution[target_idx] + 1e-8)
            probs['fc1'] = fc1_key_probs[target].item()

    # FC2 probe (weight = 1) - with suppression
    if hasattr(controller, 'fc2_probes'):
        fc2_key_probs = {}
        for entity in ['green_key', 'blue_key', 'red_key']:
            if entity in controller.fc2_probes:
                logit = controller.fc2_probes[entity](fc2.flatten(1))
                prob = F.softmax(logit, dim=-1)[:, 1]
                fc2_key_probs[entity] = prob

        if target in fc2_key_probs and len(fc2_key_probs) > 1:
            # Apply competitive softmax
            all_probs = torch.stack(list(fc2_key_probs.values()))
            entity_distribution = F.softmax(all_probs * 5.0, dim=0)

            # Find target index
            target_idx = list(fc2_key_probs.keys()).index(target)
            losses['fc2'] = -torch.log(entity_distribution[target_idx] + 1e-8)
            probs['fc2'] = fc2_key_probs[target].item()

    # FC3 probe (weight = fc3_weight) - with suppression
    if hasattr(controller, 'fc3_probes'):
        fc3_key_probs = {}
        for entity in ['green_key', 'blue_key', 'red_key']:
            if entity in controller.fc3_probes:
                logit = controller.fc3_probes[entity](fc3.flatten(1))
                prob = F.softmax(logit, dim=-1)[:, 1]
                fc3_key_probs[entity] = prob

        if target in fc3_key_probs and len(fc3_key_probs) > 1:
            # Apply competitive softmax
            all_probs = torch.stack(list(fc3_key_probs.values()))
            entity_distribution = F.softmax(all_probs * 5.0, dim=0)

            # Find target index
            target_idx = list(fc3_key_probs.keys()).index(target)
            losses['fc3'] = -torch.log(entity_distribution[target_idx] + 1e-8) * fc3_weight
            probs['fc3'] = fc3_key_probs[target].item()

    # Weighted average loss
    if losses:
        total_weight = len(losses) - 1 + fc3_weight  # Account for FC3's extra weight
        total_loss = sum(losses.values()) / total_weight
    else:
        total_loss = torch.tensor(10.0)

    # Small change penalty
    conv2a_original = getattr(controller, 'conv2a_original', conv2a_modified)
    change_penalty = 0.001 * ((conv2a_modified - conv2a_original) ** 2).mean()
    total_loss += change_penalty

    # Get other entity probs for FC3 (for logging)
    other_probs = {}
    if fc3 is not None and hasattr(controller, 'fc3_probes'):
        for entity in ['green_key', 'blue_key', 'red_key']:
            if entity != target and entity in controller.fc3_probes:
                logit = controller.fc3_probes[entity](fc3.flatten(1))
                prob = F.softmax(logit, dim=-1)[:, 1]
                other_probs[entity] = prob.item()

    metrics = {
        'target_prob': probs.get('fc3', 0),  # Main metric is FC3
        'fc1_prob': probs.get('fc1', 0),
        'fc2_prob': probs.get('fc2', 0),
        'fc3_prob': probs.get('fc3', 0),
        'other_probs': other_probs
    }

    return total_loss, metrics


def optimize_conv2a_for_fc3(controller, obs_tensor, target_entity='green_key',
                            n_steps=50, lr=0.5, threshold=0.8):
    """
    Optimize conv2a to control FC3 predictions.

    Args:
        controller: AblationController
        obs_tensor: Input observation
        target_entity: Target entity to optimize for
        n_steps: Number of optimization steps
        lr: Learning rate
        threshold: Early stopping threshold

    Returns:
        optimized_conv2a, best_prob
    """

    # Get original conv2a
    conv2a_original = get_conv2a_activations(controller, obs_tensor)
    controller.conv2a_original = conv2a_original  # Store for loss computation

    # Initialize optimization
    conv2a_modified = nn.Parameter(conv2a_original.clone().requires_grad_(True))
    optimizer = torch.optim.Adam([conv2a_modified], lr=lr)

    best_prob = 0
    best_conv2a = None

    for step in range(n_steps):
        loss, metrics = compute_conv2a_fc3_loss(
            conv2a_modified, controller, obs_tensor, target=target_entity, threshold=threshold
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([conv2a_modified], max_norm=1.0)
        optimizer.step()

        target_prob = metrics['target_prob']

        if target_prob > best_prob:
            best_prob = target_prob
            best_conv2a = conv2a_modified.detach().clone()

        # Log every 10 iterations
        if step % 20 == 0:
            other_probs_str = ", ".join([f"{k}:{v:.1%}" for k, v in sorted(metrics['other_probs'].items(), key=lambda x: -x[1])[:3]])
            print(f"    Opt iter {step:3d}: {target_entity}={target_prob:.1%}, top_others=[{other_probs_str}], loss={loss.item():.3f}")

        # Early stopping if threshold reached
        if target_prob >= threshold:
            print(f"    Early stop at iter {step}: {target_entity}={target_prob:.1%}")
            break

    return best_conv2a if best_conv2a is not None else conv2a_modified.detach(), best_prob


def optimize_conv2a_multilayer(controller, obs_tensor, target_entity='green_key',
                               n_steps=1000, lr=0.5, threshold=0.8, fc3_weight=3.0,
                               use_suppression=False):
    """
    Optimize conv2a using all layer probes with FC3 weighting.

    Args:
        controller: AblationController
        obs_tensor: Input observation
        target_entity: Target entity to optimize for
        n_steps: Number of optimization steps (default: 1000)
        lr: Learning rate
        threshold: Early stopping threshold for FC3 (default: 0.8)
        fc3_weight: Weight for FC3 vs other layers
        use_suppression: If True, use competitive loss; if False, use independent loss

    Returns:
        optimized_conv2a, best_fc3_prob
    """

    # Get original conv2a
    conv2a_original = get_conv2a_activations(controller, obs_tensor)
    controller.conv2a_original = conv2a_original

    # Initialize optimization
    conv2a_modified = nn.Parameter(conv2a_original.clone().requires_grad_(True))
    optimizer = torch.optim.Adam([conv2a_modified], lr=lr)

    best_fc3_prob = 0
    best_conv2a = None

    # Choose loss function
    loss_fn = (compute_conv2a_multilayer_loss_with_suppression if use_suppression
               else compute_conv2a_multilayer_loss)

    for step in range(n_steps):
        loss, metrics = loss_fn(
            conv2a_modified, controller, obs_tensor,
            target=target_entity, fc3_weight=fc3_weight
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([conv2a_modified], max_norm=1.0)
        optimizer.step()

        fc3_prob = metrics['fc3_prob']

        if fc3_prob > best_fc3_prob:
            best_fc3_prob = fc3_prob
            best_conv2a = conv2a_modified.detach().clone()

        # Log every 10 iterations
        if step % 20 == 0:
            mode = "SUPPRESS" if use_suppression else "NO-SUPP"
            print(f"    [{mode}] Opt iter {step:3d}: FC3={fc3_prob:.1%}, FC2={metrics['fc2_prob']:.1%}, "
                  f"FC1={metrics['fc1_prob']:.1%}, loss={loss.item():.3f}")

        # Early stopping if FC3 reaches threshold
        if fc3_prob >= threshold:
            print(f"    Early stop at iter {step}: FC3={fc3_prob:.1%}")
            break

    return best_conv2a if best_conv2a is not None else conv2a_modified.detach(), best_fc3_prob