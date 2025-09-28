#!/usr/bin/env python
"""
Differentiable ablation optimization using Gumbel-Sigmoid and gradient descent.
Much more efficient than random search for finding optimal ablation patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('.')

class DifferentiableAblationOptimizer:
    def __init__(self, n_channels, n_height, n_width, device='cpu'):  # Use CPU for compatibility
        """
        Initialize learnable mask parameters.

        Args:
            n_channels: Number of channels in the layer to ablate
            n_height: Height of feature map
            n_width: Width of feature map
            device: Device to run on
        """
        self.device = device
        # Learn logits for each spatial position - initialize to favor keeping neurons
        self.mask_logits = nn.Parameter(torch.ones(n_channels, n_height, n_width).to(device) * 2.0)  # Bias toward 1 (keep)
        self.optimizer = torch.optim.Adam([self.mask_logits], lr=0.05)  # Lower learning rate
        self.temperature = 1.0  # Start high, anneal down
        self.training = True

    def get_mask(self, hard=False):
        """Get current mask using Gumbel-sigmoid trick"""
        if self.training:
            # During training: differentiable sampling
            uniform = torch.rand_like(self.mask_logits)
            gumbel = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
            soft_mask = torch.sigmoid((self.mask_logits + gumbel) / self.temperature)

            if hard:
                # Straight-through estimator
                hard_mask = (soft_mask > 0.5).float()
                mask = hard_mask.detach() + soft_mask - soft_mask.detach()
            else:
                mask = soft_mask
        else:
            # During eval: deterministic threshold
            mask = (torch.sigmoid(self.mask_logits) > 0.5).float()

        return mask

    def optimize_for_probes(self, controller, obs, target_entity='green_key',
                           layer='conv3a', n_steps=100, min_threshold=0.8):
        """
        Optimize conv3a modifications to improve all downstream probes equally.

        Args:
            controller: AblationController instance
            obs: Observation tensor
            target_entity: Entity to optimize for
            layer: Which layer to modify (currently only 'conv3a' supported)
            n_steps: Maximum number of optimization steps
            min_threshold: Minimum probe accuracy to achieve (default 0.8)
        """
        from src.probe_guided_intervention.intervention_loss import compute_intervention_loss

        if layer != 'conv3a':
            raise ValueError("Currently only conv3a modification is supported")

        # Keep everything on CPU for compatibility with the model
        obs = obs.cpu()

        print(f"Optimizing {layer} modifications for {target_entity}...")
        print(f"Target threshold: {min_threshold:.0%} for all probes")

        # Get original conv3a
        with torch.no_grad():
            _, conv3a_original, _, _, _, _ = controller.get_conv_layers_with_hook(obs)

        # Initialize conv3a_modified as learnable parameter
        conv3a_modified = nn.Parameter(conv3a_original.clone().requires_grad_(True))
        optimizer = torch.optim.Adam([conv3a_modified], lr=0.05)  # Higher learning rate

        best_min_prob = 0
        best_conv3a = None
        patience = 0
        max_patience = 100  # More patience for harder optimizations

        for step in range(n_steps):
            # Compute loss
            loss, metrics = compute_intervention_loss(
                conv3a_modified, controller, obs, target=target_entity
            )

            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track best result
            min_prob = metrics['min_prob']
            if min_prob > best_min_prob:
                best_min_prob = min_prob
                best_conv3a = conv3a_modified.detach().clone()
                patience = 0
            else:
                patience += 1

            # Report progress
            if step % 10 == 0:
                print(f"Step {step}: min_prob={min_prob:.1%}, "
                      f"probs={metrics['probe_probs']}, "
                      f"loss={loss.item():.3f}")

                # Check if we've achieved threshold
                if min_prob >= min_threshold:
                    print(f"✓ All probes above {min_threshold:.0%} at step {step}!")
                    break

            # Early stopping if no improvement
            if patience >= max_patience:
                print(f"Early stopping - no improvement for {max_patience} steps")
                break

        print(f"\nOptimization complete:")
        print(f"  Best minimum probability: {best_min_prob:.1%}")
        print(f"  Final probe accuracies: {metrics['probe_probs']}")

        if best_min_prob < min_threshold:
            print(f"  ⚠ Warning: Failed to reach {min_threshold:.0%} threshold")

        return best_conv3a if best_conv3a is not None else conv3a_modified.detach()


def test_differentiable_optimization():
    """Test the differentiable optimizer on a sample observation"""
    import procgen
    from src.probe_guided_intervention.ablation_controller import AblationController
    from src.utils.create_intervention_mazes import create_cross_maze

    # Create controller and maze
    controller = AblationController()
    _, venv = create_cross_maze(include_locks=False)
    obs = venv.reset()

    # Convert observation
    if isinstance(obs, np.ndarray):
        obs_array = obs
    else:
        obs_array = np.asarray(obs)
    if len(obs_array.shape) == 4:
        obs_array = obs_array[0]

    obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)

    # Get layer dimensions
    with torch.no_grad():
        _, conv3a, conv4a, _, _, _ = controller.get_conv_layers_with_hook(obs_tensor)
        conv3a_shape = conv3a[0].shape
        conv4a_shape = conv4a[0].shape

    print(f"Conv3a shape: {conv3a_shape}")
    print(f"Conv4a shape: {conv4a_shape}")

    # Test conv3a optimization
    print("\n" + "="*60)
    print("Testing differentiable optimization on conv3a:")
    print("="*60)

    optimizer = DifferentiableAblationOptimizer(
        n_channels=conv3a_shape[0],
        n_height=conv3a_shape[1],
        n_width=conv3a_shape[2]
    )

    final_mask = optimizer.optimize_for_probes(
        controller,
        obs_tensor,
        target_entity='green_key',
        layer='conv3a',
        n_steps=50,
        sparsity_weight=0.001  # Much smaller penalty to avoid over-ablation
    )

    # Apply final mask and check results
    controller.ablation_mask = final_mask.cpu()
    with torch.no_grad():
        _, _, _, fc1, _, _ = controller.get_conv_layers_with_hook(obs_tensor)
        fc1_flat = fc1.view(1, -1)

        print("\nFinal FC1 predictions with optimized mask:")
        for entity, probe in controller.fc1_probes.items():
            logits = probe(fc1_flat)
            prob = F.softmax(logits, dim=-1)[0, 1].item()
            print(f"  {entity}: {prob:.1%}")

    # Test conv4a optimization
    print("\n" + "="*60)
    print("Testing differentiable optimization on conv4a:")
    print("="*60)

    controller.ablation_layer = 'conv4a'
    optimizer_conv4a = DifferentiableAblationOptimizer(
        n_channels=conv4a_shape[0],
        n_height=conv4a_shape[1],
        n_width=conv4a_shape[2]
    )

    final_mask_conv4a = optimizer_conv4a.optimize_for_probes(
        controller,
        obs_tensor,
        target_entity='green_key',
        layer='conv4a',
        n_steps=100,
        sparsity_weight=0.01
    )

    venv.close()
    print("\nTest complete!")

    return final_mask, final_mask_conv4a


if __name__ == "__main__":
    test_differentiable_optimization()