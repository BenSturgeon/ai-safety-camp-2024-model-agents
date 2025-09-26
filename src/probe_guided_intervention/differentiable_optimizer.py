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
                           layer='conv3a', n_steps=100, sparsity_weight=0.01):
        """
        Optimize mask to maximize target probe while suppressing others

        Args:
            controller: AblationController instance
            obs: Observation tensor
            target_entity: Entity to optimize for
            layer: Which layer to ablate ('conv3a', 'conv4a', 'fc1', 'fc2')
            n_steps: Number of optimization steps
            sparsity_weight: L1 penalty weight for sparsity
        """
        controller.ablation_layer = layer
        # Keep everything on CPU for compatibility with the model
        obs = obs.cpu()

        print(f"Optimizing {layer} ablations for {target_entity}...")
        print(f"Mask shape: {self.mask_logits.shape}")

        best_target_prob = 0
        best_mask = None

        for step in range(n_steps):
            # Get differentiable mask
            mask = self.get_mask(hard=True)

            # Apply mask based on layer (ensure CPU for consistency with model)
            if layer == 'conv3a':
                controller.ablation_mask = mask.cpu()
            elif layer == 'conv4a':
                controller.conv4a_ablation_mask = mask.cpu()
            elif layer == 'fc2':
                controller.fc2_ablation_mask = mask.cpu()
            else:
                raise ValueError(f"Unsupported layer: {layer}")

            # Forward pass through network
            _, conv3a, _, fc1, _, _ = controller.get_conv_layers_with_hook(obs)

            # Compute probe predictions
            fc1_flat = fc1.view(1, -1)
            target_logits = controller.fc1_probes[target_entity](fc1_flat)
            target_prob = F.softmax(target_logits, dim=-1)[0, 1]

            # Compute competing entity scores
            other_probs = []
            for entity, probe in controller.fc1_probes.items():
                if entity != target_entity:
                    logits = probe(fc1_flat)
                    prob = F.softmax(logits, dim=-1)[0, 1]
                    other_probs.append(prob)

            # Loss: maximize target, minimize others, encourage sparsity
            loss = -target_prob  # Maximize target
            loss += 0.5 * sum(other_probs) / len(other_probs) if other_probs else 0  # Minimize others
            loss += sparsity_weight * (1 - mask).mean()  # Encourage ablation (penalize zeros, not ones)

            # Backprop and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track best mask
            if target_prob > best_target_prob:
                best_target_prob = target_prob
                best_mask = mask.detach().clone()

            # Anneal temperature
            if step > 0 and step % 20 == 0:
                self.temperature = max(0.1, self.temperature * 0.8)

            # Report progress
            if step % 10 == 0:
                n_ablated = (mask < 0.5).sum().item()
                others_avg = sum(other_probs) / len(other_probs) if other_probs else 0
                print(f"Step {step}: target={target_prob:.1%}, others_avg={others_avg:.1%}, "
                      f"ablated={n_ablated}, temp={self.temperature:.3f}, loss={loss.item():.3f}")

                if target_prob > 0.9:
                    print(f"✓ Achieved >90% at step {step}!")
                    break

        # Return final binary mask
        self.training = False
        final_mask = (torch.sigmoid(self.mask_logits) > 0.5).float()

        print(f"\nOptimization complete:")
        print(f"  Best target probability: {best_target_prob:.1%}")
        print(f"  Final ablated neurons: {(final_mask < 0.5).sum().item()}/{final_mask.numel()}")

        return final_mask


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