#!/usr/bin/env python3
"""
Gradient-based neuron ablation focused on changing action output for a single frame.
Ablates neurons iteratively until the model's action changes from its original choice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Model loading
from utils.helpers import load_interpretable_model, save_observations_as_gif, save_rollout_as_gif
from utils.create_intervention_mazes import create_cross_maze


class SingleFrameGradientAblator:
    """Ablate neurons to change action output on a single observation."""

    def __init__(self, model_path='base_models/full_run/model_35001.0.pt',
                 probe_path='src/probe_training/trained_probes_20250904_080738/conv3a_probe.pt'):
        """Initialize with model and probe."""
        # Load model
        self.model = load_interpretable_model(model_path=model_path)
        self.model.eval()

        # Load probe
        probe_data = torch.load(probe_path, map_location='cpu')
        self.probe_state = probe_data['probe_state_dict']
        self.label_map = probe_data['label_map']
        self.reverse_map = {v: k for k, v in self.label_map.items()}

        # Build probe
        self.probe = self._build_probe()
        self.probe.eval()

        # Ablation tracking
        self.ablation_mask = None
        self.conv3a_activations = None
        self.total_neurons = 0
        self.ablated_count = 0

    def _build_probe(self):
        """Build probe from saved weights."""
        # Get dimensions from saved weights
        first_weight = self.probe_state['probe.0.weight']
        input_dim = first_weight.shape[1]
        hidden1_dim = first_weight.shape[0]

        second_weight = self.probe_state['probe.3.weight']
        hidden2_dim = second_weight.shape[0]

        final_weight = self.probe_state['probe.6.weight']
        num_classes = final_weight.shape[0]

        # Build probe network
        probe = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden2_dim, num_classes)
        )

        # Load saved weights
        probe[0].weight.data = self.probe_state['probe.0.weight']
        probe[0].bias.data = self.probe_state['probe.0.bias']
        probe[3].weight.data = self.probe_state['probe.3.weight']
        probe[3].bias.data = self.probe_state['probe.3.bias']
        probe[6].weight.data = self.probe_state['probe.6.weight']
        probe[6].bias.data = self.probe_state['probe.6.bias']

        # Set to eval mode to disable dropout
        for module in probe.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

        return probe

    def get_conv3a_with_hook(self, obs, requires_grad=False):
        """Get conv3a activations and apply ablation mask."""
        activations = {}

        def hook(name):
            def fn(module, input, output):
                # Apply ablation mask if it exists
                if self.ablation_mask is not None:
                    output = output * self.ablation_mask.unsqueeze(0)
                # Clone and detach to make it a leaf tensor if we need gradients
                if requires_grad:
                    output = output.clone().detach().requires_grad_(True)
                activations[name] = output
                return output
            return fn

        # Register hook
        handles = []
        for name, module in self.model.named_modules():
            if 'conv3a' in name and isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(hook('conv3a'))
                handles.append(handle)
                break

        # Forward pass
        output = self.model(obs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return output, activations.get('conv3a')

    def compute_action_change_gradients(self, obs, original_action, target_entity):
        """
        Compute gradients to change the action output.

        Args:
            obs: Single observation tensor [1, 3, 64, 64]
            original_action: The original action the model would take
            target_entity: Entity we want to redirect toward

        Returns:
            gradient_magnitude: Per-neuron gradient magnitudes
        """
        obs.requires_grad = False

        # Get conv3a with gradients enabled
        output, conv3a = self.get_conv3a_with_hook(obs, requires_grad=True)

        if conv3a is None:
            raise RuntimeError("Failed to get conv3a activations")

        # Get action logits
        if isinstance(output, tuple):
            dist = output[0]
            logits = dist.logits
        else:
            logits = output.logits

        # Create loss to minimize original action and maximize others
        action_probs = F.softmax(logits, dim=-1)

        # Loss = maximize entropy + minimize original action probability
        # This encourages the model to become uncertain and change its action
        original_action_prob = action_probs[0, original_action]
        entropy = -(action_probs * (action_probs + 1e-10).log()).sum()

        # Strong loss to change action
        loss = 3.0 * original_action_prob - 0.5 * entropy

        # Also add probe-based guidance if we have a target entity
        if target_entity:
            # Get probe predictions from conv3a
            flat_conv3a = conv3a.view(1, -1)
            probe_outputs = self.probe(flat_conv3a)
            probe_probs = F.softmax(probe_outputs, dim=-1)

            target_idx = self.reverse_map.get(target_entity)
            if target_idx is not None:
                target_prob = probe_probs[0, target_idx]
                # Add probe guidance to loss
                loss = loss - 0.5 * target_prob

        # Compute gradients
        loss.backward()

        # Get gradient magnitude per neuron
        grad_magnitude = conv3a.grad.abs()[0]  # Remove batch dimension

        return grad_magnitude

    def iterative_ablation_with_rollout(self, target_entity='green_key',
                                       max_steps=100, ablation_every_n_steps=5,
                                       percentile_per_iter=99.5, save_gif=True,
                                       ablation_mode='recalculate'):
        """
        Apply ablations progressively during a rollout to see behavior change.

        Args:
            target_entity: Entity to redirect toward
            max_steps: Maximum steps in the rollout
            ablation_every_n_steps: Apply new ablations every N steps
            percentile_per_iter: Percentile threshold for ablation each iteration
            save_gif: Whether to save rollout as GIF
            ablation_mode: 'recalculate' to find new ablations each time,
                          'cumulative' to add to existing ablations

        Returns:
            Results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Ablation During Rollout Experiment")
        print(f"Target: Redirect to {target_entity}")
        print('='*60)

        # Create cross maze
        _, venv = create_cross_maze(include_locks=False)
        obs = venv.reset()

        # Store observations for GIF
        observations_for_gif = []

        # Convert first observation
        if isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = np.asarray(obs)

        if len(obs_array.shape) == 4:
            obs_array = obs_array[0]

        # Initialize ablation mask with first observation
        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        # Get initial conv3a to set up mask dimensions
        with torch.no_grad():
            _, conv3a = self.get_conv3a_with_hook(obs_tensor)
            if conv3a is not None:
                self.ablation_mask = torch.ones_like(conv3a[0])
                self.total_neurons = self.ablation_mask.numel()

        results = {
            'target_entity': target_entity,
            'steps': [],
            'total_reward': 0
        }

        # Run rollout with progressive ablations
        for step in range(max_steps):
            # Store observation for GIF
            if save_gif:
                observations_for_gif.append(obs_array.copy())

            # Convert observation
            obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)

            # Apply ablations based on mode
            if step > 0 and step % ablation_every_n_steps == 0:
                if ablation_mode == 'recalculate':
                    print(f"\nStep {step}: Recalculating ablations for current observation...")

                    # Reset ablation mask to recalculate from scratch
                    self.ablation_mask = torch.ones(self.ablation_mask.shape) if hasattr(self, 'ablation_mask') else None

                    # Iteratively find neurons to ablate for this specific observation
                    for ablation_iter in range(10):  # Do multiple iterations to find effective ablations
                        # Get current action with current ablations
                        with torch.no_grad():
                            output, _ = self.get_conv3a_with_hook(obs_tensor)
                            if isinstance(output, tuple):
                                logits = output[0].logits
                            else:
                                logits = output.logits
                            current_action = F.softmax(logits, dim=-1).argmax().item()

                        # Get gradients for this observation
                        grad_magnitude = self.compute_action_change_gradients(
                            obs_tensor, current_action, target_entity
                        )

                        # Select neurons to ablate based on gradient magnitude
                        threshold = torch.quantile(grad_magnitude, percentile_per_iter / 100.0)
                        neurons_to_ablate = (grad_magnitude > threshold) & (self.ablation_mask > 0)

                        if neurons_to_ablate.any():
                            # Apply ablations
                            self.ablation_mask[neurons_to_ablate] = 0.0
                            new_ablations = neurons_to_ablate.sum().item()
                        else:
                            break  # No more neurons to ablate

                    self.ablated_count = (self.ablation_mask == 0).sum().item()
                    print(f"  Ablations for this frame: {self.ablated_count}/{self.total_neurons} ({self.ablated_count/self.total_neurons*100:.1f}%)")

                else:  # cumulative mode
                    print(f"\nStep {step}: Adding more ablations...")

                    # Get current action
                    with torch.no_grad():
                        output, _ = self.get_conv3a_with_hook(obs_tensor)
                        if isinstance(output, tuple):
                            logits = output[0].logits
                        else:
                            logits = output.logits
                        current_action = F.softmax(logits, dim=-1).argmax().item()

                    # Get gradients for this observation
                    grad_magnitude = self.compute_action_change_gradients(
                        obs_tensor, current_action, target_entity
                    )

                    # Select new neurons to ablate
                    threshold = torch.quantile(grad_magnitude, percentile_per_iter / 100.0)
                    neurons_to_ablate = (grad_magnitude > threshold) & (self.ablation_mask > 0)

                    if neurons_to_ablate.any():
                        # Apply ablations
                        self.ablation_mask[neurons_to_ablate] = 0.0
                        new_ablations = neurons_to_ablate.sum().item()
                        self.ablated_count = (self.ablation_mask == 0).sum().item()

                        print(f"  Added {new_ablations} new ablations")
                        print(f"  Total ablated: {self.ablated_count}/{self.total_neurons} ({self.ablated_count/self.total_neurons*100:.1f}%)")

            # Get action with current ablations
            with torch.no_grad():
                output, conv3a = self.get_conv3a_with_hook(obs_tensor)

                if isinstance(output, tuple):
                    logits = output[0].logits
                else:
                    logits = output.logits

                action_probs = F.softmax(logits, dim=-1)
                action = action_probs.argmax().item()

                # Get probe prediction
                if conv3a is not None:
                    flat_conv3a = conv3a.view(1, -1)
                    probe_out = self.probe(flat_conv3a)
                    probe_probs = F.softmax(probe_out, dim=-1)
                    pred_idx = probe_probs.argmax().item()
                    predicted_entity = self.label_map.get(pred_idx, 'unknown')
                else:
                    predicted_entity = 'unknown'

            # Log step info
            step_info = {
                'step': step,
                'action': action,
                'predicted_entity': predicted_entity,
                'ablated_neurons': self.ablated_count if hasattr(self, 'ablated_count') else 0
            }
            results['steps'].append(step_info)

            # Take action in environment
            obs, reward, done, info = venv.step(np.array([action]))
            results['total_reward'] += reward[0] if isinstance(reward, np.ndarray) else reward

            # Update observation
            if isinstance(obs, np.ndarray):
                obs_array = obs
            else:
                obs_array = np.asarray(obs)

            if len(obs_array.shape) == 4:
                obs_array = obs_array[0]

            # Check if episode ended
            is_done = done[0] if isinstance(done, np.ndarray) else done
            if is_done:
                print(f"\nEpisode ended at step {step}")
                break

        # Save final observation
        if save_gif:
            observations_for_gif.append(obs_array.copy())

        # Save GIF
        if save_gif and len(observations_for_gif) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = f'src/probe_guided_intervention/results/ablation_rollout_{target_entity}_{timestamp}.gif'

            os.makedirs('src/probe_guided_intervention/results', exist_ok=True)

            success = save_observations_as_gif(
                observations_for_gif,
                filepath=gif_path,
                fps=10,  # Faster FPS for rollout
                enhance_size=True
            )

            if success:
                print(f"\n📹 Saved ablation rollout GIF to: {gif_path}")
                results['gif_path'] = gif_path

        venv.close()

        print(f"\nFinal Stats:")
        print(f"  Total steps: {len(results['steps'])}")
        print(f"  Total reward: {results['total_reward']}")
        print(f"  Final ablations: {self.ablated_count}/{self.total_neurons} ({self.ablated_count/self.total_neurons*100:.1f}%)")

        return results

    def iterative_ablation_single_frame(self, target_entity='green_key',
                                       max_iterations=100, percentile_per_iter=99.5,
                                       save_gif=True):
        """
        Ablate neurons iteratively on a single frame until action changes.

        Args:
            target_entity: Entity to redirect toward
            max_iterations: Maximum ablation iterations
            percentile_per_iter: Percentile threshold for ablation each iteration
            save_gif: Whether to save observations as GIF

        Returns:
            Results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Single Frame Ablation Experiment")
        print(f"Target: Redirect to {target_entity}")
        print('='*60)

        # Create cross maze and get first observation
        _, venv = create_cross_maze(include_locks=False)
        initial_obs = venv.reset()

        # Convert observation to numpy array
        if isinstance(initial_obs, np.ndarray):
            obs_array = initial_obs
        else:
            obs_array = np.asarray(initial_obs)

        # Handle batched observations - take first one if batched
        if len(obs_array.shape) == 4:  # Already batched [N, H, W, C]
            obs_array = obs_array[0]  # Take first observation

        # Store observations for GIF
        observations_for_gif = []
        if save_gif:
            # Store the initial observation
            observations_for_gif.append(obs_array.copy())

        # Now convert to tensor and add batch dimension
        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)  # [1, H, W, C]
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # [1, C, H, W]

        # Get original action and probe prediction
        with torch.no_grad():
            output, conv3a = self.get_conv3a_with_hook(obs_tensor)

            if isinstance(output, tuple):
                # It's a tuple (dist, value)
                dist = output[0]
                logits = dist.logits
                action_probs = F.softmax(logits, dim=-1)
            else:
                logits = output.logits
                action_probs = F.softmax(logits, dim=-1)
            original_action = action_probs.argmax().item()
            original_probs = action_probs[0].cpu().numpy()

            # Get probe prediction
            if conv3a is not None:
                flat_conv3a = conv3a.view(1, -1)
                probe_out = self.probe(flat_conv3a)
                probe_probs = F.softmax(probe_out, dim=-1)
                pred_idx = probe_probs.argmax().item()
                original_entity = self.label_map.get(pred_idx, 'unknown')
            else:
                original_entity = 'unknown'

        print(f"\nBaseline:")
        print(f"  Original action: {original_action}")
        print(f"  Original entity prediction: {original_entity}")
        print(f"  Action probs: {original_probs[:6].round(3)}...")  # Show first 6 actions

        # Initialize ablation mask
        if conv3a is not None:
            self.ablation_mask = torch.ones_like(conv3a[0])
            self.total_neurons = self.ablation_mask.numel()

        results = {
            'target_entity': target_entity,
            'original_action': original_action,
            'original_entity': original_entity,
            'iterations': []
        }

        # Iterative ablation
        for iteration in range(max_iterations):
            # Compute gradients for changing action
            grad_magnitude = self.compute_action_change_gradients(
                obs_tensor, original_action, target_entity
            )

            # Select neurons to ablate
            threshold = torch.quantile(grad_magnitude, percentile_per_iter / 100.0)
            neurons_to_ablate = (grad_magnitude > threshold) & (self.ablation_mask > 0)

            if not neurons_to_ablate.any():
                print(f"\nIteration {iteration + 1}: No new neurons to ablate")
                break

            # Apply ablations
            self.ablation_mask[neurons_to_ablate] = 0.0
            new_ablations = neurons_to_ablate.sum().item()
            self.ablated_count = (self.ablation_mask == 0).sum().item()

            # Test new behavior
            with torch.no_grad():
                output, conv3a = self.get_conv3a_with_hook(obs_tensor)

                if isinstance(output, tuple):
                    dist = output[0]
                    logits = dist.logits
                else:
                    logits = output.logits

                action_probs = F.softmax(logits, dim=-1)
                new_action = action_probs.argmax().item()
                new_probs = action_probs[0].cpu().numpy()

                # Get probe prediction
                if conv3a is not None:
                    flat_conv3a = conv3a.view(1, -1)
                    probe_out = self.probe(flat_conv3a)
                    probe_probs = F.softmax(probe_out, dim=-1)
                    pred_idx = probe_probs.argmax().item()
                    new_entity = self.label_map.get(pred_idx, 'unknown')
                else:
                    new_entity = 'unknown'

            # Log iteration
            iteration_result = {
                'iteration': iteration + 1,
                'new_ablations': new_ablations,
                'total_ablated': self.ablated_count,
                'ablation_percentage': self.ablated_count / self.total_neurons * 100,
                'new_action': new_action,
                'new_entity': new_entity,
                'action_changed': new_action != original_action
            }
            results['iterations'].append(iteration_result)

            # Store observation for GIF every iteration (for smoother visualization)
            if save_gif:
                observations_for_gif.append(obs_array.copy())

            # Print progress every 5 iterations or when action changes
            if (iteration + 1) % 5 == 0 or new_action != original_action:
                print(f"\nIteration {iteration + 1}:")
                print(f"  Ablated: {new_ablations} new, {self.ablated_count} total ({iteration_result['ablation_percentage']:.1f}%)")
                print(f"  Action: {original_action} → {new_action} {'✓ CHANGED!' if new_action != original_action else ''}")
                print(f"  Entity: {original_entity} → {new_entity}")
                print(f"  New action probs: {new_probs[:6].round(3)}...")

            # Check if we've achieved our goal
            if new_action != original_action:
                print(f"\n✓ SUCCESS! Action changed from {original_action} to {new_action}")
                print(f"  Total neurons ablated: {self.ablated_count}/{self.total_neurons} ({iteration_result['ablation_percentage']:.1f}%)")
                results['success'] = True
                results['final_action'] = new_action
                results['final_entity'] = new_entity
                break
        else:
            print(f"\n✗ Max iterations reached without changing action")
            results['success'] = False

        # Save observations as GIF
        if save_gif and len(observations_for_gif) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = f'src/probe_guided_intervention/results/ablation_progress_{target_entity}_{timestamp}.gif'

            # Create directory if it doesn't exist
            os.makedirs('src/probe_guided_intervention/results', exist_ok=True)

            # Save the GIF
            success = save_observations_as_gif(
                observations_for_gif,
                filepath=gif_path,
                fps=2,  # Slow FPS to see changes
                enhance_size=True  # Make frames larger
            )

            if success:
                print(f"\n📹 Saved ablation progress GIF to: {gif_path}")
                results['gif_path'] = gif_path
            else:
                print("\n⚠️ Failed to save GIF")

        venv.close()
        return results

    def visualize_ablation_pattern(self, save_path=None):
        """Visualize which neurons were ablated."""
        if self.ablation_mask is None:
            print("No ablations to visualize")
            return

        # Get ablation statistics per channel
        channels = self.ablation_mask.shape[0]
        height = self.ablation_mask.shape[1]
        width = self.ablation_mask.shape[2]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # 1. Per-channel ablation percentage
        ax = axes[0]
        channel_ablation_pcts = []
        for c in range(channels):
            ablated = (self.ablation_mask[c] == 0).sum().item()
            total = height * width
            channel_ablation_pcts.append(ablated / total * 100)

        bars = ax.bar(range(channels), channel_ablation_pcts)
        ax.set_xlabel('Channel')
        ax.set_ylabel('% Neurons Ablated')
        ax.set_title(f'Ablation Distribution Across Conv3a Channels (Total: {self.ablated_count}/{self.total_neurons} = {self.ablated_count/self.total_neurons*100:.1f}%)')

        # Color bars by ablation percentage
        for i, (bar, pct) in enumerate(zip(bars, channel_ablation_pcts)):
            if pct > 50:
                bar.set_color('red')
            elif pct > 25:
                bar.set_color('orange')
            elif pct > 10:
                bar.set_color('yellow')
            else:
                bar.set_color('green')

        # 2. Spatial pattern of top ablated channels
        ax = axes[1]
        top_channels_idx = np.argsort(channel_ablation_pcts)[-8:]  # Top 8 channels

        # Create combined visualization
        combined = np.zeros((height, width * 8))
        for i, c in enumerate(top_channels_idx):
            combined[:, i*width:(i+1)*width] = self.ablation_mask[c].cpu().numpy()

        im = ax.imshow(combined, cmap='RdBu', vmin=0, vmax=1, aspect='auto')
        ax.set_title('Spatial Ablation Pattern (Top 8 Most Ablated Channels)')
        ax.set_xticks([width * (i + 0.5) for i in range(8)])
        ax.set_xticklabels([f'Ch{c}\n({channel_ablation_pcts[c]:.0f}%)' for c in top_channels_idx])
        ax.set_ylabel('Height')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='1=Active, 0=Ablated')

        plt.tight_layout()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'single_frame_ablation_{timestamp}.png'

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved ablation visualization to {save_path}")
        plt.show()


def test_robust_ablations(target_entity='green_key', num_mazes=10):
    """Test ablations that work across multiple observations."""
    ablator = SingleFrameGradientAblator()

    print(f"Creating {num_mazes} test mazes...")

    # Create test batch
    test_envs = []
    test_observations = []
    original_actions = []

    # Use viable seeds
    viable_seeds_path = 'utils/viable_seeds.txt'
    try:
        with open(viable_seeds_path, 'r') as f:
            viable_seeds = [int(line.strip()) for line in f if line.strip()]
    except:
        viable_seeds = list(range(1000, 2000))

    for i in range(num_mazes):
        _, venv = create_cross_maze(include_locks=False)
        obs = venv.reset()

        # Process observation
        if isinstance(obs, dict):
            obs_array = obs['rgb'][0]  # Extract rgb and remove batch dim
        elif isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = np.asarray(obs)

        if len(obs_array.shape) == 4:
            obs_array = obs_array[0]

        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        # Get original action
        with torch.no_grad():
            output, _ = ablator.get_conv3a_with_hook(obs_tensor)
            if isinstance(output, tuple):
                dist = output[0]
                logits = dist.logits
            else:
                logits = output.logits
            action = F.softmax(logits, dim=-1).argmax().item()

        test_envs.append(venv)
        test_observations.append(obs_tensor)
        original_actions.append(action)

        print(f"  Maze {i}: Initial action = {action}")

    print(f"\nOriginal action distribution:")
    action_counts = {}
    for action in original_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    for action, count in sorted(action_counts.items()):
        print(f"  Action {action}: {count} mazes")

    # Compute robust ablations
    print(f"\nComputing robust ablations targeting {target_entity}...")
    ablator.ablation_mask = None

    for iteration in range(50):
        # Accumulate gradients across all observations
        total_gradient = None
        num_unchanged = 0
        changed_status = []

        for i, (obs, orig_action) in enumerate(zip(test_observations, original_actions)):
            # Check current action with ablation
            with torch.no_grad():
                output, _ = ablator.get_conv3a_with_hook(obs, requires_grad=False)
                if isinstance(output, tuple):
                    dist = output[0]
                    logits = dist.logits
                else:
                    logits = output.logits
                current_action = F.softmax(logits, dim=-1).argmax().item()

            # If action hasn't changed, compute gradient
            if current_action == orig_action:
                num_unchanged += 1
                grad = ablator.compute_action_change_gradients(obs, orig_action, target_entity)

                if total_gradient is None:
                    total_gradient = grad
                else:
                    total_gradient += grad
                changed_status.append(False)
            else:
                changed_status.append(True)

        # Print progress
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}: {num_unchanged}/{len(test_observations)} still unchanged")
            print(f"    Status: {['✓' if c else '✗' for c in changed_status]}")

        # Check if all actions changed
        if num_unchanged == 0:
            print(f"\n✓ SUCCESS! All actions changed after {iteration} iterations")
            total_ablated = (ablator.ablation_mask == 0).sum().item()
            total_neurons = ablator.ablation_mask.numel()
            print(f"  Total neurons ablated: {total_ablated}/{total_neurons} ({total_ablated/total_neurons*100:.1f}%)")
            break

        # If no gradients, break
        if total_gradient is None:
            print("Warning: No gradients computed")
            break

        # Average gradient across observations
        avg_gradient = total_gradient / num_unchanged

        # Initialize mask if needed
        if ablator.ablation_mask is None:
            ablator.ablation_mask = torch.ones_like(avg_gradient)

        # Select neurons to ablate
        threshold = torch.quantile(avg_gradient, 0.995)
        neurons_to_ablate = (avg_gradient > threshold) & (ablator.ablation_mask > 0)

        if not neurons_to_ablate.any():
            # Lower threshold if stuck
            threshold = torch.quantile(avg_gradient, 0.99)
            neurons_to_ablate = (avg_gradient > threshold) & (ablator.ablation_mask > 0)
            if not neurons_to_ablate.any():
                break

        # Apply ablations
        ablator.ablation_mask[neurons_to_ablate] = 0.0

        # Check if ablating too much
        ablated_pct = (ablator.ablation_mask == 0).sum().item() / ablator.ablation_mask.numel()
        if ablated_pct > 0.3:
            print(f"  Stopping: {ablated_pct*100:.1f}% neurons ablated")
            break

    # Clean up
    for venv in test_envs:
        venv.close()

    return ablator


def main():
    """Run gradient ablation experiments."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "robust":
        # Test robust ablations across multiple observations
        print("="*60)
        print("ROBUST MULTI-OBSERVATION ABLATION TEST")
        print("="*60)
        ablator = test_robust_ablations('green_key', num_mazes=10)
    else:
        # Test single observation (original behavior)
        print("="*60)
        print("SINGLE-FRAME GRADIENT ABLATION EXPERIMENT")
        print("="*60)

        # Initialize ablator
        ablator = SingleFrameGradientAblator()

        # Test green_key
        target = 'green_key'
        ablator.ablation_mask = None
        ablator.ablated_count = 0

        # Run experiment
        results = ablator.iterative_ablation_single_frame(
            target_entity=target,
            max_iterations=50,
            percentile_per_iter=99.0  # Ablate top 1% each iteration
        )

        # Visualize if successful
        if results.get('success'):
            ablator.visualize_ablation_pattern()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'single_frame_ablation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_file}")

        if results.get('success'):
            final_iter = len(results['iterations'])
            final_ablated_pct = results['iterations'][-1]['ablation_percentage']
            print(f"SUCCESS: Action changed in {final_iter} iterations ({final_ablated_pct:.1f}% neurons ablated)")


if __name__ == "__main__":
    main()