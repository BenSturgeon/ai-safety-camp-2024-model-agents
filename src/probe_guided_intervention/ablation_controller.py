"""
Main ablation controller that coordinates probe-guided interventions.
Uses modular components for probe loading, optimization, and experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime

from src.utils.helpers import load_interpretable_model
from src.utils.create_intervention_mazes import create_cross_maze

# Import our modular components
from .probe_loaders import (
    load_conv3a_probe,
    load_conv4a_probe,
    load_fc1_binary_probes
)
from .ablation_optimizers import (
    find_optimal_ablations_random,
    find_optimal_ablations_for_conv4a,
    find_optimal_ablations_for_fc1
)
from .rollout_experiments import (
    test_green_key_first,
    test_conv4a_optimization_rollout,
    test_fc1_optimization_rollout
)


class AblationController:
    """Main controller for probe-guided neural network ablations."""

    def __init__(self,
                 model_path='base_models/full_run/model_35001.0.pt',
                 probe_path='src/probe_training/trained_probes_20250904_080738/conv3a_probe.pt',
                 conv4a_probe_path='src/probe_training/trained_probes_20250904_080738/conv4a_probe.pt',
                 fc1_probe_dir='src/probe_training/binary_probes_20250925_183257/'):
        """
        Initialize the ablation controller with model and probes.

        Args:
            model_path: Path to the trained model
            probe_path: Path to conv3a probe
            conv4a_probe_path: Path to conv4a probe
            fc1_probe_dir: Directory containing fc1 binary probes
        """
        # Load model
        self.model = load_interpretable_model(model_path=model_path)
        self.model.eval()

        # Load conv3a probe
        self.probe, self.label_map, self.reverse_map = load_conv3a_probe(probe_path)

        # Load conv4a probe
        self.conv4a_probe = load_conv4a_probe(conv4a_probe_path)

        # Load fc1 binary probes
        self.fc1_probes = load_fc1_binary_probes(fc1_probe_dir)

        # Ablation tracking
        self.ablation_mask = None
        self.total_neurons = 0
        self.ablated_count = 0

        print(f"AblationController initialized with:")
        print(f"  - Model: {model_path}")
        print(f"  - Conv3a probe: {probe_path}")
        print(f"  - Conv4a probe: {conv4a_probe_path}")
        print(f"  - FC1 probes: {len(self.fc1_probes)} entities loaded")

    def get_conv_layers_with_hook(self, obs, requires_grad=False):
        """
        Get conv3a, conv4a, and fc1 activations and apply ablation mask to conv3a.

        Args:
            obs: Observation tensor
            requires_grad: Whether to compute gradients

        Returns:
            tuple: (model_output, conv3a, conv4a, fc1)
        """
        activations = {}

        def hook(name):
            def fn(module, input, output):
                # Apply ablation mask only to conv3a
                if name == 'conv3a' and self.ablation_mask is not None:
                    output = output * self.ablation_mask.unsqueeze(0)
                # Clone and detach to make it a leaf tensor if we need gradients
                if requires_grad and name == 'conv3a':
                    output = output.clone().detach().requires_grad_(True)
                activations[name] = output
                return output if name == 'conv3a' else None  # Only modify conv3a
            return fn

        # Register hooks for conv3a, conv4a, and fc1
        handles = []
        for name, module in self.model.named_modules():
            if 'conv3a' in name and isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(hook('conv3a'))
                handles.append(handle)
            elif 'conv4a' in name and isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(hook('conv4a'))
                handles.append(handle)
            elif name == 'fc1' and isinstance(module, nn.Linear):
                handle = module.register_forward_hook(hook('fc1'))
                handles.append(handle)

        # Forward pass
        output = self.model(obs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return output, activations.get('conv3a'), activations.get('conv4a'), activations.get('fc1')

    def get_conv3a_with_hook(self, obs, requires_grad=False):
        """Backward compatibility wrapper for conv3a only."""
        output, conv3a, _, _ = self.get_conv_layers_with_hook(obs, requires_grad)
        return output, conv3a

    def find_optimal_ablations_random(self, obs, target_entity, max_trials=30):
        """Find optimal ablations using random search."""
        # Wrapper that provides the necessary context
        def get_conv3a_hook_fn(obs_tensor):
            return self.get_conv3a_with_hook(obs_tensor)

        return find_optimal_ablations_random(
            self.model, self.probe, self.ablation_mask,
            obs, target_entity, get_conv3a_hook_fn,
            self.label_map, self.reverse_map, max_trials
        )

    def find_optimal_ablations_for_conv4a(self, obs, target_entity, max_trials=50):
        """Find conv3a ablations that affect conv4a predictions."""
        # Wrapper that provides the necessary context
        def get_conv_layers_hook_fn(obs_tensor):
            return self.get_conv_layers_with_hook(obs_tensor)

        return find_optimal_ablations_for_conv4a(
            self.ablation_mask, obs, target_entity,
            get_conv_layers_hook_fn, self.probe, self.conv4a_probe,
            self.label_map, self.reverse_map, max_trials
        )

    def find_optimal_ablations_for_fc1(self, obs, target_entity='green_key', max_trials=50):
        """Find conv3a ablations that affect fc1 predictions."""
        # Wrapper that provides the necessary context
        def get_conv_layers_hook_fn(obs_tensor):
            return self.get_conv_layers_with_hook(obs_tensor)

        return find_optimal_ablations_for_fc1(
            self.ablation_mask, obs, self.fc1_probes,
            get_conv_layers_hook_fn, target_entity, max_trials
        )

    # Experiment methods
    def test_green_key_first(self, max_steps=100, save_gif=True):
        """Test if ablations steer agent to collect green_key first."""
        return test_green_key_first(self, max_steps, save_gif)

    def test_conv4a_optimization_rollout(self, max_steps=30, reoptimize_each_step=True):
        """Test conv4a optimization during rollout."""
        return test_conv4a_optimization_rollout(self, max_steps, reoptimize_each_step)

    def test_fc1_optimization_rollout(self, max_steps=30, reoptimize_each_step=True):
        """Test fc1 optimization during rollout."""
        return test_fc1_optimization_rollout(self, max_steps, reoptimize_each_step)

    def run_single_optimization_test(self, layer='fc1', target_entity='green_key'):
        """
        Run a single test of optimization on a specific layer.

        Args:
            layer: Which layer to optimize ('conv3a', 'conv4a', or 'fc1')
            target_entity: Which entity to optimize for

        Returns:
            dict: Results of the optimization
        """
        print(f"\n{'='*60}")
        print(f"SINGLE OPTIMIZATION TEST: {layer.upper()} -> {target_entity}")
        print('='*60)

        # Create test maze
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

        # Initialize ablation mask
        with torch.no_grad():
            _, conv3a, conv4a, fc1 = self.get_conv_layers_with_hook(obs_tensor)
            if conv3a is not None:
                self.ablation_mask = torch.ones_like(conv3a[0])
                self.total_neurons = self.ablation_mask.numel()

        # Run optimization based on layer
        if layer == 'conv3a':
            print("Optimizing conv3a for direct probe predictions...")
            self.ablation_mask = self.find_optimal_ablations_random(obs_tensor, target_entity)

            # Check results
            with torch.no_grad():
                _, conv3a, _, _ = self.get_conv_layers_with_hook(obs_tensor)
                flat = conv3a.view(1, -1)
                out = self.probe(flat)
                probs = F.softmax(out, dim=-1)
                pred_idx = probs.argmax().item()
                predicted = self.label_map.get(pred_idx, 'unknown')
                confidence = probs.max().item()

            result = {
                'layer': 'conv3a',
                'predicted': predicted,
                'confidence': confidence,
                'success': predicted == target_entity
            }

        elif layer == 'conv4a':
            print("Optimizing conv3a to affect conv4a predictions...")
            self.ablation_mask = self.find_optimal_ablations_for_conv4a(obs_tensor, target_entity)

            # Check results
            with torch.no_grad():
                _, _, conv4a, _ = self.get_conv_layers_with_hook(obs_tensor)
                flat = conv4a.view(1, -1)
                out = self.conv4a_probe(flat)
                probs = F.softmax(out, dim=-1)
                pred_idx = probs.argmax().item()
                predicted = self.label_map.get(pred_idx, 'unknown')
                confidence = probs.max().item()

            result = {
                'layer': 'conv4a',
                'predicted': predicted,
                'confidence': confidence,
                'success': predicted == target_entity
            }

        elif layer == 'fc1':
            print("Optimizing conv3a to affect fc1 predictions...")
            self.ablation_mask = self.find_optimal_ablations_for_fc1(obs_tensor, target_entity)

            # Check results
            with torch.no_grad():
                _, _, _, fc1 = self.get_conv_layers_with_hook(obs_tensor)
                fc1_flat = fc1.view(1, -1)

                confidences = {}
                for entity, probe in self.fc1_probes.items():
                    out = probe(fc1_flat)
                    probs = F.softmax(out, dim=-1)
                    conf = probs[0, 1].item()
                    confidences[entity] = conf

            result = {
                'layer': 'fc1',
                'confidences': confidences,
                'target_confidence': confidences.get(target_entity, 0.0),
                'success': confidences.get(target_entity, 0.0) > 0.8
            }

        else:
            raise ValueError(f"Unknown layer: {layer}")

        # Add ablation stats
        result['ablated_count'] = (self.ablation_mask == 0).sum().item()
        result['ablation_percentage'] = result['ablated_count'] / self.total_neurons * 100

        # Clean up
        venv.close()

        # Print results
        print(f"\nResults:")
        print(f"  Layer: {result['layer']}")
        if layer == 'fc1':
            print(f"  Target confidence: {result['target_confidence']:.1%}")
            print(f"  All confidences:")
            for entity, conf in result['confidences'].items():
                print(f"    {entity}: {conf:.1%}")
        else:
            print(f"  Predicted: {result['predicted']}")
            print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Ablated: {result['ablated_count']}/{self.total_neurons} ({result['ablation_percentage']:.1f}%)")
        print(f"  Success: {'✓' if result['success'] else '✗'}")

        return result


def main():
    """Example usage of the AblationController."""
    import sys

    # Create controller
    controller = AblationController()

    if len(sys.argv) > 1:
        if sys.argv[1] == "fc1":
            # Test FC1 optimization
            print("Testing FC1 optimization...")
            result = controller.run_single_optimization_test('fc1', 'green_key')

        elif sys.argv[1] == "conv4a":
            # Test Conv4a optimization
            print("Testing Conv4a optimization...")
            result = controller.run_single_optimization_test('conv4a', 'green_key')

        elif sys.argv[1] == "rollout":
            # Test rollout with FC1 re-optimization
            print("Testing FC1 rollout with re-optimization...")
            result = controller.test_fc1_optimization_rollout(max_steps=20, reoptimize_each_step=True)
            print(f"Success: {result['success']}")

        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python ablation_controller.py [fc1|conv4a|rollout]")
    else:
        # Default: run a simple FC1 test
        print("Running default FC1 optimization test...")
        result = controller.run_single_optimization_test('fc1', 'green_key')


if __name__ == "__main__":
    main()