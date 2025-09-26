"""
Probe loading and building utilities for different probe architectures.
"""

import torch
import torch.nn as nn
import os


def build_multiclass_probe(probe_state):
    """Build multi-class probe from saved weights."""
    # Get dimensions from saved weights
    first_weight = probe_state['probe.0.weight']
    input_dim = first_weight.shape[1]
    hidden1_dim = first_weight.shape[0]

    second_weight = probe_state['probe.3.weight']
    hidden2_dim = second_weight.shape[0]

    final_weight = probe_state['probe.6.weight']
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
    probe[0].weight.data = probe_state['probe.0.weight']
    probe[0].bias.data = probe_state['probe.0.bias']
    probe[3].weight.data = probe_state['probe.3.weight']
    probe[3].bias.data = probe_state['probe.3.bias']
    probe[6].weight.data = probe_state['probe.6.weight']
    probe[6].bias.data = probe_state['probe.6.bias']

    # Set to eval mode to disable dropout
    for module in probe.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    return probe


def build_binary_probe(probe_state):
    """Build binary probe from saved weights."""
    # Get dimensions from saved weights
    first_weight = probe_state['probe.0.weight']
    input_dim = first_weight.shape[1]
    hidden1_dim = first_weight.shape[0]

    second_weight = probe_state['probe.3.weight']
    hidden2_dim = second_weight.shape[0]

    final_weight = probe_state['probe.6.weight']
    num_classes = final_weight.shape[0]

    # Build probe network matching the saved architecture
    probe = nn.Sequential(
        nn.Linear(input_dim, hidden1_dim),   # 0
        nn.ReLU(),                            # 1
        nn.Dropout(0.5),                      # 2
        nn.Linear(hidden1_dim, hidden2_dim),  # 3
        nn.ReLU(),                            # 4
        nn.Dropout(0.5),                      # 5
        nn.Linear(hidden2_dim, num_classes)   # 6
    )

    # Load saved weights
    probe[0].weight.data = probe_state['probe.0.weight']
    probe[0].bias.data = probe_state['probe.0.bias']
    probe[3].weight.data = probe_state['probe.3.weight']
    probe[3].bias.data = probe_state['probe.3.bias']
    probe[6].weight.data = probe_state['probe.6.weight']
    probe[6].bias.data = probe_state['probe.6.bias']

    # Set to eval mode to disable dropout
    for module in probe.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    return probe


def load_conv3a_probe(probe_path='src/probe_training/trained_probes_20250904_080738/conv3a_probe.pt'):
    """Load conv3a multi-class probe."""
    probe_data = torch.load(probe_path, map_location='cpu')
    probe_state = probe_data['probe_state_dict']
    label_map = probe_data['label_map']
    reverse_map = {v: k for k, v in label_map.items()}

    probe = build_multiclass_probe(probe_state)
    probe.eval()

    return probe, label_map, reverse_map


def load_conv4a_probe(probe_path='src/probe_training/trained_probes_20250904_080738/conv4a_probe.pt'):
    """Load conv4a multi-class probe."""
    probe_data = torch.load(probe_path, map_location='cpu')
    probe_state = probe_data['probe_state_dict']

    probe = build_multiclass_probe(probe_state)
    probe.eval()

    return probe


def load_fc1_binary_probes(fc1_probe_dir='src/probe_training/binary_probes_20250925_183257/'):
    """Load all fc1 binary probes for each entity."""
    fc1_probes = {}
    entities = ['green_key', 'blue_key', 'red_key', 'gem', 'green_lock', 'blue_lock', 'red_lock']

    for entity in entities:
        probe_path = os.path.join(fc1_probe_dir, f'fc1_{entity}_probe.pt')
        try:
            probe_data = torch.load(probe_path, map_location='cpu')
            probe_state = probe_data['probe_state_dict']
            fc1_probes[entity] = build_binary_probe(probe_state)
            fc1_probes[entity].eval()
            print(f"Loaded fc1 probe for {entity}")
        except Exception as e:
            print(f"Warning: Could not load fc1 probe for {entity}: {e}")

    return fc1_probes