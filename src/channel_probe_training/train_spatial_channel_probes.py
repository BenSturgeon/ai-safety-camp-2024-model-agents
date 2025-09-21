#!/usr/bin/env python3
"""
Efficient version: Collect activations once, train all channel probes in parallel.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import random
from tqdm import tqdm
import json

from channel_probe_training.collect_rollout_data import collect_dataset
from channel_probe_training.collect_rollout_data_balanced import collect_balanced_dataset_efficient
from utils.helpers import load_interpretable_model, get_device

 
class SimpleProbe(nn.Module):
    """Simple linear probe for channel/neuron features."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_probe_for_channel(args):
    """Train a single probe for a channel/neuron - for parallel processing."""
    layer_name, channel_idx, train_features, train_labels, test_features, test_labels, label_map = args
    
    device = 'cpu'
    num_classes = len(label_map)
    input_dim = train_features.shape[1]
    
    probe = SimpleProbe(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_x = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.long).to(device)
    test_x = torch.tensor(test_features, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_labels, dtype=torch.long).to(device)
    
    probe.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = probe(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
    
    # Test
    probe.eval()
    with torch.no_grad():
        outputs = probe(test_x)
        _, predicted = outputs.max(1)
        predicted = predicted.cpu().numpy()
    
    test_labels_np = test_y.cpu().numpy()
    
    # Overall accuracy
    overall_acc = (predicted == test_labels_np).mean() * 100
    
    # Per-entity accuracy
    per_entity = {}
    for entity_name, label_idx in label_map.items():
        mask = test_labels_np == label_idx
        if mask.sum() > 0:
            acc = (predicted[mask] == test_labels_np[mask]).mean() * 100
            per_entity[entity_name] = acc
    
    return {
        'layer': layer_name,
        'channel': channel_idx,
        'overall_acc': overall_acc,
        'per_entity': per_entity
    }


def extract_channel_features_from_activations(activations, channel_idx, layer_type):
    """Extract features for a specific channel from pre-computed activations.

    For conv layers: Flattens spatial dimensions to preserve all information.
    For fc layers: Extracts the specific neuron.

    Args:
        activations: The activation tensor
        channel_idx: Which channel to extract
        layer_type: 'conv' or 'fc'
    """
    # Remove the extra dimension (batch dim from model forward pass)
    if len(activations.shape) > 2 and activations.shape[1] == 1:
        activations = activations.squeeze(1)

    if layer_type == 'conv':
        # activations shape: (n_samples, channels, height, width)
        # Extract specific channel and flatten spatial dimensions
        channel_acts = activations[:, channel_idx, :, :]
        # Shape: (n_samples, height * width)
        features = channel_acts.reshape(channel_acts.shape[0], -1)
    else:  # fc
        # activations shape: (n_samples, neurons)
        # Just extract the specific neuron
        features = activations[:, channel_idx].reshape(-1, 1)

    return features


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train probes on all channels efficiently')
    parser.add_argument('--env-type', choices=['empty', 'standard'], default='empty',
                       help='Environment type: empty (no enemies) or standard (normal maze)')
    parser.add_argument('--num-rollouts', type=int, default=20,
                       help='Number of rollouts to collect')
    parser.add_argument('--balanced', action='store_true',
                       help='Use balanced collection (guarantees equal samples per entity)')
    parser.add_argument('--samples-per-entity', type=int, default=500,
                       help='Samples per entity for balanced collection')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    print("="*60)
    print("EFFICIENT All Channels Per-Entity Analysis")
    print(f"Environment: {args.env_type.upper()}")
    print(f"Feature extraction: Flattened spatial (preserves all information)")
    print("="*60)
    
    device = 'cpu'
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("\nLoading model...")
    model_path = "../../base_models/full_run/model_35001.0.pt"
    if os.path.exists(model_path):
        model = load_interpretable_model(model_path=model_path)
    else:
        alt_path = os.path.join(os.path.dirname(__file__), "../../base_models/full_run/model_35001.0.pt")
        if os.path.exists(alt_path):
            model = load_interpretable_model(model_path=alt_path)
        else:
            model = load_interpretable_model()
    model.eval()
    model = model.to(device)
    
    print(f"\nCollecting BALANCED data ({args.samples_per_entity} samples per entity) from {args.env_type} environment...")
    observations, activations_dict, labels, label_map, metadata = collect_balanced_dataset_efficient(
        model,
        samples_per_entity=args.samples_per_entity,
        collect_every=1,
        env_type=args.env_type,
        logger=None
    )
    
    print(f"\nCollected {len(observations)} total samples")
    print(f"Label map: {label_map}")
    
    # Print label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    reverse_label_map = {v: k for k, v in label_map.items()}
    for label_val, count in zip(unique, counts):
        print(f"  {reverse_label_map[label_val]}: {count} ({100*count/len(labels):.1f}%)")
    
    first_layer = 'conv1a'
    if first_layer in activations_dict and len(activations_dict[first_layer]) > 0:
        n_activation_samples = len(activations_dict[first_layer])
    else:
        print("Error: No activations found")
        return
    
    print(f"Activation samples available: {n_activation_samples}")

    if args.samples_per_entity >= 1000:
        layers_config = [
            {'name': 'conv3a', 'type': 'conv', 'channels': 32},
            {'name': 'conv4a', 'type': 'conv', 'channels': 32},
        ]
    else:
        layers_config = [
            {'name': 'conv1a', 'type': 'conv', 'channels': 16},
            {'name': 'conv2a', 'type': 'conv', 'channels': 32},
            {'name': 'conv2b', 'type': 'conv', 'channels': 32},
            {'name': 'conv3a', 'type': 'conv', 'channels': 32},
            {'name': 'conv4a', 'type': 'conv', 'channels': 32},
            {'name': 'fc1', 'type': 'fc', 'channels': 256},
            {'name': 'fc2', 'type': 'fc', 'channels': 256},
            {'name': 'fc3', 'type': 'fc', 'channels': 256}
        ]
    
    # Debug: check what layers have activations
    print("\nActivations per layer:")
    for layer in layers_config:
        layer_name = layer['name']
        if layer_name in activations_dict and len(activations_dict[layer_name]) > 0:
            acts = np.array(activations_dict[layer_name])
            print(f"  {layer_name}: shape {acts.shape}")
        else:
            print(f"  {layer_name}: NO DATA")
    
    if args.balanced:
        # Data is already balanced from collection
        print("\nData is already balanced from collection")
        labels_subset = labels[:n_activation_samples]
        
        # Use all the balanced data
        final_indices = np.arange(len(labels_subset))
        np.random.shuffle(final_indices)
    else:
        # Balance the dataset - get equal samples per entity
        print("\nBalancing dataset by entity...")
        labels_subset = labels[:n_activation_samples]
        
        # Get indices for each entity type
        entity_indices = {}
        for entity_name, entity_label in label_map.items():
            entity_indices[entity_name] = np.where(labels_subset == entity_label)[0]
            print(f"  {entity_name}: {len(entity_indices[entity_name])} samples available")
        
        # Sample up to 2000 per category
        samples_per_category = 2000
        balanced_indices = []
        
        for entity_name, indices in entity_indices.items():
            if len(indices) > samples_per_category:
                # Randomly sample 2000
                sampled = np.random.choice(indices, samples_per_category, replace=False)
            else:
                # Use all available
                sampled = indices
            balanced_indices.extend(sampled)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        # Use ALL balanced data instead of sampling
        print(f"\nUsing all {len(balanced_indices)} balanced samples...")
        final_indices = balanced_indices
    
    # Split into train/test (90/10 split to maximize training data)
    n_train = int(0.9 * len(final_indices))
    n_test = len(final_indices) - n_train  # Use all remaining for test
    
    np.random.shuffle(final_indices)
    train_indices = final_indices[:n_train]
    test_indices = final_indices[n_train:n_train+n_test]
    
    train_labels = labels_subset[train_indices]
    test_labels = labels_subset[test_indices]
    
    print(f"Train: {len(train_labels)}, Test: {len(test_labels)}")
    
    # Prepare all training tasks
    all_tasks = []
    
    for layer_cfg in layers_config:
        layer_name = layer_cfg['name']
        layer_type = layer_cfg['type']
        
        # Get activations for this layer
        if layer_name not in activations_dict or len(activations_dict[layer_name]) == 0:
            print(f"Warning: No activations for {layer_name}")
            continue
        
        layer_acts = np.array(activations_dict[layer_name])
        
        # Remove extra dimension if present (from batch processing)
        if len(layer_acts.shape) > 2 and layer_acts.shape[1] == 1:
            layer_acts = layer_acts.squeeze(1)
        
        # Sample the activations
        train_acts = layer_acts[train_indices]
        test_acts = layer_acts[test_indices]
        
        # Determine number of channels to process
        if layer_type == 'conv':
            # Now shape should be (n_samples, channels, h, w)
            actual_channels = layer_acts.shape[1] if len(layer_acts.shape) > 1 else 1
            num_channels = min(actual_channels, 32)  # All conv layers have <=32 channels
            
            # Create task for each channel
            for ch_idx in range(num_channels):
                # Extract channel features with flattened spatial dimensions
                train_features = extract_channel_features_from_activations(train_acts, ch_idx, layer_type)
                test_features = extract_channel_features_from_activations(test_acts, ch_idx, layer_type)
                
                all_tasks.append((
                    layer_name, ch_idx, 
                    train_features, train_labels,
                    test_features, test_labels,
                    label_map
                ))
        else:  # fc - train on whole layer
            # For FC layers, use all features together
            train_features = train_acts.reshape(train_acts.shape[0], -1)  # Flatten to (n_samples, features)
            test_features = test_acts.reshape(test_acts.shape[0], -1)
            
            all_tasks.append((
                layer_name, 'all',  # Use 'all' instead of channel index
                train_features, train_labels,
                test_features, test_labels,
                label_map
            ))
    
    print(f"\nTraining {len(all_tasks)} probes sequentially (CPU-only)...")
    
    # Train all probes sequentially to avoid CUDA multiprocessing issues
    results = []
    for task in tqdm(all_tasks, desc="Training all probes"):
        result = train_probe_for_channel(task)
        results.append(result)
    
    layer_results = {}
    for result in results:
        layer = result['layer']
        if layer not in layer_results:
            layer_results[layer] = []
        layer_results[layer].append(result)
    
    print("\n" + "="*70)
    print("SUMMARY BY LAYER")
    print("="*70)
    
    for layer_name in ['conv1a', 'conv2a', 'conv2b', 'conv3a', 'conv4a', 'fc1', 'fc2', 'fc3']:
        if layer_name not in layer_results:
            continue
        
        print(f"\n{layer_name.upper()}:")
        
        sorted_results = sorted(layer_results[layer_name], 
                              key=lambda x: x['overall_acc'], 
                              reverse=True)
        
        # Top 5 channels
        print("  Top 5 channels:")
        for i, res in enumerate(sorted_results[:5]):
            ch_str = str(res['channel']) if res['channel'] != 'all' else 'all'
            print(f"    Ch{ch_str:>3s}: {res['overall_acc']:5.1f}% overall")
            for entity, acc in res['per_entity'].items():
                print(f"          {entity:10s}: {acc:5.1f}%")
    
    print("\n" + "="*70)
    print("ENTITY SPECIALISTS (>70% accuracy)")
    print("="*70)
    
    entity_specialists = {entity: [] for entity in label_map.keys()}
    
    for result in results:
        for entity, acc in result['per_entity'].items():
            if acc > 70:
                entity_specialists[entity].append({
                    'layer': result['layer'],
                    'channel': result['channel'],
                    'accuracy': acc
                })
    
    for entity in label_map.keys():
        print(f"\n{entity.upper()}:")
        specialists = sorted(entity_specialists[entity], 
                           key=lambda x: x['accuracy'], 
                           reverse=True)
        
        if specialists:
            for spec in specialists[:10]:
                ch_str = str(spec['channel']) if spec['channel'] != 'all' else 'all'
                print(f"  {spec['layer']:8s} ch{ch_str:3s}: {spec['accuracy']:.1f}%")
        else:
            print("  No strong specialists found")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"efficient_all_channels_{args.env_type}_seed{args.seed}_{timestamp}.json"

    # Save activations and observations for later analysis
    # Save EXACTLY what was used for training/testing for full reproducibility
    import pickle
    cache_file = f"efficient_all_channels_{args.env_type}_{timestamp}_cache.pkl"
    print(f"\nSaving dataset cache for reproducibility to {cache_file}...")

    # Sort indices to ensure consistent ordering
    train_indices_sorted = np.sort(train_indices)
    test_indices_sorted = np.sort(test_indices)

    # Create arrays to save exactly what was used
    train_observations = [observations[i] for i in train_indices_sorted if i < len(observations)]
    test_observations = [observations[i] for i in test_indices_sorted if i < len(observations)]

    train_labels_save = labels_subset[train_indices_sorted]
    test_labels_save = labels_subset[test_indices_sorted]

    # Save train/test activations separately for clarity
    train_activations = {}
    test_activations = {}
    for layer_name, layer_acts in activations_dict.items():
        if len(layer_acts) > 0:
            layer_acts_array = np.array(layer_acts)
            train_activations[layer_name] = layer_acts_array[train_indices_sorted]
            test_activations[layer_name] = layer_acts_array[test_indices_sorted]

    # Include metadata for full reproducibility
    dataset_info = {
        'train': {
            'observations': train_observations,
            'activations': train_activations,
            'labels': train_labels_save,
            'original_indices': train_indices_sorted.tolist(), 
            'n_samples': len(train_indices_sorted)
        },
        'test': {
            'observations': test_observations,
            'activations': test_activations,
            'labels': test_labels_save,
            'original_indices': test_indices_sorted.tolist(),  
            'n_samples': len(test_indices_sorted)
        },
        'label_map': label_map,
        'environment_type': args.env_type,
        'collection_params': {
            'balanced': args.balanced,
            'samples_per_entity': args.samples_per_entity if args.balanced else None,
            'num_rollouts': args.num_rollouts if not args.balanced else None,
            'random_seed': args.seed  
        },
        'split_info': {
            'train_ratio': 0.9,
            'test_ratio': 0.1,
            'total_used_samples': len(train_indices) + len(test_indices),
            'total_collected_samples': len(observations)
        },
        'metadata': metadata if 'metadata' in locals() else None,
        'timestamp': timestamp
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(dataset_info, f)

    print(f"Cache saved ({os.path.getsize(cache_file) / (1024*1024):.1f} MB)")
    print(f"  Train: {len(train_indices_sorted)} samples")
    print(f"  Test: {len(test_indices_sorted)} samples")
    print(f"  Total: {len(train_indices_sorted) + len(test_indices_sorted)} samples")
    
    # Organize all results by layer for JSON output
    json_results_by_layer = {}
    for layer_name in ['conv1a', 'conv2a', 'conv2b', 'conv3a', 'conv4a', 'fc1', 'fc2', 'fc3']:
        if layer_name not in layer_results:
            continue
        
        json_results_by_layer[layer_name] = []
        # Include ALL channels, not just top 5
        for res in layer_results[layer_name]:
            json_results_by_layer[layer_name].append({
                'channel': res['channel'],
                'overall_acc': res['overall_acc'],
                'per_entity': res['per_entity']
            })
    
    # Also save a flat list of all results for easier processing
    json_results_flat = []
    for result in results:
        json_results_flat.append({
            'layer': result['layer'],
            'channel': result['channel'],
            'overall_acc': result['overall_acc'],
            'per_entity': result['per_entity']
        })
    
    with open(output_file, 'w') as f:
        json.dump({
            'environment_type': args.env_type,
            'n_train': len(train_labels),
            'n_test': len(test_labels),
            'label_map': label_map,
            'results_by_layer': json_results_by_layer,
            'results_flat': json_results_flat,
            'entity_specialists': entity_specialists
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()