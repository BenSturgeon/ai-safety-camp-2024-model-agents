#!/usr/bin/env python3
"""
Train individual probes for each channel in conv4a and each row in fc1.
Scalable architecture for channel-wise probe training.
"""

import sys
import argparse
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial

from utils.helpers import load_interpretable_model, ModelActivations, observation_to_rgb, get_device


class SingleChannelDataset(Dataset):
    """Dataset for single channel/row features."""
    
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SingleChannelProbe(nn.Module):
    """Simple probe for single channel/row classification."""
    
    def __init__(self, input_dim, num_classes=5, hidden_dim=64):
        super().__init__()
        # Smaller network for single channel
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.probe(x)


def extract_layer_features(observations, model, layer_name='conv4a', batch_size=32):
    """
    Extract features from a specific layer.
    
    Returns:
        features: Array of shape (n_samples, channels, spatial_dims) for conv layers
                  or (n_samples, features) for fc layers
    """
    model_activations = ModelActivations(model)
    all_features = []
    
    for i in tqdm(range(0, len(observations), batch_size), 
                  desc=f"Extracting {layer_name}", leave=False):
        batch = observations[i:i+batch_size]
        batch_rgb = observation_to_rgb(batch)
        
        _, activations = model_activations.run_with_cache(batch_rgb, [layer_name])
        
        layer_key = layer_name.replace('.', '_')
        if layer_key in activations:
            feat = activations[layer_key]
            if isinstance(feat, tuple):
                feat = feat[0]
            
            all_features.append(feat.detach().cpu().numpy())
    
    model_activations.clear_hooks()
    
    if all_features:
        features = np.concatenate(all_features, axis=0)
        return features
    else:
        return None


def extract_channel_features(layer_features, channel_idx, layer_type='conv'):
    """
    Extract features for a single channel from conv layer or single row from fc layer.
    
    Args:
        layer_features: Full layer features
        channel_idx: Channel or row index
        layer_type: 'conv' or 'fc'
    
    Returns:
        channel_features: Flattened features for the channel/row
    """
    if layer_type == 'conv':
        # Conv layer: (batch, channels, height, width)
        if len(layer_features.shape) == 4:
            channel_feat = layer_features[:, channel_idx, :, :]
            # Flatten spatial dimensions
            channel_feat = channel_feat.reshape(channel_feat.shape[0], -1)
        else:
            raise ValueError(f"Expected 4D conv features, got shape {layer_features.shape}")
    
    elif layer_type == 'fc':
        # FC layer: (batch, features)
        if len(layer_features.shape) == 2:
            # For fc layers, we'll treat each neuron as a "channel"
            channel_feat = layer_features[:, channel_idx:channel_idx+1]
        else:
            raise ValueError(f"Expected 2D fc features, got shape {layer_features.shape}")
    
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    return channel_feat


def train_single_channel_probe(channel_features, labels, label_map, 
                              channel_idx, layer_name, num_epochs=30):
    """
    Train a probe for a single channel.
    
    Returns:
        Dictionary with results
    """
    # Split data
    n_samples = len(channel_features)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    
    train_features = channel_features[indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    val_features = channel_features[indices[n_train:]]
    val_labels = labels[indices[n_train:]]
    
    # Create datasets
    train_dataset = SingleChannelDataset(train_features, train_labels)
    val_dataset = SingleChannelDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Initialize probe
    input_dim = channel_features.shape[1]
    num_classes = len(label_map)
    probe = SingleChannelProbe(input_dim, num_classes)
    
    device = get_device()
    probe = probe.to(device)
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_probe_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        probe.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()
        
        # Validation
        probe.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = probe(batch_features)
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_probe_state = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Calculate per-class accuracy
    probe.load_state_dict(best_probe_state)
    probe.eval()
    
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = probe(batch_features)
            _, predicted = outputs.max(1)
            
            for label, pred in zip(batch_labels.cpu().numpy(), predicted.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[label_map[i]] = 100. * class_correct[i] / class_total[i]
        else:
            per_class_acc[label_map[i]] = 0.0
    
    return {
        'channel_idx': channel_idx,
        'layer_name': layer_name,
        'accuracy': best_val_acc,
        'per_class_accuracy': per_class_acc,
        'probe_state': best_probe_state,
        'input_dim': input_dim,
        'num_classes': num_classes
    }


def train_channel_probe_wrapper(args):
    """Wrapper for parallel processing."""
    channel_idx, layer_features, labels, label_map, layer_name, layer_type = args
    
    # Extract channel features
    channel_features = extract_channel_features(layer_features, channel_idx, layer_type)
    
    # Train probe
    result = train_single_channel_probe(
        channel_features, labels, label_map, 
        channel_idx, layer_name, num_epochs=30
    )
    
    return result


def train_all_channel_probes(layer_features, labels, label_map, 
                            layer_name, layer_type='conv', 
                            max_channels=None, n_jobs=4,
                            logger=None):
    """
    Train probes for all channels in parallel.
    
    Args:
        layer_features: Features from the layer
        labels: Target labels
        label_map: Label mapping
        layer_name: Name of the layer
        layer_type: 'conv' or 'fc'
        max_channels: Maximum number of channels to train (for testing)
        n_jobs: Number of parallel jobs
        logger: Logger instance
    
    Returns:
        results: List of dictionaries with results for each channel
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Determine number of channels
    if layer_type == 'conv':
        num_channels = layer_features.shape[1]
    else:  # fc
        num_channels = layer_features.shape[1]
    
    if max_channels:
        num_channels = min(num_channels, max_channels)
    
    logger.info(f"Training probes for {num_channels} channels/neurons in {layer_name}")
    
    # Prepare arguments for parallel processing
    args_list = [
        (i, layer_features, labels, label_map, layer_name, layer_type)
        for i in range(num_channels)
    ]
    
    # Train probes in parallel
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(train_channel_probe_wrapper, args_list),
            total=num_channels,
            desc=f"Training {layer_name} probes"
        ))
    
    return results


def analyze_results(results, label_map, top_k=10):
    """
    Analyze and summarize probe training results.
    
    Args:
        results: List of result dictionaries
        label_map: Label mapping
        top_k: Number of top channels to report
    
    Returns:
        summary: Dictionary with analysis results
    """
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    # Get top performing channels
    top_channels = sorted_results[:top_k]
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in results]
    
    summary = {
        'num_channels': len(results),
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'max_accuracy': np.max(accuracies),
        'min_accuracy': np.min(accuracies),
        'top_channels': []
    }
    
    for result in top_channels:
        summary['top_channels'].append({
            'channel_idx': result['channel_idx'],
            'accuracy': result['accuracy'],
            'per_class_accuracy': result['per_class_accuracy']
        })
    
    # Find channels specialized for each class
    class_specialists = {label: [] for label in label_map.values()}
    
    for result in results:
        for class_name, acc in result['per_class_accuracy'].items():
            if acc > 80:  # Threshold for specialization
                class_specialists[class_name].append({
                    'channel_idx': result['channel_idx'],
                    'accuracy': acc
                })
    
    # Sort specialists by accuracy
    for class_name in class_specialists:
        class_specialists[class_name] = sorted(
            class_specialists[class_name], 
            key=lambda x: x['accuracy'], 
            reverse=True
        )[:5]  # Top 5 for each class
    
    summary['class_specialists'] = class_specialists
    
    return summary


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train channel-wise probes')
    parser.add_argument('--max_channels', type=int, default=None,
                       help='Maximum number of channels to train (for testing)')
    parser.add_argument('--n_jobs', type=int, default=4,
                       help='Number of parallel jobs')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Channel-wise Probe Training")
    logger.info("="*60)
    
    # Load model
    logger.info("Loading model...")
    model_path = "../../base_models/full_run/model_35001.0.pt"
    if os.path.exists(model_path):
        model = load_interpretable_model(model_path=model_path)
    else:
        model = load_interpretable_model()
    model.eval()
    
    # Load dataset
    dataset_files = [f for f in os.listdir(os.path.dirname(__file__)) 
                    if f.startswith('empty_maze_dataset') and f.endswith('.pkl')]
    
    if not dataset_files:
        logger.error("No dataset found. Please run collect_empty_maze_data.py first.")
        return
    
    dataset_path = os.path.join(os.path.dirname(__file__), sorted(dataset_files)[-1])
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    observations = dataset['observations']
    labels = dataset['labels']
    label_map = dataset['label_map']
    
    logger.info(f"Dataset: {len(observations)} samples")
    
    # Configuration
    layers_to_train = [
        {'name': 'conv4a', 'type': 'conv'},
        {'name': 'fc1', 'type': 'fc'}
    ]
    
    # For testing, limit channels
    max_channels = args.max_channels
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), f'channel_probes_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    for layer_config in layers_to_train:
        layer_name = layer_config['name']
        layer_type = layer_config['type']
        
        logger.info(f"\nProcessing {layer_name} ({layer_type} layer)")
        
        # Extract layer features
        layer_features = extract_layer_features(observations, model, layer_name)
        
        if layer_features is None:
            logger.error(f"Failed to extract features for {layer_name}")
            continue
        
        logger.info(f"Extracted features shape: {layer_features.shape}")
        
        # Train probes for all channels
        results = train_all_channel_probes(
            layer_features, labels, label_map,
            layer_name, layer_type,
            max_channels=max_channels,
            n_jobs=args.n_jobs,
            logger=logger
        )
        
        # Analyze results
        summary = analyze_results(results, label_map)
        
        # Save results
        layer_results = {
            'layer_name': layer_name,
            'layer_type': layer_type,
            'results': results,
            'summary': summary
        }
        
        all_results[layer_name] = layer_results
        
        # Save individual layer results
        layer_file = os.path.join(results_dir, f'{layer_name}_results.pkl')
        with open(layer_file, 'wb') as f:
            pickle.dump(layer_results, f)
        
        # Print summary
        logger.info(f"\n{layer_name} Summary:")
        logger.info(f"  Mean accuracy: {summary['mean_accuracy']:.2f}%")
        logger.info(f"  Best accuracy: {summary['max_accuracy']:.2f}%")
        logger.info(f"  Top channels:")
        for ch in summary['top_channels'][:5]:
            logger.info(f"    Channel {ch['channel_idx']}: {ch['accuracy']:.2f}%")
    
    # Save all results
    all_results_file = os.path.join(results_dir, 'all_results.pkl')
    with open(all_results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save summary JSON
    summary_data = {}
    for layer_name, layer_results in all_results.items():
        summary_data[layer_name] = {
            'mean_accuracy': layer_results['summary']['mean_accuracy'],
            'max_accuracy': layer_results['summary']['max_accuracy'],
            'num_channels': layer_results['summary']['num_channels'],
            'top_5_channels': [
                {'idx': ch['channel_idx'], 'acc': ch['accuracy']} 
                for ch in layer_results['summary']['top_channels'][:5]
            ]
        }
    
    summary_file = os.path.join(results_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"\nResults saved to {results_dir}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()