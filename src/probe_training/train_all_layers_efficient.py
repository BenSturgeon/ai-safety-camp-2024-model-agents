#!/usr/bin/env python3
"""Train probes for all layers efficiently with single-pass feature extraction."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import time
import logging
from tqdm import tqdm
from datetime import datetime

from utils.helpers import load_interpretable_model, ModelActivations, observation_to_rgb, get_device


class ProbeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class NextTargetProbe(nn.Module):
    def __init__(self, input_dim, num_classes=7, hidden_dim=256):
        super().__init__()
        if input_dim < 1000:
            hidden_dim = min(hidden_dim, input_dim * 2)
        
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.probe(x)


def extract_all_features_single_pass(observations, model, layer_names, logger=None):
    """Extract features from ALL layers in a single pass through the data."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting features from {len(layer_names)} layers in single pass...")
    
    model_activations = ModelActivations(model)
    all_features = {layer: [] for layer in layer_names}
    
    batch_size = 32
    for i in tqdm(range(0, len(observations), batch_size), desc="Feature extraction"):
        batch = observations[i:i+batch_size]
        batch_rgb = observation_to_rgb(batch)
        
        _, activations = model_activations.run_with_cache(batch_rgb, layer_names)
        
        for layer_name in layer_names:
            layer_key = layer_name.replace('.', '_')
            if layer_key in activations:
                feat = activations[layer_key]
                if isinstance(feat, tuple):
                    feat = feat[0]
                
                if feat.ndim == 4:  
                    feat = feat.reshape(feat.shape[0], -1)
                elif feat.ndim == 3 and len(batch) == 1:
                    feat = feat.reshape(1, -1)
                elif feat.ndim == 2:
                    pass
                else:
                    feat = feat.reshape(feat.shape[0], -1)
                
                all_features[layer_name].append(feat.detach().cpu().numpy())
    
    model_activations.clear_hooks()
    
    for layer_name in layer_names:
        if all_features[layer_name]:
            all_features[layer_name] = np.concatenate(all_features[layer_name], axis=0)
            logger.info(f"  {layer_name}: shape {all_features[layer_name].shape}")
        else:
            logger.warning(f"  {layer_name}: no features extracted")
            all_features[layer_name] = None
    
    return all_features


def train_probe(features, labels, label_map, num_epochs=50):
    """Train a probe on extracted features."""
    n_samples = len(features)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    
    train_features = features[indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    val_features = features[indices[n_train:]]
    val_labels = labels[indices[n_train:]]
    
    train_dataset = ProbeDataset(train_features, train_labels)
    val_dataset = ProbeDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    input_dim = features.shape[1]
    num_classes = len(label_map)
    probe = NextTargetProbe(input_dim, num_classes)
    
    device = get_device()
    probe = probe.to(device)
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
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
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_acc


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Loading model...")
    model = load_interpretable_model()
    model.eval()
    
    dataset_path = os.path.join(os.path.dirname(__file__), 'balanced_dataset.pkl')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), '../next_target_probe/dataset_balanced.pkl')
    
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    observations = dataset['observations']
    labels = dataset['labels']
    label_map = dataset['label_map']
    
    logger.info(f"Dataset: {len(observations)} samples")
    
    layer_names = [
        'conv1a', 'conv2a', 'conv2b', 'conv3a', 'conv4a',
        'fc1', 'fc2', 'fc3', 'value_fc'
    ]
    
    start_time = time.time()
    all_features = extract_all_features_single_pass(observations, model, layer_names, logger)
    extraction_time = time.time() - start_time
    logger.info(f"Feature extraction completed in {extraction_time:.1f} seconds")
    
    results = {}
    
    logger.info("\nTraining probes for each layer...")
    for layer_name in layer_names:
        if all_features[layer_name] is not None:
            logger.info(f"Training probe for {layer_name}...")
            features = all_features[layer_name]
            
            try:
                best_acc = train_probe(features, labels, label_map, num_epochs=50)
                results[layer_name] = {
                    'accuracy': best_acc,
                    'dim': features.shape[1]
                }
                logger.info(f"  {layer_name}: Accuracy = {best_acc:.2f}%")
            except Exception as e:
                logger.error(f"  Error training {layer_name}: {e}")
                results[layer_name] = {'accuracy': 0.0, 'dim': features.shape[1]}
        else:
            results[layer_name] = {'accuracy': 0.0, 'dim': 0}
    
    results_file = os.path.join(os.path.dirname(__file__), 'probe_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("PROBE RESULTS - ALL LAYERS")
    print("="*60)
    print(f"{'Layer':<10} {'Dim':<8} {'Accuracy':<10}")
    print("-" * 35)
    
    for layer_name in layer_names:
        if layer_name in results:
            r = results[layer_name]
            print(f"{layer_name:<10} {r['dim']:<8} {r['accuracy']:6.2f}%")
    
    print("="*60)
    print(f"Total time: {time.time() - start_time:.1f}s (extraction: {extraction_time:.1f}s)")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()