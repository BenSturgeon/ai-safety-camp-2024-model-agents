#!/usr/bin/env python3
"""
Visualize samples from the empty maze dataset.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils.helpers import observation_to_rgb


def visualize_samples(dataset_path, num_samples=6):
    """Visualize samples from the dataset."""
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    observations = dataset['observations']
    labels = dataset['labels']
    label_map = dataset['label_map']
    metadata = dataset['metadata']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    # Show first num_samples
    for i in range(min(num_samples, len(observations))):
        obs = observations[i]
        label = labels[i]
        meta = metadata[i]
        
        # Convert to RGB for display
        img = observation_to_rgb(obs[np.newaxis, ...])[0]
        # Transpose from (C, H, W) to (H, W, C) for matplotlib
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label_map[label]}\n"
                         f"Player pos: {meta['player_pos']}\n"
                         f"Entities: {len(meta['entities'])}")
        axes[i].axis('off')
        
        # Add entity annotations
        for entity in meta['entities']:
            if entity['type'] == 'key':
                color_names = {0: 'B', 1: 'G', 2: 'R'}
                text = color_names.get(entity['color'], '?')
                axes[i].text(entity['x'] * 8, entity['y'] * 8, text, 
                           color='white', fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='circle', facecolor='black', alpha=0.5))
            elif entity['type'] == 'gem':
                axes[i].text(entity['x'] * 8, entity['y'] * 8, 'G', 
                           color='yellow', fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='circle', facecolor='black', alpha=0.5))
    
    plt.suptitle(f"Empty Maze Dataset Samples\nTotal: {len(observations)} samples")
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'sample_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Print label distribution
    print("\nLabel Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label_val, count in zip(unique, counts):
        print(f"  {label_map[label_val]}: {count} ({100*count/len(labels):.1f}%)")
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(observations)}")
    print(f"  Entity types: {dataset.get('entity_types', 'unknown')}")
    
    # Analyze entity distribution
    total_keys = 0
    total_gems = 0
    for meta in metadata:
        for entity in meta['entities']:
            if entity['type'] == 'key':
                total_keys += 1
            elif entity['type'] == 'gem':
                total_gems += 1
    
    print(f"  Average keys per maze: {total_keys/len(metadata):.2f}")
    print(f"  Average gems per maze: {total_gems/len(metadata):.2f}")
    
    plt.show()


def main():
    # Find the most recent dataset
    dataset_files = [f for f in os.listdir(os.path.dirname(__file__)) 
                    if f.startswith('empty_maze_dataset') and f.endswith('.pkl')]
    
    if not dataset_files:
        print("No dataset found. Please run collect_empty_maze_data.py first.")
        return
    
    dataset_path = os.path.join(os.path.dirname(__file__), sorted(dataset_files)[-1])
    print(f"Loading dataset from: {dataset_path}")
    
    visualize_samples(dataset_path)


if __name__ == "__main__":
    main()