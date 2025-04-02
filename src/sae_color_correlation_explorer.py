#!/usr/bin/env python
# SAE Color Correlation Explorer
# 
# This script explores how SAE features are affected by color correlation matrices.
# %% 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import argparse

# Add parent directory to path if needed
import sys
if not os.path.exists('src') and os.path.exists('../src'):
    sys.path.append('..')
    sys.path.append('.')

    print("Added parent directory to path")

# Import helper functions from existing modules
from utils.helpers import load_interpretable_model
from sae_cnn import ConvSAE
from feature_vis_sae import (
    load_sae_from_checkpoint,
    SAEFeatureVisualizer,
    ordered_layer_names
)
from feature_vis_impala import (
    total_variation, 
    jitter, 
    random_scale, 
    random_rotate, 
    apply_color_correlation,
    load_color_correlation_matrix, 
    get_num_channels
)
from extract_sae_features import replace_layer_with_sae


# %%
def create_color_correlation_matrix(matrix_type='identity', device=None):
    """
    Create different types of color correlation matrices
    
    Args:
        matrix_type: Type of matrix to create ('identity', 'random', 'red_emphasis', etc.)
        device: Device to create the matrix on
        
    Returns:
        Color correlation matrix as a tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if matrix_type == 'identity':
        return torch.eye(3, device=device)
    elif matrix_type == 'random':
        # Random matrix with values between -1 and 1
        matrix = torch.randn(3, 3, device=device)
        # Normalize to have reasonable values
        matrix = matrix / matrix.norm(dim=1, keepdim=True)
        return matrix
    elif matrix_type == 'red_emphasis':
        # Emphasize red channel
        matrix = torch.eye(3, device=device)
        matrix[0, 0] = 2.0  # Boost red channel
        return matrix
    elif matrix_type == 'green_emphasis':
        # Emphasize green channel
        matrix = torch.eye(3, device=device)
        matrix[1, 1] = 2.0  # Boost green channel
        return matrix
    elif matrix_type == 'blue_emphasis':
        # Emphasize blue channel
        matrix = torch.eye(3, device=device)
        matrix[2, 2] = 2.0  # Boost blue channel
        return matrix
    elif matrix_type == 'grayscale':
        # Convert to grayscale (all channels equally weighted)
        matrix = torch.ones(3, 3, device=device) / 3
        return matrix
    else:
        print(f"Unknown matrix type: {matrix_type}, using identity")
        return torch.eye(3, device=device)

def compare_color_correlation_matrices(visualizer, feature_idx, output_dir="color_correlation_comparison"):
    """
    Compare visualizations with different color correlation matrices
    
    Args:
        visualizer: SAEFeatureVisualizer instance
        feature_idx: Feature to visualize
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List of matrix types to compare
    matrix_types = ['identity', 'red_emphasis', 'green_emphasis', 'blue_emphasis', 'grayscale', 'random']
    
    # Create a figure to display all visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Visualize with each matrix type
    for i, matrix_type in enumerate(matrix_types):
        print(f"\nTesting color correlation matrix: {matrix_type}")
        
        # Create the matrix
        matrix = create_color_correlation_matrix(matrix_type, visualizer.device)
        
        # Update the visualizer's color correlation matrix
        visualizer.color_correlation = matrix
        
        # Visualize the feature
        vis, activation = visualizer.visualize_sae_feature(
            LAYER_NUMBER, 
            feature_idx,
            num_steps=500,  # Fewer steps for quicker comparison
            lr=0.08,
            tv_weight=1e-3,
            l2_weight=1e-3,
            jitter_amount=8
        )
        
        # Display the visualization
        axes[i].imshow(vis)
        axes[i].set_title(f"{matrix_type}\nActivation: {activation:.4f}")
        axes[i].axis('off')
        
        # Save individual visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(vis)
        plt.title(f"Feature {feature_idx} with {matrix_type}\nActivation: {activation:.4f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"feature_{feature_idx}_{matrix_type}.png"), dpi=150)
        plt.close()
    
    # Save the comparison figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_{feature_idx}_comparison.png"), dpi=150)
    plt.close(fig)
    
    print(f"Saved color correlation comparison to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='SAE Color Correlation Explorer')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/sae_checkpoint_step_4500000.pt",
                        help='Path to SAE checkpoint')
    parser.add_argument('--layer', type=int, default=8,
                        help='Layer number (default: 8 for conv4a)')
    parser.add_argument('--feature', type=int, default=0,
                        help='Feature index to visualize (default: 0)')
    parser.add_argument('--color_matrix', type=str, default="src/color_correlation.pt",
                        help='Path to color correlation matrix')
    parser.add_argument('--output_dir', type=str, default="color_correlation_results",
                        help='Directory to save results')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    args = parser.parse_args()
    
    # Set device
    global LAYER_NUMBER
    LAYER_NUMBER = args.layer
    
    if args.cpu:
        device = torch.device('cpu')
        print("Forcing CPU usage")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the base model
    print("Loading model...")
    model = load_interpretable_model()
    model.eval()
    model = model.to(device)
    
    # Print model device
    print(f"Model device: {next(model.parameters()).device}")
    
    # Print layer mapping for reference
    print("\nLayer mapping:")
    for num, name in ordered_layer_names.items():
        print(f"  Layer {num}: {name}")
    
    # Load the SAE from checkpoint
    print(f"\nLoading SAE from checkpoint: {args.checkpoint}")
    sae, checkpoint_layer_number = load_sae_from_checkpoint(args.checkpoint, device)
    print(f"Loaded SAE with {sae.hidden_channels} hidden channels")
    
    # Use layer number from checkpoint if available and no layer specified
    if checkpoint_layer_number is not None and args.layer == 8:  # 8 is the default
        LAYER_NUMBER = checkpoint_layer_number
        print(f"Using layer number from checkpoint: {LAYER_NUMBER} ({ordered_layer_names[LAYER_NUMBER]})")
    else:
        print(f"Using specified layer number: {LAYER_NUMBER} ({ordered_layer_names[LAYER_NUMBER]})")
    
    # Load or create color correlation matrix
    if os.path.exists(args.color_matrix):
        color_correlation = load_color_correlation_matrix(args.color_matrix, device)
        print(f"Loaded color correlation matrix from {args.color_matrix}")
    else:
        color_correlation = torch.eye(3, device=device)
        print("Using default identity color correlation matrix")
    
    # Save the color correlation matrix visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(color_correlation.cpu().numpy())
    plt.colorbar()
    plt.title("Color Correlation Matrix")
    plt.savefig(os.path.join(args.output_dir, "color_correlation_matrix.png"), dpi=150)
    plt.close()
    
    # Initialize the visualizer
    print("\nInitializing SAE Feature Visualizer...")
    visualizer = SAEFeatureVisualizer(model, args.color_matrix)
    visualizer.load_sae(sae, LAYER_NUMBER)
    
    # Compare different color correlation matrices
    print(f"\nComparing color correlation matrices for feature {args.feature}...")
    compare_color_correlation_matrices(
        visualizer, 
        args.feature, 
        output_dir=os.path.join(args.output_dir, "matrix_comparison")
    )
    
    # Test with and without color correlation
    print("\nComparing with and without color correlation...")
    
    # With color correlation (original)
    print("\nVisualizing with color correlation...")
    vis_with, act_with = visualizer.visualize_sae_feature(
        LAYER_NUMBER, 
        args.feature,
        num_steps=1000
    )
    
    # Without color correlation (identity matrix)
    print("\nVisualizing without color correlation...")
    visualizer.color_correlation = torch.eye(3, device=device)
    vis_without, act_without = visualizer.visualize_sae_feature(
        LAYER_NUMBER, 
        args.feature,
        num_steps=1000
    )
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(vis_with)
    axes[0].set_title(f"With Color Correlation\nActivation: {act_with:.4f}")
    axes[0].axis('off')
    
    axes[1].imshow(vis_without)
    axes[1].set_title(f"Without Color Correlation\nActivation: {act_without:.4f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"feature_{args.feature}_with_without_comparison.png"), dpi=150)
    plt.close()
    
    print(f"\nAll results saved to {args.output_dir}")
    
    # Clean up
    visualizer._remove_hooks()
    print("Cleaned up hooks and resources")

if __name__ == "__main__":
    main() 
# %%
