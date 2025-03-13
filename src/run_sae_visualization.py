#!/usr/bin/env python
# Script to run SAE feature visualization

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import load_interpretable_model
import gym
import argparse
import sys

# Print all modules being imported
print("Python path:", sys.path)
print("Current directory:", os.getcwd())

# Import SAE visualization function
print("Importing from feature_vis_sae...")
from feature_vis_sae import visualize_sae_features
print("Successfully imported visualize_sae_features")

# Import helpers from feature_vis_impala
print("Importing from feature_vis_impala...")
from feature_vis_impala import FeatureVisualizer, get_num_channels
print("Successfully imported FeatureVisualizer and get_num_channels")

# Check if we're accidentally importing the wrong function
print("Checking globals after imports:", [g for g in globals() if 'visualize' in g])

def main():
    parser = argparse.ArgumentParser(description='Visualize SAE features')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sae_checkpoint_step_4500000.pt', 
                        help='Path to SAE checkpoint')
    parser.add_argument('--layer', type=int, default=8, 
                        help='Layer number (if not extractable from checkpoint name)')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Number of features per image')
    parser.add_argument('--num_batches', type=int, default=32, 
                        help='Number of batch images to generate')
    parser.add_argument('--color_matrix', type=str, default='src/color_correlation.pt', 
                        help='Path to color correlation matrix')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    args = parser.parse_args()
    
    # Import ordered_layer_names to check layer mapping
    from extract_sae_features import ordered_layer_names
    
    # Print layer mapping for clarity
    print("\nLayer mapping:")
    for num, name in ordered_layer_names.items():
        print(f"  Layer {num}: {name}")
    
    # Check if the specified layer is valid
    if args.layer not in ordered_layer_names:
        print(f"WARNING: Layer number {args.layer} is not in the ordered_layer_names dictionary!")
    else:
        print(f"\nUsing layer {args.layer}: {ordered_layer_names[args.layer]}")
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
        print("Forcing CPU usage")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try with ../
        checkpoint_path = os.path.join('..', checkpoint_path)
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found at {args.checkpoint} or {checkpoint_path}")
            return
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load model
    print("Loading model...")
    model = load_interpretable_model()
    model.eval()
    
    # Move model to the correct device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Verify model is on the correct device
    print(f"Model device check: {next(model.parameters()).device}")
    
    # Print model structure if debug is enabled
    if args.debug:
        print("\nModel structure:")
        for name, module in model.named_modules():
            if hasattr(module, 'out_channels'):
                print(f"  {name}: {module.__class__.__name__} with {module.out_channels} output channels")
    
    # Visualize SAE features
    print(f"Visualizing SAE features from checkpoint: {checkpoint_path}")
    print(f"Layer number: {args.layer}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    
    # Check if we're accidentally running the regular feature visualization
    if 'visualize_all_channels' in globals():
        print("WARNING: Regular feature visualization function is in globals!")
    
    visualize_sae_features(
        model=model,
        sae_checkpoint_path=checkpoint_path,
        layer_number=args.layer,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        color_correlation_path=args.color_matrix
    )
    
if __name__ == "__main__":
    main() 