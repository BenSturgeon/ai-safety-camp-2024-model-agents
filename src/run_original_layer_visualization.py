#!/usr/bin/env python
# Script to run original layer feature visualization using the SAE visualization pipeline

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import load_interpretable_model
import gym
import argparse
import sys

# Import the visualization function
from feature_vis_sae import visualize_original_layer

def main():
    parser = argparse.ArgumentParser(description='Visualize original model layer features')
    parser.add_argument('--layer', type=str, default='conv4a', 
                        help='Layer name to visualize (e.g., conv4a)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Number of features per image')
    parser.add_argument('--num_batches', type=int, default=1, 
                        help='Number of batch images to generate')
    parser.add_argument('--color_matrix', type=str, default='src/color_correlation.pt', 
                        help='Path to color correlation matrix')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    args = parser.parse_args()
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
        print("Forcing CPU usage")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Visualize original layer features
    print(f"Visualizing original layer features for: {args.layer}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    
    visualize_original_layer(
        model=model,
        layer_name=args.layer,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        color_correlation_path=args.color_matrix
    )
    
if __name__ == "__main__":
    main() 