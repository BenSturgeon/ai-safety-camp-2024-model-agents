#!/usr/bin/env python3
import os
import argparse
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sae_cnn import (
    load_sae_from_checkpoint,
    ordered_layer_names,
    load_interpretable_model,
    get_module_by_path
)
from utils import heist

def identify_dead_channels(sae, model, layer_number, num_samples=1000, batch_size=64, threshold=1e-6):
    device = next(model.parameters()).device
    layer_name = ordered_layer_names[layer_number]
    
    # Get module for the target layer
    module = get_module_by_path(model, layer_name)
    
    # Track activations
    all_activations = []
    
    def hook_fn(module, input, output):
        with t.no_grad():
            output = output.to(device)
            
            # Get the latent activations from SAE encoder
            latent_acts = sae.encode(output)
            
            # Record activations
            all_activations.append(latent_acts.detach().cpu())
        
        return output
    
    # Register the forward hook
    handle = module.register_forward_hook(hook_fn)
    
    # Use heist environments to gather activations
    print(f"Collecting activations for {num_samples} maze samples...")
    
    # Create venv environments for sampling
    venv = heist.create_venv(num=batch_size, num_levels=0, start_level=1)
    
    # Collect samples in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    collected_samples = 0
    
    for _ in tqdm(range(num_batches)):
        try:
            # Get observations from the environment
            obs = venv.reset()
            
            # Convert to torch tensor if needed
            if isinstance(obs, np.ndarray):
                obs = t.tensor(obs, dtype=t.float32)
            
            # Forward through the model to trigger hook
            with t.no_grad():
                model(obs.to(device))
            
            collected_samples += obs.shape[0]
            
            # Check if we've collected enough samples
            if collected_samples >= num_samples:
                break
                
            venv = heist.create_venv(num=batch_size, num_levels=0, start_level=0)

        except Exception as e:
            print(f"Error collecting activations: {e}")
            continue
    
    # Clean up
    handle.remove()
    venv.close()
    
    # Process collected activations
    if not all_activations:
        print("No activations collected!")
        return [], []
    
    # Concatenate and analyze activations
    all_acts = t.cat(all_activations, dim=0)
    print(f"Collected activations with shape: {all_acts.shape}")
    
    # Calculate maximum activation for each channel across all samples and spatial positions
    # For convolutional layers - (B, C, H, W) - reduce across B, H, W
    # For FC layers - (B, C) - reduce across B
    if len(all_acts.shape) == 4:  # Conv layer
        max_activations = all_acts.max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
    else:  # FC layer
        max_activations = all_acts.max(dim=0)[0]
    
    # Identify dead channels
    dead_channels = (max_activations <= threshold).nonzero().squeeze(1).tolist()
    
    # Compute activation statistics per channel
    channel_stats = []
    for i in range(max_activations.shape[0]):
        if len(all_acts.shape) == 4:  # Conv layer
            channel_acts = all_acts[:, i, :, :].flatten()
        else:  # FC layer
            channel_acts = all_acts[:, i]
            
        mean_act = channel_acts.mean().item()
        max_act = channel_acts.max().item()
        sparsity = (channel_acts <= threshold).float().mean().item() * 100  # % of zeros
        
        channel_stats.append({
            'channel': i,
            'mean': mean_act,
            'max': max_act,
            'sparsity': sparsity,
            'is_dead': i in dead_channels
        })
    
    return dead_channels, channel_stats

def plot_activation_stats(channel_stats, output_path=None):
    channels = [stat['channel'] for stat in channel_stats]
    max_acts = [stat['max'] for stat in channel_stats]
    sparsity = [stat['sparsity'] for stat in channel_stats]
    dead_status = [stat['is_dead'] for stat in channel_stats]
    
    # Sort by max activation
    sort_idx = np.argsort(max_acts)
    sorted_channels = [channels[i] for i in sort_idx]
    sorted_max_acts = [max_acts[i] for i in sort_idx]
    sorted_sparsity = [sparsity[i] for i in sort_idx]
    sorted_dead = [dead_status[i] for i in sort_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot max activations
    bars = ax1.bar(sorted_channels, sorted_max_acts, color=['r' if is_dead else 'b' for is_dead in sorted_dead])
    ax1.set_yscale('log')
    ax1.set_ylabel('Max Activation (log scale)')
    ax1.set_title('Channel Activation Analysis')
    
    # Plot sparsity
    ax2.bar(sorted_channels, sorted_sparsity, color=['r' if is_dead else 'g' for is_dead in sorted_dead])
    ax2.set_xlabel('Channel Index (sorted by max activation)')
    ax2.set_ylabel('Sparsity % (zeros)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def export_simple_csv(channel_stats, csv_path):
    """Export a simplified CSV with just channel numbers and dead status"""
    simple_data = [{
        'channel': stat['channel'],
        'is_dead': stat['is_dead']
    } for stat in channel_stats]
    
    simple_df = pd.DataFrame(simple_data)
    simple_df.to_csv(csv_path, index=False)
    print(f"Simple dead channel report saved to {csv_path}")

def main(sae_path, num_samples=1000, batch_size=64, threshold=1e-6, output_dir=None):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    
    # Load the SAE model
    print(f"Loading SAE from checkpoint: {sae_path}")
    sae = load_sae_from_checkpoint(sae_path, device)
    
    # Extract layer number from checkpoint path if possible
    layer_match = Path(sae_path).name
    layer_number = None
    for part in Path(sae_path).parts:
        if part.startswith("layer_"):
            try:
                layer_number = int(part.split("_")[1])
                break
            except:
                pass
    
    if layer_number is None:
        print("Layer number not found in checkpoint path. Please specify:")
        for num, name in ordered_layer_names.items():
            print(f"  {num}: {name}")
        layer_number = int(input("Enter layer number: "))
    
    # Load the main model
    print("Loading base model...")
    model = load_interpretable_model().to(device)
    model.eval()
    
    # Identify dead channels
    print(f"Analyzing layer {layer_number} ({ordered_layer_names[layer_number]})...")
    dead_channels, channel_stats = identify_dead_channels(
        sae, model, layer_number, num_samples, batch_size, threshold
    )
    
    # Output results
    total_channels = len(channel_stats)
    dead_count = len(dead_channels)
    
    print(f"\nResults for layer {layer_number} ({ordered_layer_names[layer_number]}):")
    print(f"Total channels: {total_channels}")
    print(f"Dead channels: {dead_count} ({dead_count/total_channels*100:.1f}%)")
    
    if dead_channels:
        print(f"Dead channel indices: {dead_channels}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        layer_name = ordered_layer_names[layer_number]
        
        # Save visualization
        plot_path = os.path.join(output_dir, f"channel_analysis_{layer_name}.png")
        plot_activation_stats(channel_stats, plot_path)
        print(f"Plot saved to {plot_path}")
        
        # Save detailed channel stats to CSV
        detailed_csv_path = os.path.join(output_dir, f"channel_stats_detailed_{layer_name}.csv")
        stats_df = pd.DataFrame(channel_stats)
        stats_df.to_csv(detailed_csv_path, index=False)
        print(f"Detailed channel statistics saved to {detailed_csv_path}")
        
        # Save simple dead channel report
        simple_csv_path = os.path.join(output_dir, f"dead_channels_{layer_name}.csv")
        export_simple_csv(channel_stats, simple_csv_path)
    else:
        plot_activation_stats(channel_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect dead channels in SAE models")
    parser.add_argument("sae_path", help="Path to the SAE checkpoint")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to analyze")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generating activations")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Activation threshold below which a channel is considered dead")
    parser.add_argument("--output-dir", help="Directory to save analysis results")
    
    args = parser.parse_args()
    main(args.sae_path, args.samples, args.batch_size, args.threshold, args.output_dir) 