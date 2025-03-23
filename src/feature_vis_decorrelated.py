import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import load_interpretable_model
import gym
from tqdm import tqdm
import os
import argparse

# Import our color decorrelation module
from color_decorrelation import (
    compute_color_correlation_matrix,
    collect_dataset_from_env,
    apply_whitening,
    apply_unwhitening,
    load_color_matrices
)

# Import visualization utilities
from feature_vis_impala import (
    total_variation, 
    jitter, 
    random_scale, 
    random_rotate,
    get_num_channels
)

# Import SAE-related modules if needed
try:
    from sae_cnn import ConvSAE
    from extract_sae_features import replace_layer_with_sae
    SAE_AVAILABLE = True
except ImportError:
    SAE_AVAILABLE = False
    print("SAE modules not available. Only original layer visualization will be supported.")

# Ordered layer names mapping
ordered_layer_names = {
    1: "conv1a",
    2: "pool1",
    3: "conv2a",
    4: "conv2b",
    5: "pool2",
    6: "conv3a",
    7: "pool3",
    8: "conv4a",
    9: "pool4",
    10: "fc1",
    11: "fc2",
    12: "fc3",
    13: "value_fc",
    14: "dropout_conv",
    15: "dropout_fc",
}

class DecorrelatedFeatureVisualizer:
    def __init__(self, model, color_matrices_path=None):
        """
        Initialize the decorrelated feature visualizer
        
        Args:
            model: The base model 
            color_matrices_path: Path to color matrices file
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.activations = {}
        self.hooks = []
        
        # Load color matrices if provided
        if color_matrices_path and os.path.exists(color_matrices_path):
            print(f"Loading color matrices from {color_matrices_path}")
            self.cov_matrix, self.whitening_matrix, self.unwhitening_matrix, self.mean_color = load_color_matrices(
                color_matrices_path, self.device
            )
            self.use_decorrelation = True
        else:
            print("No color matrices provided. Using identity transformation.")
            self.whitening_matrix = torch.eye(3, device=self.device)
            self.unwhitening_matrix = torch.eye(3, device=self.device)
            self.mean_color = torch.zeros(3, device=self.device)
            self.use_decorrelation = False

    def _register_hooks(self, layer_name):
        """
        Register hooks to capture layer activations
        
        Args:
            layer_name: The name of the layer to hook
        """
        # Clear existing hooks
        self._remove_hooks()
        
        # Define a hook to capture the layer activations
        def activation_hook(module, input, output):
            self.activations[layer_name] = output
            return output
            
        # Find the module and register the hook
        for name, layer in self.model.named_modules():
            if name == layer_name:
                hook = layer.register_forward_hook(activation_hook)
                self.hooks.append(hook)
                print(f"Registered hook on layer: {name}")
                break

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def visualize_feature(self, layer_name, feature_idx, 
                         num_steps=2560, lr=0.08, tv_weight=1e-3, 
                         l2_weight=1e-3, jitter_amount=8,
                         batch_info=None):
        """
        Visualize what maximally activates a specific feature using color decorrelation
        
        Args:
            layer_name: Name of the layer containing the feature
            feature_idx: Index of the feature to visualize
            num_steps: Number of optimization steps
            lr: Learning rate
            tv_weight: Weight for total variation regularization
            l2_weight: Weight for L2 regularization
            jitter_amount: Amount of jitter to apply during optimization
            batch_info: Optional tuple (batch_idx, batch_size, total_batches, feature_in_batch, total_features)
                        for progress reporting
            
        Returns:
            tuple: (visualization_image, activation_strength)
                - visualization_image: numpy array of the visualization (H, W, C)
                - activation_strength: float value of the highest activation achieved
        """
        # Register hook for the target layer
        self._register_hooks(layer_name)
        
        # Create progress description if batch info is provided
        progress_desc = ""
        if batch_info:
            batch_idx, batch_size, total_batches, feature_in_batch, total_features = batch_info
            current_feature = batch_idx * batch_size + feature_in_batch
            overall_progress = (current_feature / total_features) * 100
            progress_desc = f"Batch {batch_idx+1}/{total_batches}, Feature {feature_in_batch+1}/{batch_size} ({feature_idx}) - {overall_progress:.1f}%"
        else:
            progress_desc = f"Feature {feature_idx}"
        
        print(f"\nVisualizing {progress_desc}")
        print(f"Layer: {layer_name}")
        print(f"Using color decorrelation: {self.use_decorrelation}")
        
        # Initialize padded random image
        padded_size = 64 + 8  # 4 pixels on each side
        input_img = torch.randint(0, 256, (1, 3, padded_size, padded_size), 
                                device=self.device, 
                                dtype=torch.float32).requires_grad_(True)
        
        print(f"Input image shape: {input_img.shape}, device: {input_img.device}")
        
        optimizer = optim.Adam([input_img], lr=lr)
        
        # Parameters for transformations
        scales = [1.0, 0.975, 1.025, 0.95, 1.05]
        angles = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        
        best_activation = float('-inf')
        best_img = None
        
        try:
            # Use tqdm for progress tracking
            pbar = tqdm(range(num_steps), desc=progress_desc)
            for step in pbar:
                optimizer.zero_grad()
                
                # Create a copy for transformations
                processed_img = input_img.clone()
                
                # Apply sequence of transformations
                if jitter_amount > 0:
                    # First jitter
                    ox, oy = np.random.randint(-jitter_amount, jitter_amount+1, 2)
                    processed_img = jitter(processed_img, ox, oy)
                    
                    # Scale
                    processed_img = random_scale(processed_img, scales)
                    
                    # Rotate
                    processed_img = random_rotate(processed_img, angles)
                    
                    # Second jitter with smaller magnitude
                    ox, oy = np.random.randint(-4, 5, 2)
                    processed_img = jitter(processed_img, ox, oy)
                
                # Crop padding
                processed_img = processed_img[:, :, 4:-4, 4:-4]
                
                # Ensure values are in [0, 255] before normalizing
                processed_img.data.clamp_(0, 255)
                
                # Apply color whitening if enabled
                if self.use_decorrelation:
                    # Normalize to [0,1] for whitening
                    normalized_img = processed_img / 255.0
                    
                    # Apply whitening transformation
                    whitened_img = apply_whitening(normalized_img, self.whitening_matrix, self.mean_color)
                    
                    # Use the whitened image for the forward pass
                    model_input = whitened_img
                else:
                    # Just normalize without whitening
                    model_input = processed_img / 255.0
                
                # Forward pass - this will trigger our hooks
                _ = self.model(model_input)
                
                # Get the activation for the target feature
                if layer_name in self.activations:
                    # Extract the specific feature activation
                    feature_activation = self.activations[layer_name][0, feature_idx]
                    activation_loss = -feature_activation.mean()  # negative because we want to maximize
                else:
                    activation_loss = torch.tensor(0.0, device=self.device)
                
                # Calculate regularization losses
                tv_loss = tv_weight * total_variation(input_img / 255.0)
                l2_loss = l2_weight * torch.norm(input_img / 255.0)
                
                # Total loss
                loss = activation_loss + tv_loss + l2_loss
                
                loss.backward()
                optimizer.step()
                
                # Post-processing steps
                with torch.no_grad():
                    input_img.data.clamp_(0, 255)
                    
                    # Track best activation
                    current_activation = -activation_loss.item()
                    if current_activation > best_activation:
                        best_activation = current_activation
                        best_img = processed_img.clone()
                
                # Update progress bar every 10 steps
                if step % 10 == 0:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'act': current_activation
                    })
            
            # If we used whitening, we need to unwhiten the result
            if self.use_decorrelation and best_img is not None:
                # Normalize to [0,1]
                normalized_img = best_img / 255.0
                
                # Apply whitening
                whitened_img = apply_whitening(normalized_img, self.whitening_matrix, self.mean_color)
                
                # Apply unwhitening to get back to normal color space
                unwhitened_img = apply_unwhitening(whitened_img, self.unwhitening_matrix, self.mean_color)
                
                # Ensure values are in [0,1]
                unwhitened_img = torch.clamp(unwhitened_img, 0, 1)
                
                # Convert to numpy
                result = unwhitened_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            else:
                # Just normalize the best image
                result = (best_img.detach().cpu().squeeze().permute(1, 2, 0) / 255.0).numpy()
            
            print(f"{progress_desc} - Final activation: {best_activation:.4f}")
            return result, best_activation
            
        finally:
            # Clean up all hooks
            self._remove_hooks()


def visualize_features_batch(model, layer_name, color_matrices_path=None, num_features=None, 
                            batch_size=8, num_batches=1, output_dir="decorrelated_visualizations"):
    """
    Visualize features in batches and save as separate images
    
    Args:
        model: The base model
        layer_name: Name of the layer to visualize
        color_matrices_path: Path to color matrices file (optional)
        num_features: Number of features to visualize (if None, visualize all)
        batch_size: Number of features to include in each image
        num_batches: Number of batch images to generate
        output_dir: Directory to save visualizations
    """
    # Initialize the visualizer
    visualizer = DecorrelatedFeatureVisualizer(model, color_matrices_path)
    
    # Determine total number of features to visualize
    total_features = get_num_channels(model, layer_name)
    if total_features is None:
        raise ValueError(f"Could not determine number of channels in layer {layer_name}")
    
    if num_features is not None:
        total_features = min(total_features, num_features)
    
    print(f"Visualizing {total_features} features in layer {layer_name}")
    print(f"Will create {num_batches} images with {batch_size} features each")
    print(f"Color decorrelation: {'Enabled' if visualizer.use_decorrelation else 'Disabled'}")
    
    # Calculate total features to process based on batch_size and num_batches
    features_to_process = min(total_features, batch_size * num_batches)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to directory: {os.path.abspath(output_dir)}")
    
    # Process features in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, features_to_process)
        
        if start_idx >= features_to_process:
            break
            
        print(f"\nProcessing batch {batch_idx+1}/{num_batches} (features {start_idx}-{end_idx-1})")
        
        # Calculate grid dimensions for this batch
        batch_features = end_idx - start_idx
        grid_size = int(np.ceil(np.sqrt(batch_features)))
        
        try:
            # Create figure for this batch
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
            title_suffix = " (Decorrelated)" if visualizer.use_decorrelation else " (Standard)"
            fig.suptitle(f'Features in {layer_name}{title_suffix} (Batch {batch_idx+1}/{num_batches})', fontsize=16)
            
            # Flatten axes for easier indexing
            axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            # Visualize each feature in this batch
            for i, feature_idx in enumerate(range(start_idx, end_idx)):
                # Calculate overall progress
                feature_num = batch_idx * batch_size + i
                progress_percent = (feature_num / features_to_process) * 100
                
                # Create batch info for progress tracking
                batch_info = (batch_idx, batch_size, num_batches, i, features_to_process)
                
                try:
                    # Visualize the feature
                    vis, activation = visualizer.visualize_feature(
                        layer_name, 
                        feature_idx,
                        batch_info=batch_info
                    )
                    
                    axes_flat[i].imshow(vis)
                    axes_flat[i].set_title(f'Feature {feature_idx} (Act: {activation:.2f})')
                    axes_flat[i].axis('off')
                except Exception as e:
                    print(f"Error visualizing feature {feature_idx}: {str(e)}")
                    # Fill with a blank image
                    axes_flat[i].imshow(np.zeros((64, 64, 3)))
                    axes_flat[i].set_title(f'Feature {feature_idx} (Error)')
                    axes_flat[i].axis('off')
            
            # Remove empty subplots
            for idx in range(batch_features, len(axes_flat)):
                fig.delaxes(axes_flat[idx])
            
            plt.tight_layout()
            
            # Save the figure with full path
            filename_suffix = "_decorrelated" if visualizer.use_decorrelation else "_standard"
            output_path = os.path.join(output_dir, f'features_{layer_name}{filename_suffix}_batch{batch_idx+1}.png')
            plt.savefig(output_path, dpi=150)
            print(f"Saved batch image to: {output_path}")
            
            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}: {str(e)}")
        
    print(f"Completed visualization of {features_to_process} features")


def main():
    parser = argparse.ArgumentParser(description='Visualize features with color decorrelation')
    parser.add_argument('--layer_name', type=str, default='conv4a',
                        help='Name of the layer to visualize')
    parser.add_argument('--color_matrices', type=str, default=None,
                        help='Path to color matrices file (if not provided, no decorrelation will be used)')
    parser.add_argument('--compute_matrices', action='store_true',
                        help='Compute color matrices from environment observations')
    parser.add_argument('--env_name', type=str, default='procgen:procgen-heist-v0',
                        help='Environment to collect observations from (if computing matrices)')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of observations to collect (if computing matrices)')
    parser.add_argument('--num_features', type=int, default=None,
                        help='Number of features to visualize (if None, visualize all)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of features per batch image')
    parser.add_argument('--num_batches', type=int, default=1,
                        help='Number of batch images to generate')
    parser.add_argument('--output_dir', type=str, default='decorrelated_visualizations',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_interpretable_model()
    model.to(device)
    model.eval()
    
    # Compute color matrices if requested
    if args.compute_matrices:
        print(f"Computing color matrices from {args.env_name}...")
        matrices_path = args.color_matrices or "color_matrices.pt"
        
        # Collect dataset
        dataset = collect_dataset_from_env(args.env_name, args.num_samples)
        dataset = dataset.to(device)
        
        # Compute matrices
        compute_color_correlation_matrix(dataset, matrices_path)
        
        # Use the computed matrices
        args.color_matrices = matrices_path
    
    # Visualize features
    visualize_features_batch(
        model=model,
        layer_name=args.layer_name,
        color_matrices_path=args.color_matrices,
        num_features=args.num_features,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 