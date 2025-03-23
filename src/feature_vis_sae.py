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
import re

# Import SAE-related modules
from sae_cnn import ConvSAE
from extract_sae_features import replace_layer_with_sae
from feature_vis_impala import (
    total_variation, 
    jitter, 
    random_scale, 
    random_rotate, 
    apply_color_correlation,
    load_color_correlation_matrix, 
    get_num_channels
)

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

def load_sae_from_checkpoint(checkpoint_path, device=None):
    """
    Load a SAE model from a checkpoint file
    
    Args:
        checkpoint_path: Path to the SAE checkpoint
        device: The device to load the model onto
        
    Returns:
        sae: The loaded SAE model
        layer_number: The layer number the SAE was trained on (extracted from checkpoint path)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading SAE from checkpoint: {checkpoint_path}")
    
    # Extract layer information from checkpoint path if possible
    layer_match = re.search(r'layer_(\d+)_(\w+)', checkpoint_path)
    layer_number = None
    if layer_match:
        layer_number = int(layer_match.group(1))
        layer_name = layer_match.group(2)
        print(f"Detected layer: {layer_name} (#{layer_number})")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters
    state_dict = checkpoint["model_state_dict"]
    
    # Print state_dict keys to understand structure
    print("State dict keys:", state_dict.keys())
    
    # For ConvSAE, we need to determine the in_channels and hidden_channels
    if 'conv_enc.weight' in state_dict:
        # This is a ConvSAE
        conv_enc_weight = state_dict["conv_enc.weight"]
        in_channels = conv_enc_weight.shape[1]
        hidden_channels = conv_enc_weight.shape[0]
        
        print(f"ConvSAE detected with conv_enc.weight shape: {conv_enc_weight.shape}")
        print(f"in_channels: {in_channels}, hidden_channels: {hidden_channels}")
        
        # Create the SAE with the correct dimensions
        sae = ConvSAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            l1_coeff=1e-5  # Default value, doesn't matter for inference
        )
        
        print(f"Created ConvSAE with in_channels={in_channels}, hidden_channels={hidden_channels}")
    else:
        raise ValueError("Unsupported SAE type in checkpoint or incorrect checkpoint format")
    
    # Load the state dict into the model
    sae.load_state_dict(state_dict)
    sae.eval()
    sae.to(device)
    
    # Print model structure
    print("\nSAE Model Structure:")
    for name, param in sae.named_parameters():
        print(f"  {name}: {param.shape}")
    
    return sae


class SAEFeatureVisualizer:
    def __init__(self, model, color_correlation_path=None):
        """
        Initialize the SAE feature visualizer
        
        Args:
            model: The base model 
            color_correlation_path: Path to color correlation matrix
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.activations = {}
        self.sae_activations = {}
        self.hooks = []
        self.sae = None
        self.sae_hook_handle = None
        
        # Initialize color correlation - either load from file or use identity
        if color_correlation_path and os.path.exists(color_correlation_path):
            self.color_correlation = load_color_correlation_matrix(color_correlation_path, self.device)
            print(f"Loaded color correlation matrix from {color_correlation_path}")
        else:
            self.color_correlation = torch.eye(3, device=self.device)
            print("Using default identity color correlation matrix")

    def _register_sae_hooks(self, layer_number):
        """
        Register hooks to capture SAE activations
        
        Args:
            layer_number: The layer number where the SAE is attached
        """
        if self.sae is None:
            raise ValueError("No SAE model loaded. Call load_sae() first.")
        
        # Clear existing hooks
        self._remove_hooks()
        
        # Get the layer name
        layer_name = ordered_layer_names[layer_number]
        
        # Define a hook to capture the SAE encoder activations
        def sae_encoder_hook(module, input, output):
            _, _, acts, _ = self.sae(output)
            self.sae_activations['encoder'] = acts
            return output  # Return original output so model functions normally
            
        # Find the module and register the hook
        for name, layer in self.model.named_modules():
            if name == layer_name:
                hook = layer.register_forward_hook(sae_encoder_hook)
                self.hooks.append(hook)
                print(f"Registered hook on layer: {name}")
                break

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.sae_activations = {}
        
        # Also remove SAE hook if any
        if self.sae_hook_handle is not None:
            self.sae_hook_handle.remove()
            self.sae_hook_handle = None

    def load_sae(self, sae, layer_number):
        """
        Load an SAE model and integrate it with the base model
        
        Args:
            sae: The SAE model
            layer_number: The layer number where the SAE should be attached
        """
        self.sae = sae
        self.sae.eval()
        self._register_sae_hooks(layer_number)
        
    def visualize_sae_feature(self, layer_number, feature_idx, 
                         num_steps=2560, lr=0.08, tv_weight=1e-3, 
                         l2_weight=1e-3, jitter_amount=8,
                         batch_info=None):
        """
        Visualize what maximally activates a specific SAE feature
        
        Args:
            layer_number: Layer number where the SAE is attached
            feature_idx: Index of the SAE feature to visualize
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
        if self.sae is None:
            raise ValueError("No SAE model loaded. Call load_sae() first.")
        
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
        print(f"Layer number: {layer_number}, Layer name: {ordered_layer_names[layer_number]}")
        print(f"SAE type: {type(self.sae).__name__}")
        
        # Register temporary hook to replace the layer with SAE during visualization
        self.sae_hook_handle = replace_layer_with_sae(self.model, s.sae, layer_number)
        
        # Also register hooks to capture the SAE encoder activations specifically
        def sae_feature_hook(module, input, output):
            # For ConvSAE, the 3rd return value (index 2) contains the activations
            if isinstance(output, tuple) and len(output) >= 3:
                self.sae_activations['target_feature'] = output[2][:, feature_idx]
                print(f"SAE feature hook captured output tuple of length {len(output)}")
                print(f"Activation shape: {output[2].shape}")
            else:
                print(f"SAE feature hook received unexpected output type: {type(output)}")
                if not isinstance(output, tuple):
                    print(f"Output shape: {output.shape}")
        
        # Register the sae_feature_hook on the SAE itself
        hook_sae = self.sae.register_forward_hook(sae_feature_hook)
        self.hooks.append(hook_sae)
        
        # Hook directly to the SAE encoder's output
        hook = self.sae.conv_enc.register_forward_hook(
            lambda module, input, output: 
            self.sae_activations.update({'pre_relu': output})
        )
        self.hooks.append(hook)
        
        # Also hook after ReLU to get the actual activations
        hook2 = self.sae.register_forward_hook(
            lambda module, input, output: 
            self.sae_activations.update({'post_relu': output[2]})  # index 2 contains activations
        )
        self.hooks.append(hook2)
        
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
                
                # Apply color correlation
                processed_img = apply_color_correlation(processed_img, self.color_correlation)
                
                # Crop padding
                processed_img = processed_img[:, :, 4:-4, 4:-4]
                
                # Ensure values are in [0, 255] before normalizing
                processed_img.data.clamp_(0, 255)
                
                # Normalize to [0,1] for model input
                processed_img = processed_img / 255.0
                
                # Forward pass - this will trigger our hooks
                _ = self.model(processed_img)
                
                # Get the activation for the target SAE feature
                if 'post_relu' in self.sae_activations:
                    # Extract the specific feature activation
                    feature_activation = self.sae_activations['post_relu'][0, feature_idx]
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
                        best_img = input_img[:, :, 4:-4, 4:-4].clone()
                
                # Update progress bar every 10 steps
                if step % 10 == 0:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'act': current_activation
                    })
            
            # Return the image that achieved the highest activation
            result = best_img.detach().cpu().squeeze().permute(1, 2, 0).numpy() / 255.0
            print(f"{progress_desc} - Final activation: {best_activation:.4f}")
            return result, best_activation
            
        finally:
            # Clean up all hooks
            self._remove_hooks()
            if self.sae_hook_handle is not None:
                self.sae_hook_handle.remove()
                self.sae_hook_handle = None


def visualize_sae_features(model, sae_checkpoint_path, layer_number=None, num_features=None, color_correlation_path=None, batch_size=8, num_batches=16, use_color_correlation=True):
    """
    Visualize SAE features in batches and save as separate images
    
    Args:
        model: The base model
        sae_checkpoint_path: Path to the SAE checkpoint
        layer_number: Layer number where the SAE should be attached (optional, extracted from checkpoint path if not provided)
        num_features: Number of SAE features to visualize (if None, visualize all)
        color_correlation_path: Path to color correlation matrix (optional)
        batch_size: Number of features to include in each image
        num_batches: Number of batch images to generate
        use_color_correlation: Whether to apply color correlation (default: True)
    """
    # Load the SAE
    device = next(model.parameters()).device
    sae, checkpoint_layer_number = load_sae_from_checkpoint(sae_checkpoint_path, device)
    
    # Use layer number from checkpoint if not explicitly provided
    if layer_number is None:
        layer_number = checkpoint_layer_number
        if layer_number is None:
            raise ValueError("Layer number could not be extracted from checkpoint path and was not provided")
    
    # Get layer name
    layer_name = ordered_layer_names[layer_number]
    
    # Initialize the visualizer
    visualizer = SAEFeatureVisualizer(model, color_correlation_path if use_color_correlation else None)
    visualizer.load_sae(sae, layer_number)
    
    # If not using color correlation, override with identity matrix
    if not use_color_correlation:
        visualizer.color_correlation = torch.eye(3, device=device)
        print("Using identity color correlation matrix (color correlation disabled)")
    
    # Determine total number of features to visualize
    if hasattr(sae, 'hidden_channels'):
        total_features = sae.hidden_channels
    else:
        total_features = sae.cfg.d_hidden
    
    if num_features is not None:
        total_features = min(total_features, num_features)
    
    print(f"Visualizing {total_features} SAE features for layer {layer_name} (#{layer_number})")
    print(f"Will create {num_batches} images with {batch_size} features each")
    print(f"Color correlation: {'Enabled' if use_color_correlation else 'Disabled'}")
    
    # Calculate total features to process based on batch_size and num_batches
    features_to_process = min(total_features, batch_size * num_batches)
    
    # Create output directory if it doesn't exist
    output_dir = "sae_visualizations"
    if not use_color_correlation:
        output_dir = "sae_visualizations_no_color_corr"
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
            title_suffix = " (No Color Correlation)" if not use_color_correlation else ""
            fig.suptitle(f'SAE Features in {layer_name}{title_suffix} (Batch {batch_idx+1}/{num_batches})', fontsize=16)
            
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
                    vis, activation = visualizer.visualize_sae_feature(
                        layer_number, 
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
            filename_suffix = "_no_color_corr" if not use_color_correlation else ""
            output_path = os.path.join(output_dir, f'sae_filters_{layer_name}{filename_suffix}_batch{batch_idx+1}.png')
            plt.savefig(output_path, dpi=150)
            print(f"Saved batch image to: {output_path}")
            
            # Also save a copy in the current directory for backward compatibility
            plt.savefig(f'sae_filters_{layer_name}{filename_suffix}_batch{batch_idx+1}.png', dpi=150)
            
            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}: {str(e)}")
        
    print(f"Completed visualization of {features_to_process} SAE features")

def visualize_original_layer(model, layer_name, num_features=None, color_correlation_path=None, batch_size=8, num_batches=1):
    """
    Visualize features in an original model layer using the SAE visualization pipeline
    
    Args:
        model: The base model
        layer_name: Name of the layer to visualize (e.g., 'conv4a')
        num_features: Number of features to visualize (if None, visualize all)
        color_correlation_path: Path to color correlation matrix (optional)
        batch_size: Number of features to include in each image
        num_batches: Number of batch images to generate
    """
    # Initialize the visualizer (reusing the SAE visualizer class)
    visualizer = SAEFeatureVisualizer(model, color_correlation_path)
    
    # Register hooks to capture the original model's activations
    visualizer.activations = {}
    
    def capture_activations(module, input, output):
        visualizer.activations[layer_name] = output
        return output
    
    # Find the target layer and register the hook
    target_layer = None
    for name, layer in model.named_modules():
        if name == layer_name:
            target_layer = layer
            hook = layer.register_forward_hook(capture_activations)
            visualizer.hooks.append(hook)
            break
    
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Determine total number of features to visualize
    total_features = get_num_channels(model, layer_name)
    if total_features is None:
        raise ValueError(f"Could not determine number of channels in layer {layer_name}")
    
    if num_features is not None:
        total_features = min(total_features, num_features)
    
    print(f"Visualizing {total_features} features in original layer {layer_name}")
    print(f"Will create {num_batches} images with {batch_size} features each")
    
    # Calculate total features to process based on batch_size and num_batches
    features_to_process = min(total_features, batch_size * num_batches)
    
    # Create output directory if it doesn't exist
    output_dir = "original_layer_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to directory: {os.path.abspath(output_dir)}")
    
    # Define a new method to visualize original layer features
    def visualize_original_feature(feature_idx, num_steps=2560, lr=0.08, tv_weight=1e-3, 
                                  l2_weight=1e-3, jitter_amount=8, batch_info=None):
        """Visualize what maximally activates a specific feature in the original layer"""
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
        
        # Initialize padded random image
        padded_size = 64 + 8  # 4 pixels on each side
        input_img = torch.randint(0, 256, (1, 3, padded_size, padded_size), 
                                device=visualizer.device, 
                                dtype=torch.float32).requires_grad_(True)
        
        print(f"Input image shape: {input_img.shape}, device: {input_img.device}")
        
        optimizer = optim.Adam([input_img], lr=lr)
        
        # Parameters for transformations
        scales = [1.0, 0.975, 1.025, 0.95, 1.05]
        angles = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        
        best_activation = float('-inf')
        best_img = None
        
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
            
            # Apply color correlation
            processed_img = apply_color_correlation(processed_img, visualizer.color_correlation)
            
            # Crop padding
            processed_img = processed_img[:, :, 4:-4, 4:-4]
            
            # Ensure values are in [0, 255] before normalizing
            processed_img.data.clamp_(0, 255)
            
            # Normalize to [0,1] for model input
            processed_img = processed_img / 255.0
            
            # Forward pass - this will trigger our hooks
            _ = model(processed_img)
            
            # Get the activation for the target feature in the original layer
            if layer_name in visualizer.activations:
                # Extract the specific feature activation
                feature_activation = visualizer.activations[layer_name][0, feature_idx]
                activation_loss = -feature_activation.mean()  # negative because we want to maximize
            else:
                activation_loss = torch.tensor(0.0, device=visualizer.device)
            
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
                    best_img = input_img[:, :, 4:-4, 4:-4].clone()
            
            # Update progress bar every 10 steps
            if step % 10 == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'act': current_activation
                })
        
        # Return the image that achieved the highest activation
        result = best_img.detach().cpu().squeeze().permute(1, 2, 0).numpy() / 255.0
        print(f"{progress_desc} - Final activation: {best_activation:.4f}")
        return result, best_activation
    
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
            fig.suptitle(f'Original Layer Features in {layer_name} (Batch {batch_idx+1}/{num_batches})', fontsize=16)
            
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
                    vis, activation = visualize_original_feature(
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
            output_path = os.path.join(output_dir, f'original_filters_{layer_name}_batch{batch_idx+1}.png')
            plt.savefig(output_path, dpi=150)
            print(f"Saved batch image to: {output_path}")
            
            # Also save a copy in the current directory for backward compatibility
            plt.savefig(f'original_filters_{layer_name}_batch{batch_idx+1}.png', dpi=150)
            
            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}: {str(e)}")
    
    # Clean up hooks
    for hook in visualizer.hooks:
        hook.remove()
    
    print(f"Completed visualization of {features_to_process} original features in layer {layer_name}")


if __name__ == "__main__":
    # Setup
    env = gym.make('procgen:procgen-heist-v0')
    model = load_interpretable_model()
    
    # Specify your SAE checkpoint
    sae_checkpoint_path = "checkpoints/sae_checkpoint_step_4500000.pt"
    
    # Visualize SAE features with color correlation
    visualize_sae_features(
        model=model,
        sae_checkpoint_path=sae_checkpoint_path,
        batch_size=8,           # 8 features per image
        num_batches=1,         # Create 1 image
        color_correlation_path="src/color_correlation.pt",  # Optional
        use_color_correlation=True
    )
    
    # Visualize SAE features without color correlation
    visualize_sae_features(
        model=model,
        sae_checkpoint_path=sae_checkpoint_path,
        batch_size=8,           # 8 features per image
        num_batches=1,         # Create 1 image
        color_correlation_path="src/color_correlation.pt",  # Will be ignored
        use_color_correlation=False
    ) 