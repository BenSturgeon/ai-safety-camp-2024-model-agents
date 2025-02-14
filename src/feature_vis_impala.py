# %%
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

def total_variation(x):
    """Calculate total variation of the image to encourage smoothness"""
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

def jitter(img, ox, oy):
    """Randomly shift the image by (ox, oy) pixels"""
    return torch.roll(img, shifts=(ox, oy), dims=(2, 3))

def random_scale(img, scales):
    """Randomly scale the image by one of the given scale factors"""
    scale = np.random.choice(scales)
    if scale == 1.0:
        return img
    h, w = img.shape[2:]
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
    # Pad or crop to original size
    if scale > 1.0:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        scaled = scaled[:, :, start_h:start_h+h, start_w:start_w+w]
    else:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        scaled = F.pad(scaled, (pad_w, pad_w, pad_h, pad_h))
    return scaled

def random_rotate(img, angles):
    """Randomly rotate the image by one of the given angles (in degrees)"""
    angle = np.random.choice(angles)
    if angle == 0:
        return img
    # Convert to radians
    theta = torch.tensor([angle * np.pi / 180], device=img.device, dtype=torch.float32)
    # Create affine transformation matrix
    rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                           [torch.sin(theta), torch.cos(theta), 0]], 
                           device=img.device, 
                           dtype=torch.float32)
    grid = F.affine_grid(rot_mat.unsqueeze(0), img.size(), align_corners=False)
    return F.grid_sample(img, grid, align_corners=False)

def apply_color_correlation(x, color_matrix):
    """Apply a 3x3 color correlation matrix to an image tensor"""
    return torch.einsum('ij,bjhw->bihw', color_matrix, x)

def compute_color_correlation_matrix(num_samples=10000, num_envs=8, episode_length=150, save_path="color_correlation.pt"):
    """
    Compute color correlation matrix from environment observations.
    This helps make feature visualizations look more natural by capturing
    color statistics of the environment.
    
    Args:
        num_samples: Number of observations to collect
        num_envs: Number of parallel environments
        episode_length: Maximum length of each episode
        save_path: Where to save the color correlation matrix
    """
    venv = gym.make('procgen:procgen-heist-v0')
    observations = []
    total_samples = 0
    
    with tqdm(total=num_samples, desc="Collecting observations") as pbar:
        while total_samples < num_samples:
            obs = venv.reset()
            done = False
            steps = 0
            
            while not done and steps < episode_length:
                observations.append(obs)
                total_samples += 1
                pbar.update(1)
                
                # Random actions for exploration
                action = venv.action_space.sample()
                obs, _, done, _ = venv.step(action)
                steps += 1
                
                if total_samples >= num_samples:
                    break
    
    venv.close()
    
    # Convert observations to tensor and reshape
    obs_tensor = torch.tensor(np.array(observations[:num_samples]), dtype=torch.float32)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    # Flatten spatial dimensions
    flat_obs = obs_tensor.reshape(-1, 3)  # Shape: (N*H*W, 3)
    
    # Compute correlation matrix
    mean_color = flat_obs.mean(dim=0, keepdim=True)
    centered_colors = flat_obs - mean_color
    cov_matrix = torch.mm(centered_colors.t(), centered_colors) / (flat_obs.size(0) - 1)
    
    # Get correlation matrix through eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sqrt_eigenvalues = torch.sqrt(eigenvalues + 1e-8)
    color_correlation = torch.mm(
        eigenvectors * sqrt_eigenvalues.unsqueeze(0),
        eigenvectors.t()
    )
    
    # Print statistics
    print("\nColor Correlation Matrix:")
    print(color_correlation)
    print("\nEigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
    
    # Visualize the correlation matrix
    plt.figure(figsize=(10, 4))
    
    # Plot correlation matrix
    plt.subplot(121)
    plt.imshow(color_correlation.cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Color Correlation Matrix')
    plt.xticks([0,1,2], ['R', 'G', 'B'])
    plt.yticks([0,1,2], ['R', 'G', 'B'])
    
    # Plot eigenspectrum
    plt.subplot(122)
    plt.bar(range(3), eigenvalues.cpu().numpy())
    plt.title('Eigenspectrum')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    
    plt.tight_layout()
    plt.savefig('color_correlation_vis.png')
    plt.close()
    
    # Save the matrix
    torch.save({
        'correlation_matrix': color_correlation,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'num_samples': num_samples,
    }, save_path)
    
    print(f"\nSaved color correlation matrix to {save_path}")
    return color_correlation

def load_color_correlation_matrix(load_path="color_correlation.pt", device=None):
    """Load a previously computed color correlation matrix"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = torch.load(load_path, map_location=device)
    return data['correlation_matrix'].to(device)

class FeatureVisualizer:
    def __init__(self, model, color_correlation_path=None):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.activations = {}
        self.hooks = []
        
        # Initialize color correlation - either load from file or use identity
        if color_correlation_path and os.path.exists(color_correlation_path):
            self.color_correlation = load_color_correlation_matrix(color_correlation_path, self.device)
            print(f"Loaded color correlation matrix from {color_correlation_path}")
        else:
            self.color_correlation = torch.eye(3, device=self.device)
            print("Using default identity color correlation matrix")

    def _register_hooks(self, target_layers):
        """Register forward hooks on target layers"""
        self.hooks = []
        for name, layer in self.model.named_modules():
            if name in target_layers:
                hook = layer.register_forward_hook(
                    lambda module, input, output, name=name: 
                    self.activations.update({name: output})
                )
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def update_color_correlation(self, num_samples=10000, save_path="color_correlation.pt"):
        """Update the color correlation matrix based on environment observations"""
        self.color_correlation = compute_color_correlation_matrix(
            num_samples=num_samples, 
            save_path=save_path
        ).to(self.device)

    def visualize_channel(self, target_layer, channel_idx, 
                         num_steps=2560, lr=0.05, tv_weight=1e-3, 
                         l2_weight=1e-3, jitter_amount=8):
        """
        Visualize what maximally activates a specific channel in a target layer.
        Using techniques from the Distill feature visualization article.
        """
        self._register_hooks([target_layer])
        
        # Initialize padded random image directly
        padded_size = 64 + 8  # 4 pixels on each side
        input_img = torch.randint(0, 256, (1, 3, padded_size, padded_size), 
                                device=self.device, 
                                dtype=torch.float32).requires_grad_(True)
        
        optimizer = optim.Adam([input_img], lr=lr)
        
        # Parameters for transformations
        scales = [1.0, 0.975, 1.025, 0.95, 1.05]
        angles = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        
        best_activation = float('-inf')
        best_img = None
        
        try:
            for step in range(num_steps):
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
                    ox, oy = np.random.randint(-4, 5, 2)  # Reduced from 8 to 4
                    processed_img = jitter(processed_img, ox, oy)
                
                # Apply color correlation
                processed_img = apply_color_correlation(processed_img, self.color_correlation)
                
                # Crop padding
                processed_img = processed_img[:, :, 4:-4, 4:-4]  # Changed from 16 to 4
                
                # Ensure values are in [0, 255] before normalizing
                processed_img.data.clamp_(0, 255)
                
                # Normalize to [0,1] for model input
                processed_img = processed_img / 255.0
                
                # Forward pass
                _ = self.model(processed_img)
                
                # Get activation for target layer
                act = self.activations[target_layer]
                if isinstance(act, tuple):
                    act = act[0]
                
                # Extract target channel activation
                channel_activation = act[0, channel_idx, :, :]
                activation_loss = -channel_activation.mean()
                
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
                    
                    # Track best activation (using unpadded region)
                    current_activation = -activation_loss.item()
                    if current_activation > best_activation:
                        best_activation = current_activation
                        best_img = input_img[:, :, 4:-4, 4:-4].clone()  # Changed from 16 to 4
                
                if step % 20 == 0:
                    print(f"Step {step:03d} - Loss: {loss.item():.4f}, Act: {-activation_loss.item():.4f}")
            
            # Return the image that achieved the highest activation
            result = best_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            print(f"Final best activation: {best_activation:.4f}")
            return result
            
        finally:
            self._remove_hooks()

def get_num_channels(model, layer_name):
    """Get number of channels in a layer"""
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer.out_channels
    return None


# Setup
env = gym.make('procgen:procgen-heist-v0')
model = load_interpretable_model()
visualizer = FeatureVisualizer(model)

# Visualize all channels in each layer
layers_to_visualize = ['conv1a', 'conv1b', 'conv2a', 'conv2b', 'conv3a', 'conv3b', 'conv4a', 'conv4b']

for layer_name in layers_to_visualize:
    num_channels = get_num_channels(model, layer_name)
    if num_channels is None:
        print(f"Could not find layer {layer_name}")
        continue
        
    print(f"\nVisualizing {layer_name} - {num_channels} channels")
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
    fig.suptitle(f'All Channels in {layer_name}', fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Visualize each channel
    for channel_idx in range(num_channels):
        print(f"Processing channel {channel_idx}/{num_channels}")
        vis = visualizer.visualize_channel(layer_name, channel_idx)
        
        axes_flat[channel_idx].imshow(vis)
        axes_flat[channel_idx].set_title(f'Ch {channel_idx}')
        axes_flat[channel_idx].axis('off')
    
    # Remove empty subplots
    for idx in range(num_channels, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    plt.tight_layout()
    plt.show()

# %%
