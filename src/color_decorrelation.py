import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import gym
import argparse

def compute_color_correlation_matrix(dataset, save_path="color_correlation.pt"):
    """
    Compute color correlation matrix from a dataset of images.
    
    Args:
        dataset: Tensor of shape (N, C, H, W) where N is number of images, C is channels (3 for RGB)
        save_path: Where to save the color correlation matrix
        
    Returns:
        color_correlation: The computed color correlation matrix
        whitening_matrix: The whitening matrix (C^(-1/2))
        unwhitening_matrix: The unwhitening matrix (C^(1/2))
    """
    # Flatten spatial dimensions
    flat_data = dataset.reshape(-1, 3)  # Shape: (N*H*W, 3)
    
    # Compute mean color
    mean_color = flat_data.mean(dim=0, keepdim=True)
    
    # Center the data
    centered_colors = flat_data - mean_color
    
    # Compute covariance matrix
    cov_matrix = torch.mm(centered_colors.t(), centered_colors) / (flat_data.size(0) - 1)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Ensure numerical stability
    eigenvalues = torch.clamp(eigenvalues, min=1e-8)
    
    # Compute matrices for whitening and unwhitening
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    inv_sqrt_eigenvalues = 1.0 / sqrt_eigenvalues
    
    # C^(1/2) = U * D^(1/2) * U^T
    unwhitening_matrix = torch.mm(
        eigenvectors * sqrt_eigenvalues.unsqueeze(0),
        eigenvectors.t()
    )
    
    # C^(-1/2) = U * D^(-1/2) * U^T
    whitening_matrix = torch.mm(
        eigenvectors * inv_sqrt_eigenvalues.unsqueeze(0),
        eigenvectors.t()
    )
    
    # Print statistics
    print("\nColor Correlation Matrix (Covariance):")
    print(cov_matrix)
    print("\nEigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
    print("\nWhitening Matrix (C^(-1/2)):")
    print(whitening_matrix)
    print("\nUnwhitening Matrix (C^(1/2)):")
    print(unwhitening_matrix)
    
    # Visualize the matrices
    plt.figure(figsize=(15, 5))
    
    # Plot covariance matrix
    plt.subplot(131)
    plt.imshow(cov_matrix.cpu().numpy(), cmap='RdBu')
    plt.colorbar()
    plt.title('Color Covariance Matrix')
    plt.xticks([0,1,2], ['R', 'G', 'B'])
    plt.yticks([0,1,2], ['R', 'G', 'B'])
    
    # Plot whitening matrix
    plt.subplot(132)
    plt.imshow(whitening_matrix.cpu().numpy(), cmap='RdBu')
    plt.colorbar()
    plt.title('Whitening Matrix (C^(-1/2))')
    plt.xticks([0,1,2], ['R', 'G', 'B'])
    plt.yticks([0,1,2], ['R', 'G', 'B'])
    
    # Plot eigenspectrum
    plt.subplot(133)
    plt.bar(range(3), eigenvalues.cpu().numpy())
    plt.title('Eigenspectrum')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    
    plt.tight_layout()
    plt.savefig('color_matrices_vis.png')
    plt.close()
    
    # Save the matrices
    torch.save({
        'covariance_matrix': cov_matrix,
        'whitening_matrix': whitening_matrix,
        'unwhitening_matrix': unwhitening_matrix,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'mean_color': mean_color.squeeze(),
    }, save_path)
    
    print(f"\nSaved color matrices to {save_path}")
    return cov_matrix, whitening_matrix, unwhitening_matrix, mean_color.squeeze()

def collect_dataset_from_env(env_name='procgen:procgen-heist-v0', num_samples=10000, episode_length=150):
    """
    Collect a dataset of observations from an environment
    
    Args:
        env_name: Name of the environment to collect from
        num_samples: Number of observations to collect
        episode_length: Maximum length of each episode
        
    Returns:
        dataset: Tensor of shape (N, C, H, W)
    """
    venv = gym.make(env_name)
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
    
    return obs_tensor

def apply_whitening(x, whitening_matrix, mean_color=None):
    """
    Apply whitening transformation to an image tensor
    
    Args:
        x: Image tensor of shape (B, C, H, W)
        whitening_matrix: The whitening matrix (C^(-1/2))
        mean_color: Mean color to subtract (optional)
        
    Returns:
        Whitened image tensor of shape (B, C, H, W)
    """
    # Store original shape
    original_shape = x.shape
    
    # Reshape to (B*H*W, C)
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
    
    # Center if mean_color is provided
    if mean_color is not None:
        x_flat = x_flat - mean_color
    
    # Apply whitening
    x_whitened = torch.mm(x_flat, whitening_matrix.t())
    
    # Reshape back to original shape
    x_whitened = x_whitened.reshape(original_shape[0], original_shape[2], original_shape[3], 3).permute(0, 3, 1, 2)
    
    return x_whitened

def apply_unwhitening(x, unwhitening_matrix, mean_color=None):
    """
    Apply unwhitening transformation to an image tensor (reverse of whitening)
    
    Args:
        x: Whitened image tensor of shape (B, C, H, W)
        unwhitening_matrix: The unwhitening matrix (C^(1/2))
        mean_color: Mean color to add back (optional)
        
    Returns:
        Unwhitened image tensor of shape (B, C, H, W)
    """
    # Store original shape
    original_shape = x.shape
    
    # Reshape to (B*H*W, C)
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
    
    # Apply unwhitening
    x_unwhitened = torch.mm(x_flat, unwhitening_matrix.t())
    
    # Add back mean if provided
    if mean_color is not None:
        x_unwhitened = x_unwhitened + mean_color
    
    # Reshape back to original shape
    x_unwhitened = x_unwhitened.reshape(original_shape[0], original_shape[2], original_shape[3], 3).permute(0, 3, 1, 2)
    
    return x_unwhitened

def load_color_matrices(load_path="color_correlation.pt", device=None):
    """
    Load previously computed color matrices
    
    Args:
        load_path: Path to the saved matrices
        device: Device to load the matrices to
        
    Returns:
        covariance_matrix: The color covariance matrix
        whitening_matrix: The whitening matrix (C^(-1/2))
        unwhitening_matrix: The unwhitening matrix (C^(1/2))
        mean_color: The mean color vector
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = torch.load(load_path, map_location=device)
    
    return (
        data['covariance_matrix'].to(device),
        data['whitening_matrix'].to(device),
        data['unwhitening_matrix'].to(device),
        data['mean_color'].to(device)
    )

def visualize_whitening_effect(dataset, whitening_matrix, unwhitening_matrix, mean_color=None, num_samples=5):
    """
    Visualize the effect of whitening and unwhitening on sample images
    
    Args:
        dataset: Tensor of shape (N, C, H, W)
        whitening_matrix: The whitening matrix
        unwhitening_matrix: The unwhitening matrix
        mean_color: Mean color vector
        num_samples: Number of samples to visualize
    """
    # Select random samples
    indices = torch.randperm(dataset.shape[0])[:num_samples]
    samples = dataset[indices]
    
    # Apply whitening
    whitened_samples = apply_whitening(samples, whitening_matrix, mean_color)
    
    # Apply unwhitening to verify reversibility
    reconstructed_samples = apply_unwhitening(whitened_samples, unwhitening_matrix, mean_color)
    
    # Visualize
    plt.figure(figsize=(15, num_samples * 3))
    
    for i in range(num_samples):
        # Original
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(samples[i].permute(1, 2, 0).cpu().numpy() / 255.0)
        plt.title(f'Original {i+1}')
        plt.axis('off')
        
        # Whitened
        plt.subplot(num_samples, 3, i*3 + 2)
        # Scale for visualization since whitened images can have arbitrary range
        whitened_img = whitened_samples[i].permute(1, 2, 0).cpu().numpy()
        whitened_img = (whitened_img - whitened_img.min()) / (whitened_img.max() - whitened_img.min())
        plt.imshow(whitened_img)
        plt.title(f'Whitened {i+1}')
        plt.axis('off')
        
        # Reconstructed
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(reconstructed_samples[i].permute(1, 2, 0).cpu().numpy() / 255.0)
        plt.title(f'Reconstructed {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('whitening_effect.png', dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compute and visualize color decorrelation matrices')
    parser.add_argument('--env_name', type=str, default='procgen:procgen-heist-v0',
                        help='Environment to collect observations from')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of observations to collect')
    parser.add_argument('--save_path', type=str, default='color_matrices.pt',
                        help='Path to save the color matrices')
    parser.add_argument('--visualize_samples', type=int, default=5,
                        help='Number of sample images to visualize whitening effect on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Collect dataset
    print(f"Collecting dataset from {args.env_name}...")
    dataset = collect_dataset_from_env(args.env_name, args.num_samples)
    dataset = dataset.to(device)
    
    # Compute color matrices
    print("Computing color matrices...")
    cov_matrix, whitening_matrix, unwhitening_matrix, mean_color = compute_color_correlation_matrix(
        dataset, args.save_path
    )
    
    # Visualize whitening effect
    print("Visualizing whitening effect...")
    visualize_whitening_effect(
        dataset, whitening_matrix, unwhitening_matrix, mean_color, args.visualize_samples
    )
    
    print("Done!")

if __name__ == "__main__":
    main() 