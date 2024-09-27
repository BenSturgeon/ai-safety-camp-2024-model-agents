# sae.py
# %%

import os
import sys
import glob
import random
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass
from tqdm.auto import tqdm
import wandb
import einops
import math

# Import your heist environment module
sys.path.append('../')  # Adjust the path if necessary to import your modules
from src.utils import helpers, heist

# Set device
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# Ordered layer names
ordered_layer_names = {
    1: 'conv1a',
    2: 'pool1',
    3: 'conv2a',
    4: 'conv2b',
    5: 'pool2',
    6: 'conv3a',
    7: 'pool3',
    8: 'conv4a',
    9: 'pool4',
    10: 'fc1',
    11: 'fc2',
    12: 'fc3',
    13: 'value_fc',
    14: 'dropout_conv',
    15: 'dropout_fc'
}

# Define layer types
layer_types = {
    'conv1a': 'conv',
    'pool1': 'pool',
    'conv2a': 'conv',
    'conv2b': 'conv',
    'pool2': 'pool',
    'conv3a': 'conv',
    'pool3': 'pool',
    'conv4a': 'conv',
    'pool4': 'pool',
    'fc1': 'fc',
    'fc2': 'fc',
    'fc3': 'fc',
    'value_fc': 'fc',
    'dropout_conv': 'dropout_conv',
    'dropout_fc': 'dropout_fc'
}

# ModelActivations class modified for batch processing
class ModelActivations:
    def __init__(self, model, layer_paths):
        self.activations = {}
        self.model = model
        self.hooks = []
        self.layer_paths = layer_paths
        self.register_hooks()
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self):
        for path in self.layer_paths:
            elements = path.split('.')
            module = self.model
            for element in elements:
                if '[' in element and ']' in element:
                    base, idx = element.split('[')
                    idx = int(idx[:-1])
                    module = getattr(module, base)[idx]
                else:
                    module = getattr(module, element)
            hook = module.register_forward_hook(self.get_activation(path.replace('.', '_')))
            self.hooks.append(hook)
    
    def run_with_cache(self, inputs):
        self.activations = {}
        # Check if inputs have the expected shape (batch_size, 3, 64, 64)
        if inputs.shape[1:] != (3, 64, 64):
            # If not, rearrange the inputs to the expected shape
            inputs = einops.rearrange(inputs, "b h c w -> b w c h ").to(device)
        inputs = inputs.to(next(self.model.parameters()).to(device))
        outputs = self.model(inputs)
        return outputs, self.activations

def jump_relu(x, jump_value=0.1):
    return t.where(x > 0, x + jump_value, t.zeros_like(x))

def topk_activation(x, k):
    """
    Applies Top-k activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features).
        k (int): Number of top activations to keep per sample.

    Returns:
        torch.Tensor: Tensor with only the top k activations per sample kept.
    """
    # Get the top k values and their indices along dimension 1
    values, indices = t.topk(x, k=k, dim=1)
    # Create a mask of zeros and scatter ones at the indices of top k values
    mask = t.zeros_like(x)
    mask.scatter_(1, indices, 1)
    # Multiply the input by the mask to keep only top k activations
    return x * mask

# SAE configuration and class
@dataclass
class SAEConfig:
    d_in: int = None  # Input dimension, to be set based on layer activations
    d_hidden: int = 128  # Hidden layer dimension
    l1_coeff: float = 0.1
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    jump_value: float = 0.1  # Parameter for Jump ReLU

class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder weights and biases
        self.W_enc = nn.Parameter(t.empty(self.cfg.d_in, self.cfg.d_hidden))
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity='relu')
        self.b_enc = nn.Parameter(t.zeros(self.cfg.d_hidden))

        # Decoder weights and biases
        if self.cfg.tied_weights:
            self.W_dec = self.W_enc.t()
        else:
            self.W_dec = nn.Parameter(t.empty(self.cfg.d_hidden, self.cfg.d_in))
            nn.init.kaiming_uniform_(self.W_dec, nonlinearity='relu')
        self.b_dec = nn.Parameter(t.zeros(self.cfg.d_in))

        self.to(device)

    def forward(self, h):
        # Encoder
        z = t.matmul(h, self.W_enc) + self.b_enc
        acts = jump_relu(z, jump_value=self.cfg.jump_value)

        # Decoder
        h_reconstructed = t.matmul(acts, self.W_dec) + self.b_dec  # [batch_size, d_in]

        # Loss components
        L_reconstruction = F.mse_loss(h_reconstructed, h, reduction='mean')
        L_sparsity = acts.abs().mean()
        loss = L_reconstruction + self.cfg.l1_coeff * L_sparsity

        return loss, L_reconstruction, L_sparsity, acts, h_reconstructed

# Function to generate batch activations in parallel
def generate_batch_activations_parallel(model, model_activations, layer_number, batch_size=32, num_envs=8, episode_length=150):
    """
    Generate activations from multiple environments running in parallel.
    """
    # Create vectorized environments
    venv = heist.create_venv(num=num_envs, num_levels=0, start_level=random.randint(1, 100000))

    activation_buffer = []
    obs_buffer = []
    activation_shape = None  # To be set later

    obs = venv.reset()
    dones = [False] * num_envs
    steps = 0

    while len(activation_buffer) * num_envs < batch_size:
        # Convert observations to tensors and move to GPU
        observations = t.tensor(obs, dtype=t.float32).to(device)  # Shape: [num_envs, height, width, channels]

        # Run the model and capture activations
        with t.no_grad():
            outputs, activations = model_activations.run_with_cache(observations)

        # Extract activations from the specified layer for all environments
        layer_name = ordered_layer_names[layer_number]
        layer_activation = activations[layer_name.replace('.', '_')]

        if activation_shape is None:
            activation_shape = layer_activation.shape[1:]  # Exclude batch dimension

        # Flatten activations and append to the buffer
        batch_layer_activations = layer_activation.reshape(layer_activation.size(0), -1)
        activation_buffer.append(batch_layer_activations)
        obs_buffer.append(observations)

        # Get actions from the model outputs
        logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        probabilities = t.softmax(logits, dim=-1)
        actions = t.multinomial(probabilities, num_samples=1).squeeze(1).cpu().numpy()

        # Step environments
        obs, rewards, dones_env, infos = venv.step(actions)
        dones = [d or (steps > episode_length) for d in dones_env]
        steps += 1

    venv.close()

    # Concatenate buffers
    activations_tensor = t.cat(activation_buffer, dim=0)[:batch_size]
    observations_tensor = t.cat(obs_buffer, dim=0)[:batch_size]

    return activations_tensor, observations_tensor, activation_shape



# Helper function to find the latest checkpoint
def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'sae_checkpoint_step_*.pt'))
    if not checkpoint_files:
        return None, None
    # Extract step numbers and find the maximum
    steps = [int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]) for f in checkpoint_files]
    latest_step = max(steps)
    latest_checkpoint = os.path.join(checkpoint_dir, f'sae_checkpoint_step_{latest_step}.pt')
    return latest_checkpoint, latest_step

# Function to assign hyperparameters based on layer type
def get_layer_hyperparameters(layer_name, layer_types):
    layer_type = layer_types.get(layer_name, 'other')
    if layer_type == 'conv':
        return {
            'd_hidden': 512,     # Example value for conv layers
            'l1_coeff': 0.0001     # Example value for conv layers
        }
    elif layer_type == 'fc':
        return {
            'd_hidden': 2048,    # Example value for fully connected layers
            'l1_coeff': 0.0001      # Example value for fully connected layers
        }
    else:
        # Default hyperparameters for other layer types (e.g., pooling, dropout)
        return {
            'd_hidden': 256,     # Example default value
            'l1_coeff': 0.0001    # Example default value
        }

def train_sae(
    sae,
    model,
    model_activations,
    layer_number,
    layer_name,
    batch_size=128,
    steps=200,
    lr=1e-3,
    num_envs=64,
    episode_length=150,
    log_freq=10,
    checkpoint_dir='checkpoints',
    stats_dir='global_stats',
    wandb_project="SAE_training",
):
    # Load global statistics
    stats_path = os.path.join(stats_dir, f'layer_{layer_number}_{layer_name}_stats.pt')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Global stats file not found for layer {layer_number}: {stats_path}")
    stats = t.load(stats_path, map_location=device)
    global_mean = stats['mean'].to(device)
    global_std = stats['std'].to(device)
    print(global_mean, global_std)

    # Unique identifier for each layer to manage separate checkpoints and wandb runs
    layer_identifier = f'layer_{layer_number}_{layer_name}'

    # Define a separate directory for each layer's checkpoints
    layer_checkpoint_dir = os.path.join(checkpoint_dir, layer_identifier)
    os.makedirs(layer_checkpoint_dir, exist_ok=True)

    # Check for existing checkpoints
    latest_checkpoint, latest_step = find_latest_checkpoint(layer_checkpoint_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = t.load(latest_checkpoint, map_location=device)
        sae.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(sae.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        # Initialize wandb with resume
        wandb.init(project=wandb_project, resume="allow", config={
            "batch_size": batch_size,
            "steps": steps,
            "lr": lr,
            "num_envs": num_envs,
            "episode_length": episode_length,
            "layer_number": layer_number,
            "resume_step": latest_step,
            "layer_identifier": layer_identifier
        }, name=f"SAE_{layer_identifier}")
    else:
        # Initialize optimizer
        optimizer = optim.Adam(sae.parameters(), lr=lr)
        # Initialize wandb
        wandb.init(project=wandb_project, config={
            "batch_size": batch_size,
            "steps": steps,
            "lr": lr,
            "num_envs": num_envs,
            "episode_length": episode_length,
            "layer_number": layer_number,
            "layer_identifier": layer_identifier
        }, name=f"SAE_{layer_identifier}")
        start_step = 0

    # Get the module at the specified layer
    def get_module_by_path(model, layer_path):
        elements = layer_path.split('.')
        module = model
        for element in elements:
            if '[' in element and ']' in element:
                base, idx = element.split('[')
                idx = int(idx[:-1])
                module = getattr(module, base)[idx]
            else:
                module = getattr(module, element)
        return module

    module_at_layer = get_module_by_path(model, layer_name)

    progress_bar = tqdm(range(start_step, steps), desc=f"Training {layer_identifier}")
    loss_history = []
    L_reconstruction_history = []
    L_sparsity_history = []
    variance_explained_history = []
    logits_diff_history = []

    model.eval()
    for step in progress_bar:
        try:
            # Generate batch activations and observations (already on GPU)
            activations, observations, activation_shape = generate_batch_activations_parallel(
                model, model_activations, layer_number,
                batch_size=batch_size, num_envs=num_envs, episode_length=episode_length
            )
            # Normalize using global statistics
            batch_normalized = (activations - global_mean) / global_std

        except Exception as e:
            print(f"Error during activation generation: {e}")
            continue  # Skip this iteration and proceed

        optimizer.zero_grad()
        loss, L_reconstruction, L_sparsity, acts, h_reconstructed = sae(batch_normalized)
        # nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)

        # Reshape h_reconstructed back to original activation shape
        batch_size_curr = h_reconstructed.size(0)
        h_reconstructed_reshaped = h_reconstructed.view(batch_size_curr, *activation_shape)

        # Prepare observations for the model
        observations_prepared = einops.rearrange(observations, "b h w c -> b c h w")

        # Compute original logits
        with t.no_grad():
            # Ensure model is in evaluation mode
            model.eval()
            # Run the model to get original logits
            outputs_original = model(observations_prepared)
            logits_original = outputs_original[0].logits if isinstance(outputs_original, tuple) else outputs_original.logits

        # Compute logits with reconstructed activations
        def replace_activation_hook(module, input, output):
            return h_reconstructed_reshaped

        hook_handle = module_at_layer.register_forward_hook(replace_activation_hook)
        # Run the model to get reconstructed logits
        outputs_reconstructed = model(observations_prepared)
        logits_reconstructed = outputs_reconstructed[0].logits if isinstance(outputs_reconstructed, tuple) else outputs_reconstructed.logits
        # Remove the hook
        hook_handle.remove()

        # Compute the difference in logits
        logits_diff = F.mse_loss(logits_reconstructed, logits_original.detach(), reduction='mean')

        # Add the logits difference to the total loss
        total_loss = loss + logits_diff

        # Backpropagate
        total_loss.backward()
        optimizer.step()

        with t.no_grad():
            # Compute variance explained
            numerator = t.sum((activations - h_reconstructed) ** 2, dim=1)
            denominator = t.sum(activations ** 2, dim=1)
            variance_explained = 1 - numerator / (denominator + 1e-8)
            variance_explained = variance_explained.mean().item()
            acts_mean = acts.mean().item()
            acts_min = acts.min().item()
            acts_max = acts.max().item()
        if step % log_freq == 0 or step == steps - 1:
            progress_bar.set_postfix({
                'Loss': loss.item(),
                'Reconstruction': L_reconstruction.item(),
                'Sparsity': L_sparsity.item(),
                'Logits Diff': logits_diff.item()
            })
            loss_history.append(loss.item())
            L_reconstruction_history.append(L_reconstruction.item())
            L_sparsity_history.append(L_sparsity.item())
            variance_explained_history.append(variance_explained)
            logits_diff_history.append(logits_diff.item())

            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            # Create a figure for activation comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.set_title('Original Activation')
            ax1.plot(activations[0].cpu().numpy())

            ax2.set_title('Reconstructed Activation')
            ax2.plot(h_reconstructed[0].detach().cpu().numpy())

            # Log metrics to wandb
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "total_loss": total_loss.item(),
                "reconstruction_loss": L_reconstruction.item(),
                "sparsity_loss": L_sparsity.item(),
                "logits_diff": logits_diff.item(),
                "acts_mean": acts_mean,
                "acts_min": acts_min,
                "acts_max": acts_max,
                "variance_explained": variance_explained,
                "activation_comparison": wandb.Image(fig)
            }, step=step)

            plt.close(fig)
        # Save checkpoint every 100 steps
        if (step + 1) % 100 == 0 or step == steps - 1:
            checkpoint_path = os.path.join(layer_checkpoint_dir, f'sae_checkpoint_step_{step+1}.pt')
            t.save({
                'step': step,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            wandb.save(checkpoint_path)  # Save checkpoint to wandb

    wandb.finish()
    return loss_history, L_reconstruction_history, L_sparsity_history, variance_explained_history



# Load your model
def load_interpretable_model(model_path="../model_interpretable.pt"):
    import gym
    from src.interpretable_impala import CustomCNN  # Import your model class
    env_name = "procgen:procgen-heist-v0"
    env = gym.make(env_name, start_level=100, num_levels=200, render_mode="rgb_array", distribution_mode="easy")
    observation_space = env.observation_space
    action_space = env.action_space.n
    model = CustomCNN(observation_space, action_space)
    model.load_from_file(model_path, device=device)
    return model

# Function to compute and save global statistics for a layer
def compute_and_save_global_stats(
    model,
    layer_number,
    layer_name,
    num_samples=10000,
    batch_size=128,
    num_envs=64,
    save_dir='global_stats'
):
    """
    Computes and saves the global mean and std for a given layer's activations.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Define layer paths for activation capture
    layer_paths = [layer_name]
    # Initialize ModelActivations
    model_activations = ModelActivations(model, layer_paths=layer_paths)
    activation_samples = []
    total_samples = 0
    with tqdm(total=num_samples, desc=f'Computing global stats for {layer_name}') as pbar:
        while total_samples < num_samples:
            activations, _, _ = generate_batch_activations_parallel(
                model, model_activations, layer_number, batch_size=batch_size, num_envs=num_envs
            )
            activation_samples.append(activations)
            total_samples += activations.size(0)
            pbar.update(activations.size(0))
    all_activations = t.cat(activation_samples, dim=0)[:num_samples]
    global_mean = all_activations.mean(dim=0, keepdim=True)
    global_std = all_activations.std(dim=0, keepdim=True) + 1e-8
    # Save to disk
    t.save({'mean': global_mean, 'std': global_std},
           os.path.join(save_dir, f'layer_{layer_number}_{layer_name}_stats.pt'))
    # Clear hooks
    model_activations.clear_hooks()

# Function to compute global stats for all layers
def compute_global_stats_for_all_layers(
    model,
    ordered_layer_names,
    num_samples_per_layer=10000,
    batch_size=128,
    num_envs=128,
    save_dir='global_stats'
):
    for layer_number, layer_name in ordered_layer_names.items():
        compute_and_save_global_stats(
            model,
            layer_number,
            layer_name,
            num_samples=num_samples_per_layer,
            batch_size=batch_size,
            num_envs=num_envs,
            save_dir=save_dir
        )

# Function to train SAEs for all layers
def train_all_layers(
    model,
    ordered_layer_names,
    layer_types,
    checkpoint_dir='checkpoints',
    stats_dir='global_stats',
    wandb_project="SAE_training",
    steps_per_layer=1000,
    batch_size=64,
    lr=1e-3,
    num_envs=128,
    episode_length=150,
    log_freq=10,
):
    for layer_number, layer_name in ordered_layer_names.items():
        print(f"\n=== Training SAE for Layer {layer_number}: {layer_name} ===")
        
        # Get hyperparameters based on layer type
        hyperparams = get_layer_hyperparameters(layer_name, layer_types)
        d_hidden = hyperparams['d_hidden']
        l1_coeff = hyperparams['l1_coeff']

        # Define layer paths for activation capture
        layer_paths = [layer_name]  # Capture only the target layer

        # Initialize ModelActivations
        model_activations = ModelActivations(model, layer_paths=layer_paths)

        # Generate a sample activation to determine input dimension
        sample_activations, _, _ = generate_batch_activations_parallel(
            model, model_activations, layer_number, batch_size=1
        )
        d_in = sample_activations.shape[-1]  # Input dimension for SAE

        # Configure SAE
        sae_cfg = SAEConfig(
            d_in=d_in,
            d_hidden=d_hidden,
            l1_coeff=l1_coeff,
            tied_weights=False     # Adjust if needed
        )

        # Initialize SAE
        sae_model = SAE(sae_cfg)

        # Train SAE
        loss_history, L_reconstruction_history, L_sparsity_history, variance_explained_history = train_sae(
            sae=sae_model,
            model=model,
            model_activations=model_activations,
            layer_number=layer_number,
            layer_name=layer_name,
            batch_size=batch_size,
            steps=steps_per_layer,
            lr=lr,
            num_envs=num_envs,
            episode_length=episode_length,
            log_freq=log_freq,
            checkpoint_dir=checkpoint_dir,
            stats_dir=stats_dir,
            wandb_project=wandb_project,
        )

        # Optional: Plot training losses for the current layer
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(loss_history)
        plt.title(f'Layer {layer_number} Total Loss')

        plt.subplot(1, 3, 2)
        plt.plot(L_reconstruction_history)
        plt.title(f'Layer {layer_number} Reconstruction Loss')

        plt.subplot(1, 3, 3)
        plt.plot(L_sparsity_history)
        plt.title(f'Layer {layer_number} Sparsity Loss')

        plt.tight_layout()
        plt.show()

        # Clear hooks to prevent accumulation
        model_activations.clear_hooks()



# %%
