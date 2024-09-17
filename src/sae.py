import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np
import random
import math
from dataclasses import dataclass
import einops
import wandb
import os
import sys
import glob

# Append parent directory to import modules
sys.path.append('../')  # Adjust the path if necessary to import your modules

import notebooks.heist as heist  # Import your heist environment module

# Set device
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

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
# %%
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
        inputs = einops.rearrange(inputs, "b h c w -> b w c h ")
        inputs = inputs.to(next(self.model.parameters()).device)
        outputs = self.model(inputs)
        return outputs, self.activations

# SAE configuration and class
@dataclass
class SAEConfig:
    d_in: int = None  # Input dimension, to be set based on layer activations
    d_hidden: int = 128  # Hidden layer dimension
    l1_coeff: float = 0.1
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False

class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder weights and biases
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(t.empty(self.cfg.d_in, self.cfg.d_hidden))
        )
        self.b_enc = nn.Parameter(t.zeros(self.cfg.d_hidden))

        # Decoder weights and biases
        if self.cfg.tied_weights:
            self.W_dec = self.W_enc.t()
        else:
            self.W_dec = nn.Parameter(
                nn.init.kaiming_uniform_(t.empty(self.cfg.d_hidden, self.cfg.d_in))
            )
        self.b_dec = nn.Parameter(t.zeros(self.cfg.d_in))

        self.to(device)

    def forward(self, h):
        # Encoder
        acts = F.leaky_relu(t.matmul(h, self.W_enc) + self.b_enc)


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
    venv = heist.create_venv(num=num_envs, num_levels=1, start_level=random.randint(1, 100000))

    activation_buffer = []

    obs = venv.reset()
    dones = [False] * num_envs
    steps = 0

    while len(activation_buffer) < batch_size:
        # Convert observations to tensors without permuting
        observations = t.tensor(obs, dtype=t.float32)  # Shape: [batch_size, height, width, channels]

        # Run the model and capture activations
        with t.no_grad():
            outputs, activations = model_activations.run_with_cache(observations)

        # Extract activations from the specified layer for all environments
        layer_name = ordered_layer_names[layer_number]
        layer_activation = activations[layer_name.replace('.', '_')]

        # Flatten activations and append to the buffer
        batch_layer_activations = layer_activation.view(layer_activation.size(0), -1)
        activation_buffer.append(batch_layer_activations.cpu())

        # Get actions from the model outputs
        logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        probabilities = t.softmax(logits, dim=-1)
        actions = t.multinomial(probabilities, num_samples=1).squeeze(1).cpu().numpy()

        # Step environments
        obs, rewards, dones_env, infos = venv.step(actions)
        dones = [d or (steps > episode_length) for d in dones_env]
        steps += 1

    venv.close()

    activations_tensor = t.cat(activation_buffer, dim=0)[:batch_size]

    return activations_tensor

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
            'd_hidden': 124,     # Example value for conv layers
            'l1_coeff': 0.5     # Example value for conv layers
        }
    elif layer_type == 'fc':
        return {
            'd_hidden': 1024,    # Example value for fully connected layers
            'l1_coeff': 0.1      # Example value for fully connected layers
        }
    else:
        # Default hyperparameters for other layer types (e.g., pooling, dropout)
        return {
            'd_hidden': 256,     # Example default value
            'l1_coeff': 0.2     # Example default value
        }

# Training function for SAE with checkpointing and wandb logging
def train_sae(
    sae,
    model,
    model_activations,
    layer_number,
    batch_size=16,
    steps=200,  # Reduced steps to 100 (1/10th of 1000)
    lr=1e-3,
    num_envs=4,
    episode_length=150,
    log_freq=10,
    checkpoint_dir='checkpoints',
    wandb_project="SAE_training",
):
    # Unique identifier for each layer to manage separate checkpoints and wandb runs
    layer_identifier = f'layer_{layer_number}_{ordered_layer_names[layer_number]}'

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

    progress_bar = tqdm(range(start_step, steps), desc=f"Training {layer_identifier}")
    loss_history = []
    L_reconstruction_history = []
    L_sparsity_history = []
    variance_explained_history = []

    for step in progress_bar:
        try:
            # Generate batch activations
            activations = generate_batch_activations_parallel(
                model, model_activations, layer_number,
                batch_size=batch_size, num_envs=num_envs, episode_length=episode_length
            )
            # Move activations directly to the device
            batch = activations.to(device)
        except Exception as e:
            print(f"Error during activation generation: {e}")
            continue  # Skip this iteration and proceed

        optimizer.zero_grad()
        loss, L_reconstruction, L_sparsity, acts, h_reconstructed = sae(batch)
        nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()

        with t.no_grad():
            # Compute variance explained
            numerator = t.sum((batch - h_reconstructed) ** 2, dim=1)
            denominator = t.sum(batch ** 2, dim=1)
            variance_explained = 1 - numerator / (denominator + 1e-8)
            variance_explained = variance_explained.mean().item()
            acts_mean = acts.mean().item()
            acts_min = acts.min().item()
            acts_max = acts.max().item()
        if step % log_freq == 0 or step == steps - 1:
            progress_bar.set_postfix({
                'Loss': loss.item(),
                'Reconstruction': L_reconstruction.item(),
                'Sparsity': L_sparsity.item()
            })
            loss_history.append(loss.item())
            L_reconstruction_history.append(L_reconstruction.item())
            L_sparsity_history.append(L_sparsity.item())
            variance_explained_history.append(variance_explained)

            # Log to wandb
            wandb.log({
            "step": step,
            "loss": loss.item(),
            "reconstruction_loss": L_reconstruction.item(),
            "sparsity_loss": L_sparsity.item(),
            "acts_mean": acts_mean,
            "acts_min": acts_min,
            "acts_max": acts_max,
            "variance_explained": variance_explained
        })

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

# Example usage:

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

# Load the model
model = load_interpretable_model()
model.to(device)
model.eval()  # Set model to evaluation mode

# Function to train SAEs for all layers
def train_all_layers(
    model,
    ordered_layer_names,
    layer_types,  # Pass the layer_types dictionary
    checkpoint_dir='checkpoints',
    wandb_project="SAE_training",
    steps_per_layer=100,  # Reduced steps to 100
    batch_size=24,        # Adjust based on your hardware
    lr=1e-3,
    num_envs=4,           # Number of parallel environments
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
        sample_activations = generate_batch_activations_parallel(
            model, model_activations, layer_number, batch_size=1
        )
        d_in = sample_activations.shape[-1]  # Input dimension for SAE

        # Configure SAE
        sae_cfg = SAEConfig(
            d_in=d_in,
            d_hidden=d_hidden,    # Assigned based on layer type
            l1_coeff=l1_coeff,    # Assigned based on layer type
            tied_weights=False     # Adjust if needed
        )

        # Initialize SAE
        sae = SAE(sae_cfg)

        # Train SAE
        loss_history, L_reconstruction_history, L_sparsity_history = train_sae(
            sae=sae,
            model=model,
            model_activations=model_activations,
            layer_number=layer_number,
            batch_size=batch_size,   # Adjust based on your hardware
            steps=steps_per_layer,   # Reduced number of steps
            lr=lr,
            num_envs=num_envs,        # Number of parallel environments
            episode_length=episode_length,
            log_freq=log_freq,
            checkpoint_dir=checkpoint_dir,
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