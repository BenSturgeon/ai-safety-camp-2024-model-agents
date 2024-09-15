# %%

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

import sys
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
        acts = F.relu(t.matmul(h, self.W_enc) + self.b_enc)  # [batch_size, d_hidden]

        # Decoder
        h_reconstructed = t.matmul(acts, self.W_dec) + self.b_dec  # [batch_size, d_in]

        # Loss components
        L_reconstruction = F.mse_loss(h_reconstructed, h, reduction='mean')
        L_sparsity = acts.abs().mean()
        loss = L_reconstruction + self.cfg.l1_coeff * L_sparsity

        return loss, L_reconstruction, L_sparsity, acts

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

# Training function for SAE
def train_sae(
    sae,
    model,
    model_activations,
    layer_number,
    batch_size=16,
    steps=1000,
    lr=1e-3,
    num_envs=4,
    episode_length=150,
    log_freq=10,
):
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    progress_bar = tqdm(range(steps))
    loss_history = []
    L_reconstruction_history = []
    L_sparsity_history = []

    for step in progress_bar:
        # Generate batch activations
        activations = generate_batch_activations_parallel(
            model, model_activations, layer_number,
            batch_size=batch_size, num_envs=num_envs, episode_length=episode_length
        )

        # Create DataLoader for efficient data loading
        dataset = TensorDataset(activations)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_iter = iter(data_loader)

        try:
            batch = next(data_iter)[0].to(device)
        except StopIteration:
            continue  # Should not happen as we have one batch

        optimizer.zero_grad()
        loss, L_reconstruction, L_sparsity, acts = sae(batch)
        loss.backward()
        optimizer.step()

        if step % log_freq == 0 or step == steps - 1:
            progress_bar.set_postfix({
                'Loss': loss.item(),
                'Reconstruction': L_reconstruction.item(),
                'Sparsity': L_sparsity.item()
            })
            loss_history.append(loss.item())
            L_reconstruction_history.append(L_reconstruction.item())
            L_sparsity_history.append(L_sparsity.item())

    return loss_history, L_reconstruction_history, L_sparsity_history

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

# Choose the layer number you're interested in
layer_number = 8  # For example

# Define layer paths for activation capture
layer_paths = [ordered_layer_names[layer_number]]  # Capture only the target layer

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
    d_hidden=64,    # Adjust based on your requirements
    l1_coeff=0.1,   # Adjust regularization coefficient as needed
    tied_weights=False
)

# Initialize SAE
sae = SAE(sae_cfg)

# Train SAE
loss_history, L_reconstruction_history, L_sparsity_history = train_sae(
    sae=sae,
    model=model,
    model_activations=model_activations,
    layer_number=layer_number,
    batch_size=24,   # Adjust based on your hardware
    steps=1000,        # Number of training steps
    lr=1e-3,
    num_envs=4,        # Number of parallel environments
    episode_length=150,
    log_freq=10,
)

# Plot training losses (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(loss_history)
plt.title('Total Loss')

plt.subplot(1, 3, 2)
plt.plot(L_reconstruction_history)
plt.title('Reconstruction Loss')

plt.subplot(1, 3, 3)
plt.plot(L_sparsity_history)
plt.title('Sparsity Loss')

plt.tight_layout()
plt.show()
