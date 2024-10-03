# %%
import sys
from src.utils import helpers, heist
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
import glob
import matplotlib.pyplot as plt
import sae
from src.utils import helpers, heist

# from src.perform_sae_analysis import measure_logit_difference, collect_strong_activations

# Set device
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Ordered layer names
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

# Define layer types
layer_types = {
    "conv1a": "conv",
    "pool1": "pool",
    "conv2a": "conv",
    "conv2b": "conv",
    "pool2": "pool",
    "conv3a": "conv",
    "pool3": "pool",
    "conv4a": "conv",
    "pool4": "pool",
    "fc1": "fc",
    "fc2": "fc",
    "fc3": "fc",
    "value_fc": "fc",
    "dropout_conv": "dropout_conv",
    "dropout_fc": "dropout_fc",
}


def measure_logit_difference(model, sae, layer_number, num_samples=100):
    # Generate observations
    observations = []
    env = heist.create_venv(num=1, num_levels=0, start_level=random.randint(1, 100000))
    obs = env.reset()
    for _ in range(num_samples):
        observations.append(obs[0].copy())
        action = env.action_space.sample()
        obs, reward, done, info = env.step(np.array([action]))
        if done[0]:
            obs = env.reset()

    # Convert observations to tensor
    obs_tensor = t.tensor(np.array(observations), dtype=t.float32)
    print(obs_tensor.shape)
    obs_tensor = einops.rearrange(obs_tensor, " b c h w -> b h  w c").to(device)
    print(obs_tensor.shape)

    # Get logits without SAE
    with t.no_grad():
        outputs = model(obs_tensor)
        logits_without_sae = (
            outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        )
        logits_without_sae = logits_without_sae.cpu().numpy()

    # Register the hook to replace the layer with SAE outputs
    handle = replace_layer_with_sae(model, sae, layer_number)

    # Get logits with SAE
    with t.no_grad():
        outputs = model(obs_tensor)
        logits_with_sae = (
            outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        )
        logits_with_sae = logits_with_sae.cpu().numpy()

    # Remove the hook
    handle.remove()

    # Compute differences
    logit_differences = logits_with_sae - logits_without_sae

    # Return the differences
    return logits_without_sae, logits_with_sae, logit_differences


# %%


# Function to collect strongly activating features and corresponding observations
def collect_strong_activations(
    sae,
    model,
    layer_number,
    threshold=1.0,
    num_episodes=10,
    max_steps_per_episode=1000,
    feature_indices=None,
    device=device,
):
    # Ensure model and sae are in eval mode
    model.eval()
    sae.eval()

    # Initialize ModelActivations for the layer
    layer_name = ordered_layer_names[layer_number]
    layer_paths = [layer_name]
    model_activations = helpers.ModelActivations(model, layer_paths=layer_paths)

    # Initialize data structure to store observations
    from collections import defaultdict

    feature_observations = defaultdict(list)

    # If feature_indices is None, monitor all features
    if feature_indices is None:
        feature_indices = range(sae.cfg.d_hidden)

    # Create the environment
    env = heist.create_venv(
        num=1,
        num_levels=0,
        start_level=random.randint(1, 100000),
        distribution_mode="easy",
    )

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            obs_tensor = t.tensor(obs, dtype=t.float32)  # Add batch dimension

            # Run the model and get activations
            with t.no_grad():
                outputs, activations = model_activations.run_with_cache(obs_tensor)
                # Get the layer activation
                layer_activation = activations[layer_name.replace(".", "_")]
                # Flatten layer activation
                h = layer_activation.view(1, -1).to(device)
                # Pass through SAE to get feature activations
                _, _, _, acts = sae(h)
                acts = acts.cpu().numpy()[0]  # Get numpy array, remove batch dimension

            # Check for strongly activated features
            for idx in feature_indices:
                activation_value = acts[idx]
                if activation_value >= threshold:
                    # Record the observation and activation value
                    feature_observations[idx].append((obs.copy(), activation_value))

            # Get action from the model outputs
            logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
            probabilities = t.softmax(logits, dim=-1)
            action = t.multinomial(probabilities, num_samples=1).item()

            # Step the environment
            obs, reward, done, info = env.step(np.array([action]))
            steps += 1

    env.close()
    return feature_observations


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
            elements = path.split(".")
            module = self.model
            for element in elements:
                if "[" in element and "]" in element:
                    base, idx = element.split("[")
                    idx = int(idx[:-1])
                    module = getattr(module, base)[idx]
                else:
                    module = getattr(module, element)
            hook = module.register_forward_hook(
                self.get_activation(path.replace(".", "_"))
            )
            self.hooks.append(hook)

    def run_with_cache(self, inputs):
        self.activations = {}
        inputs = einops.rearrange(inputs, "b h c w -> b c h w ")
        inputs = inputs.to(next(self.model.parameters()).device)
        outputs, value = self.model(inputs)
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
        L_reconstruction = F.mse_loss(h_reconstructed, h, reduction="mean")
        L_sparsity = acts.abs().mean()
        loss = L_reconstruction + self.cfg.l1_coeff * L_sparsity

        return loss, L_reconstruction, L_sparsity, acts, h_reconstructed


# Function to load the main model (policy network)
def load_interpretable_model(model_path="../model_interpretable.pt"):
    import gym
    from src.interpretable_impala import CustomCNN  # Import your model class

    env_name = "procgen:procgen-heist-v0"
    env = gym.make(
        env_name,
        start_level=100,
        num_levels=200,
        render_mode="rgb_array",
        distribution_mode="easy",
    )
    observation_space = env.observation_space
    action_space = env.action_space.n
    model = CustomCNN(observation_space, action_space)
    model.load_from_file(model_path, device=device)
    return model


# Function to load the trained SAE model and get activation shape
def load_sae_model(layer_number, sae_model_path):
    # Load the main model
    model = load_interpretable_model()
    model.to(device)
    model.eval()

    # Define the layer paths
    layer_name = ordered_layer_names[layer_number]
    layer_paths = [layer_name]

    # Initialize ModelActivations
    model_activations = ModelActivations(model, layer_paths=layer_paths)

    # Get a sample observation
    env = heist.create_venv(num=1, num_levels=0, start_level=random.randint(1, 100000))
    obs = env.reset()
    obs = t.tensor(obs, dtype=t.float32)  # Add batch dimension

    # Run model_activations to get activation shape
    with t.no_grad():
        outputs, activations = model_activations.run_with_cache(obs)
    layer_activation = activations[layer_name.replace(".", "_")]
    activation_shape = layer_activation.shape  # [batch_size, channels, height, width]
    _, channels, height, width = activation_shape
    d_in = channels * height * width

    # Load the checkpoint
    try:
        checkpoint = t.load(sae_model_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: SAE model file not found at {sae_model_path}")
        return None
    except Exception as e:
        print(f"Error loading SAE model: {str(e)}")
        return None

    if checkpoint is None:
        print("Error: Failed to load SAE model checkpoint")
        return None
    state_dict = checkpoint["model_state_dict"]

    # Infer d_in and d_hidden from the shapes of W_enc
    W_enc_shape = state_dict["W_enc"].shape  # Should be [d_in, d_hidden]
    inferred_d_in, inferred_d_hidden = W_enc_shape

    # Optionally, verify that inferred_d_in matches d_in
    if inferred_d_in != d_in:
        print(
            f"Warning: Inferred d_in ({inferred_d_in}) does not match computed d_in ({d_in})"
        )
        d_in = inferred_d_in  # Use inferred d_in

    # Create SAEConfig using inferred dimensions
    sae_cfg = SAEConfig(
        d_in=inferred_d_in,
        d_hidden=inferred_d_hidden,
        l1_coeff=0.05,  # You may retrieve l1_coeff and tied_weights from elsewhere if needed
        tied_weights=False,
    )

    # Initialize SAE
    sae = SAE(sae_cfg)

    # Load state dict
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()  # Set to evaluation mode

    print("Returning values:")
    print("sae:", sae)
    print("activation_shape:", activation_shape)
    print("model_activations:", model_activations)
    print("model:", model)
    print("layer_name:", layer_name)

    return sae, activation_shape, model_activations, model, layer_name


# %%


def replace_layer_with_sae(model, sae, layer_number):
    # Get the layer name
    layer_name = ordered_layer_names[layer_number]
    # Locate the module
    elements = layer_name.split(".")
    module = model
    for element in elements:
        if "[" in element and "]" in element:
            base, idx = element.split("[")
            idx = int(idx[:-1])
            module = getattr(module, base)[idx]
        else:
            module = getattr(module, element)

    # Define the hook function
    def hook_fn(module, input, output):

        print(input[0].shape)
        # h = layer_activation.view(1, -1).to(device)

        h = input[0].flatten()
        print(h.shape)
        # Pass through SAE
        return None
        _, _, _, acts, h_reconstructed = sae(h)

        # Reshape h_reconstructed back to original shape using reshape_as
        h_reconstructed = h_reconstructed.reshape_as(output)

        print("Input:", input)
        print("Reconstructed:", h_reconstructed)
        print("Original output:", output)
        print(output.shape, h_reconstructed.shape)

        print("Output diff", output - h_reconstructed)

        return h_reconstructed

    # Register the forward hook
    handle = module.register_forward_hook(hook_fn)
    return handle  # Return the handle to remove the hook later


sae_model_path = "../src/checkpoints/layer_6_conv3a/sae_checkpoint_step_100000.pt"  # Path to your saved SAE model


# # Assuming you have trained the SAE for the desired layer
layer_number = 6  # Replace with your target layer number
sae, _, _, model, _ = load_sae_model(layer_number, sae_model_path)
# %%
# Measure logit differences
logits_without_sae, logits_with_sae, logit_differences = measure_logit_difference(
    model, sae, layer_number, num_samples=100
)

# Analyze the differences
mean_difference = np.mean(np.abs(logit_differences))
print(f"Mean absolute difference in logits: {mean_difference}")

# Optionally, visualize the differences
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(logit_differences.flatten(), bins=50)
plt.title(f"Histogram of Logit Differences (Layer {layer_number})")
plt.xlabel("Logit Difference")
plt.ylabel("Frequency")
plt.show()


# %%
def evaluate_model_performance(model, sae, layer_number, num_episodes=10):
    # Function to run episodes and collect total rewards
    def run_episodes(model, num_episodes, save_gif=False, with_sae=False):
        total_rewards = []
        for episode in range(num_episodes):
            env = heist.create_venv(
                num=1, num_levels=0, start_level=random.randint(1, 100000)
            )
            obs = env.reset()
            done = False
            total_reward = 0
            frames = []
            steps = 0
            print(f"Episode {episode + 1}/{num_episodes}")
            while not done and steps < 100:
                obs = helpers.observation_to_rgb(obs)
                obs_tensor = t.tensor(np.array(obs), dtype=t.float32)
                obs_tensor = einops.rearrange(obs_tensor, " b c h w -> b h w c").to(
                    device
                )
                with t.no_grad():
                    outputs = model(obs_tensor)
                    logits = (
                        outputs[0].logits
                        if isinstance(outputs, tuple)
                        else outputs.logits
                    )
                    action = t.argmax(logits, dim=-1).cpu().numpy()
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                steps += 1
                if save_gif:
                    frame = env.render(mode="rgb_array")
                    frames.append(frame)
            total_rewards.append(total_reward)

            if save_gif:
                import imageio

                # Save the gif
                sae_suffix = "with_sae" if with_sae else "without_sae"
                gif_path = f"episode_{episode}_{sae_suffix}.gif"
                imageio.mimsave(gif_path, frames, fps=30)

        env.close()
        return total_rewards

    # Run episodes without SAE
    model.eval()
    rewards_without_sae = run_episodes(
        model, num_episodes, save_gif=False, with_sae=False
    )

    # Register the hook
    handle = replace_layer_with_sae(model, sae, layer_number)

    # Run episodes with SAE
    model.eval()
    rewards_with_sae = run_episodes(model, num_episodes, save_gif=False, with_sae=True)

    # Remove the hook
    handle.remove()

    # Compare results
    avg_reward_without_sae = np.mean(rewards_without_sae)
    avg_reward_with_sae = np.mean(rewards_with_sae)

    print(f"Average reward without SAE: {avg_reward_without_sae}")
    print(f"Average reward with SAE: {avg_reward_with_sae}")

    return rewards_without_sae, rewards_with_sae


model = load_interpretable_model().to(device)
evaluate_model_performance(model, sae, layer_number, 10)

# %%

# %%
