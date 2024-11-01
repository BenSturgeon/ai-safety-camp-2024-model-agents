
# sae.py
import io
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
from PIL import Image
from src.extract_sae_features import replace_layer_with_sae

from src.utils import helpers, heist

def get_device():
    if t.cuda.is_available():
        return t.device("cuda")
    elif hasattr(t.backends, "mps") and t.backends.mps.is_available():
        return t.device("mps")
    else:
        return t.device("cpu")

device = get_device()
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

class ReplayBuffer:
    """
    A replay buffer to store and sample activations and observations.
    """
    def __init__(self, capacity, activation_shape, observation_shape, device):
        self.capacity = capacity
        self.activation_shape = activation_shape  
        self.observation_shape = observation_shape
        self.device = device
        self.activations = t.empty((capacity, *activation_shape), device=device)
        self.observations = t.empty((capacity, *observation_shape), device=device)
        self.size = 0
        self.position = 0  

    def add(self, activation, observation):
        self.activations[self.position] = activation.to(self.device)
        self.observations[self.position] = observation.to(self.device)
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("Not enough samples in the replay buffer to sample.")
        indices = random.sample(range(self.size), batch_size)
        sampled_activations = self.activations[indices]
        sampled_observations = self.observations[indices]
        return sampled_activations, sampled_observations

    def is_full(self):
        return self.size >= self.capacity

    def empty(self):
        self.size = 0
        self.position = 0
        self.activations = t.empty((self.capacity, *self.activation_shape), device=self.device)
        self.observations = t.empty((self.capacity, *self.observation_shape), device=self.device)

def jump_relu(x, jump_value=0.1):
    """
    Applies a jump ReLU activation function.
    """
    out = t.where(x > 0, x + jump_value, t.zeros_like(x))
    return out

def topk_activation(x, k):
    """
    Applies Top-k activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features).
        k (int): Number of top activations to keep per sample.

    Returns:
        torch.Tensor: Tensor with only the top k activations per sample kept.
    """
    values, indices = t.topk(x, k=k, dim=1)
    mask = t.zeros_like(x)
    mask.scatter_(1, indices, 1)
    return x * mask

@dataclass
class SAEConfig:
    """
    Configuration for the Sparse Autoencoder (SAE).
    """
    d_in: int = None
    d_hidden: int = 128
    l1_coeff: float = 0.1
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    jump_value: float = 0.1

class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.W_enc = nn.Parameter(t.empty(self.cfg.d_in, self.cfg.d_hidden, device=device))
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        self.b_enc = nn.Parameter(t.zeros(self.cfg.d_hidden, device=device))

        if self.cfg.tied_weights:
            self.W_dec = self.W_enc.t()
        else:
            self.W_dec = nn.Parameter(t.empty(self.cfg.d_hidden, self.cfg.d_in, device=device))
            nn.init.kaiming_uniform_(self.W_dec, nonlinearity="relu")
        self.b_dec = nn.Parameter(t.zeros(self.cfg.d_in, device=device))

        self.to(device)

    def forward(self, h):
        h = h.to(self.device)
        z = t.matmul(h, self.W_enc) + self.b_enc
        acts = F.relu(z)

        h_reconstructed = t.matmul(acts, self.W_dec) + self.b_dec

        L_reconstruction = F.mse_loss(h_reconstructed, h, reduction="mean")
        L_sparsity = acts.abs().mean()
        loss = self.cfg.l1_coeff * L_sparsity

        return loss, L_reconstruction, L_sparsity, acts, h_reconstructed

def collect_activations_into_replay_buffer(
    model,
    model_activations,
    layer_number,
    replay_buffer,
    num_envs=8,
    episode_length=150,
):
    """
    Run multiple environments in parallel, collect activations over full episodes,
    and store them into the replay buffer.
    """
    venv = heist.create_venv(
        num=num_envs,
        num_levels=0,
        start_level=random.randint(1, 100000),
    )

    obs = venv.reset()
    steps = np.zeros(num_envs, dtype=int)
    episode_counts = np.zeros(num_envs, dtype=int)
    max_episodes_per_env = 1

    activation_lists = [[] for _ in range(num_envs)]
    observation_lists = [[] for _ in range(num_envs)]

    while not replay_buffer.is_full():
        observations = t.tensor(obs, dtype=t.float32).to(device)
        observations = einops.rearrange(observations, "b h w c -> b c h w")

        layer_name = ordered_layer_names[layer_number]
        with t.no_grad():
            outputs, activations = model_activations.run_with_cache(
                observations, layer_name
            )
        layer_activation = activations[layer_name.replace(".", "_")]

        if isinstance(layer_activation, tuple):
            for i in range(num_envs):
                activation_i = layer_activation[i]
                activation_lists[i].append(activation_i.cpu())
                observation_lists[i].append(observations[i].cpu())
        else:
            for i in range(num_envs):
                activation_lists[i].append(layer_activation[i].cpu())
                observation_lists[i].append(observations[i].cpu())

        logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        probabilities = t.softmax(logits, dim=-1)
        actions = t.multinomial(probabilities, num_samples=1).squeeze(1).cpu().numpy()

        obs, rewards, dones_env, infos = venv.step(actions)
        steps += 1

        for i, done in enumerate(dones_env):
            if done or steps[i] >= episode_length:
                activations_tensor = t.stack(activation_lists[i], dim=0)
                observations_tensor = t.stack(observation_lists[i], dim=0)
                for t_step in range(activations_tensor.size(0)):
                    replay_buffer.add(
                        activations_tensor[t_step],
                        observations_tensor[t_step],
                    )
                activation_lists[i] = []
                observation_lists[i] = []
                steps[i] = 0
                episode_counts[i] += 1

        if np.all(episode_counts >= max_episodes_per_env):
            break

    venv.close()

def generate_batch_activations_parallel(
    model,
    model_activations,
    layer_number,
    batch_size=32,
    num_envs=8,
    episode_length=150,
):
    """
    Generate activations from multiple environments running in parallel.
    """
    venv = heist.create_venv(
        num=num_envs, num_levels=0, start_level=random.randint(1, 100000)
    )

    activation_buffer = []
    obs_buffer = []
    activation_shapes = None

    obs = venv.reset()
    steps = 0

    while len(activation_buffer) * num_envs < batch_size:
        observations = t.tensor(obs, dtype=t.float32).to(device)
        observations = einops.rearrange(observations, "b h w c -> b c h w")

        layer_name = ordered_layer_names[layer_number]
        with t.no_grad():
            outputs, activations = model_activations.run_with_cache(
                observations, layer_name
            )
        layer_activation = activations[layer_name.replace(".", "_")]

        activation_buffer.append(layer_activation.cpu())
        obs_buffer.append(observations.cpu())

        logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        probabilities = t.softmax(logits, dim=-1)
        actions = t.multinomial(probabilities, num_samples=1).squeeze(1).cpu().numpy()

        obs, rewards, dones_env, infos = venv.step(actions)
        steps += 1

        if steps > episode_length:
            break

    venv.close()

    activations_tensor = t.cat(activation_buffer, dim=0)
    if activations_tensor.size(0) >= batch_size:
        activations_tensor = activations_tensor[:batch_size]
    else:
        print(
            f"Warning: Collected only {activations_tensor.size(0)} activations, expected {batch_size}."
        )

    observations_tensor = t.cat(obs_buffer, dim=0)[:batch_size]

    return activations_tensor, observations_tensor, activation_shapes

def find_latest_checkpoint(checkpoint_dir):
    """
    Helper function to find the latest checkpoint.
    """
    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, "sae_checkpoint_step_*.pt")
    )
    if not checkpoint_files:
        return None, None
    steps = [
        int(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
        for f in checkpoint_files
    ]
    latest_step = max(steps)
    latest_checkpoint = os.path.join(
        checkpoint_dir, f"sae_checkpoint_step_{latest_step}.pt"
    )
    return latest_checkpoint, latest_step

def get_layer_hyperparameters(layer_name, layer_types):
    """
    Function to assign hyperparameters based on layer type.
    """
    layer_type = layer_types.get(layer_name, "other")
    if layer_type == "conv":
        return {
            "d_hidden": 16384,
            "l1_coeff": 0.000001,
        }
    elif layer_type == "fc":
        return {
            "d_hidden": 2048,
            "l1_coeff": 0.0001,
        }
    else:
        return {
            "d_hidden": 256,
            "l1_coeff": 0.0001,
        }

def get_module_by_path(model, layer_path):
    """
    Get a module from the model by its path.
    """
    elements = layer_path.split(".")
    module = model
    for element in elements:
        if "[" in element and "]" in element:
            base, idx = element.split("[")
            idx = int(idx[:-1])
            module = getattr(module, base)[idx]
        else:
            module = getattr(module, element)
    return module

def run_test_episodes(model, num_envs=8, episode_length=150):
    """
    Run test episodes and return rewards.
    """
    venv = heist.create_venv(
        num=num_envs, num_levels=0, start_level=random.randint(1, 100000)
    )
    obs = venv.reset()
    total_rewards = np.zeros(num_envs)
    steps = np.zeros(num_envs, dtype=int)

    while np.any(steps < episode_length):
        observations = t.tensor(obs, dtype=t.float32).to(device)
        observations = einops.rearrange(observations, "b h w c -> b c h w")

        with t.no_grad():
            outputs = model(observations)
            logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
            actions = t.argmax(logits, dim=-1).cpu().numpy()

        obs, rewards, dones, _ = venv.step(actions)
        total_rewards += rewards
        steps += 1

        for i, done in enumerate(dones):
            if done:
                total_rewards[i] = 0
                steps[i] = 0

    venv.close()
    return total_rewards

def train_sae(
    sae,
    model,
    model_activations,
    layer_number,
    layer_name,
    batch_size=128,
    steps=200,
    lr=1e-3,
    num_envs=8,
    episode_length=150,
    log_freq=10,
    checkpoint_dir="checkpoints",
    stats_dir="global_stats",
    wandb_project="SAE_training",
):
    """
    Train a Sparse Autoencoder (SAE) for a specific layer.
    """
    layer_identifier = f"layer_{layer_number}_{layer_name}"
    layer_checkpoint_dir = os.path.join(checkpoint_dir, layer_identifier)
    os.makedirs(layer_checkpoint_dir, exist_ok=True)

    latest_checkpoint, latest_step = find_latest_checkpoint(layer_checkpoint_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = t.load(latest_checkpoint, map_location=device)
        sae.load_state_dict(checkpoint["model_state_dict"])
        optimizer = optim.Adam(sae.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        wandb.init(
            project=wandb_project,
            resume="allow",
            config={
                "batch_size": batch_size,
                "steps": steps,
                "lr": lr,
                "num_envs": num_envs,
                "episode_length": episode_length,
                "layer_number": layer_number,
                "resume_step": latest_step,
                "layer_identifier": layer_identifier,
            },
            name=f"SAE_{layer_identifier}",
        )
    else:
        optimizer = optim.Adam(sae.parameters(), lr=lr)
        wandb.init(
            project=wandb_project,
            config={
                "batch_size": batch_size,
                "steps": steps,
                "lr": lr,
                "num_envs": num_envs,
                "episode_length": episode_length,
                "layer_number": layer_number,
                "layer_identifier": layer_identifier,
            },
            name=f"SAE_{layer_identifier}",
        )
        start_step = 0

    module_at_layer = get_module_by_path(model, layer_name)

    sample_activations, sample_observations, activation_shapes = (
        generate_batch_activations_parallel(
            model,
            model_activations,
            layer_number,
            batch_size=batch_size,
            num_envs=num_envs,
            episode_length=episode_length,
        )
    )
    d_in = sample_activations.view(sample_activations.size(0), -1).shape[-1]
    activation_shape = sample_activations.shape[1:]
    observation_shape = sample_observations.shape[1:]

    buffer_capacity = batch_size * 50
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity,
        activation_shape=activation_shape,
        observation_shape=observation_shape,
        device=device,
    )

    progress_bar = tqdm(range(start_step, steps), desc=f"Training {layer_identifier}")
    loss_history = []
    L_reconstruction_history = []
    L_sparsity_history = []
    variance_explained_history = []

    model.eval()
    active_logits_histogram = None
    for step in progress_bar:
        if step % 50 == 0:
            replay_buffer.empty()
            collect_activations_into_replay_buffer(
                model,
                model_activations,
                layer_number,
                replay_buffer,
                num_envs=num_envs,
                episode_length=episode_length,
            )

            all_activations = replay_buffer.activations[: replay_buffer.size]
            buffer_mean = t.mean(all_activations, dim=0)
            buffer_std = t.std(all_activations, dim=0)

            print(
                f"Replay Buffer Stats - Mean: {buffer_mean.mean().item():.4f}, Std: {buffer_std.mean().item():.4f}"
            )

        try:
            activations_unflattened, observations = replay_buffer.sample(batch_size)
        except Exception as e:
            print(f"Error during activation sampling: {e}")
            continue

        activations_unflattened = (activations_unflattened - buffer_mean) / (
            buffer_std + 1e-8
        )

        activations = activations_unflattened.view(batch_size, -1).to(device)

        observations = observations.to(device)

        optimizer.zero_grad()
        loss, L_reconstruction, L_sparsity, acts, h_reconstructed = sae(activations)

        batch_size_curr = h_reconstructed.size(0)

        buffer_std = buffer_std.view(1, 32, 16, 16)
        buffer_mean = buffer_mean.view(1, 32, 16, 16)

        h_reconstructed = h_reconstructed.view(batch_size_curr, 32, 16, 16)

        h_reconstructed = h_reconstructed * buffer_std + buffer_mean

        observations_prepared = einops.rearrange(observations, "b h w c -> b c h w")

        with t.no_grad():
            model.eval()
            outputs_original = model(observations_prepared)
            logits_original = (
                outputs_original[0].logits
                if isinstance(outputs_original, tuple)
                else outputs_original.logits
            )

        def replace_activation_hook(module, input, output):
            return h_reconstructed

        hook_handle = module_at_layer.register_forward_hook(replace_activation_hook)
        outputs_reconstructed = model(observations_prepared)
        logits_reconstructed = (
            outputs_reconstructed[0].logits
            if isinstance(outputs_reconstructed, tuple)
            else outputs_reconstructed.logits
        )
        hook_handle.remove()

        logits_diff = F.kl_div(
            F.log_softmax(logits_reconstructed, dim=-1),
            F.log_softmax(logits_original, dim=-1),
            reduction="batchmean",
            log_target=True,
        )

        total_loss = 10 * logits_diff + L_reconstruction

        total_loss.backward()
        optimizer.step()

        with t.no_grad():
            activations_reshaped = activations.view(batch_size_curr, 32, 16, 16)
            h_reconstructed_reshaped = h_reconstructed.view(batch_size_curr, 32, 16, 16)
            
            activations_flat = activations_reshaped.view(batch_size_curr, -1)
            h_reconstructed_flat = h_reconstructed_reshaped.view(batch_size_curr, -1)
            
            numerator = t.sum((activations_flat - h_reconstructed_flat) ** 2, dim=1)
            denominator = t.sum(activations_flat**2, dim=1)
            variance_explained = 1 - numerator / (denominator + 1e-8)
            variance_explained = variance_explained.mean().item()
            acts_mean = acts.mean().item()
            acts_min = acts.min().item()
            acts_max = acts.max().item()

        if step % log_freq == 0 or step == steps - 1:
            progress_bar.set_postfix(
                {
                    "Loss": loss.item(),
                    "total loss": total_loss.item(),
                    "Reconstruction": L_reconstruction.item(),
                    "Sparsity": L_sparsity.item(),
                    "Variance Explained": variance_explained,
                    "Logits diff": logits_diff,
                }
            )
            loss_history.append(loss.item())
            L_reconstruction_history.append(L_reconstruction.item())
            L_sparsity_history.append(L_sparsity.item())
            variance_explained_history.append(variance_explained)

            wandb.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "total loss": total_loss.item(),
                    "reconstruction_loss": L_reconstruction.item(),
                    "sparsity_loss": L_sparsity.item(),
                    "logits_diff": logits_diff.item(),
                    "acts_mean": acts_mean,
                    "acts_min": acts_min,
                    "acts_max": acts_max,
                    "variance_explained": variance_explained,
                },
                step=step,
            )

            if step % 1000 == 0:
                import matplotlib.pyplot as plt

                logit_differences = logits_reconstructed - logits_original
                plt.figure(figsize=(10, 6))
                plt.hist(logit_differences.flatten().cpu().detach().numpy(), bins=50)
                plt.title(f"Histogram of Logit Differences (Layer {layer_number})")
                plt.xlabel("Logit Difference")
                plt.ylabel("Frequency")

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                img = Image.open(buf)

                rewards_without_sae = run_test_episodes(
                    model, num_envs=8, episode_length=150
                )

                handle = replace_layer_with_sae(model, sae, layer_number)

                rewards_with_sae = run_test_episodes(
                    model, num_envs=8, episode_length=50
                )
                avg_reward_with_sae = sum(rewards_with_sae) / len(rewards_with_sae)

                handle.remove()

                wandb.log(
                    {
                        "logit_differences_histogram": wandb.Image(img),
                        "avg_reward_with_sae": avg_reward_with_sae,
                    },
                    step=step,
                )

                plt.close()

        if (step + 1) % 10000 == 0 and step > 1 or step == steps - 1:
            checkpoint_path = os.path.join(
                layer_checkpoint_dir, f"sae_checkpoint_step_{step+1}.pt"
            )
            t.save(
                {
                    "step": step,
                    "model_state_dict": sae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                checkpoint_path,
            )

    wandb.finish()
    return (
        loss_history,
        L_reconstruction_history,
        L_sparsity_history,
        variance_explained_history,
    )


def load_interpretable_model(model_path="../model_interpretable.pt"):
    import gym
    from src.interpretable_impala import CustomCNN

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


def compute_and_save_global_stats(
    model,
    layer_number,
    layer_name,
    num_samples=10000,
    batch_size=128,
    num_envs=64,
    save_dir="global_stats",
):
    """
    Computes and saves the global mean and std for a given layer's activations.
    """
    os.makedirs(save_dir, exist_ok=True)
    layer_paths = [layer_name]
    model_activations = helpers.ModelActivations(model)
    activation_samples = []
    total_samples = 0
    with tqdm(
        total=num_samples, desc=f"Computing global stats for {layer_name}"
    ) as pbar:
        while total_samples < num_samples:
            activations, _, _ = generate_batch_activations_parallel(
                model,
                model_activations,
                layer_number,
                batch_size=batch_size,
                num_envs=num_envs,
            )
            activation_samples.append(activations)
            total_samples += activations.size(0)
            pbar.update(activations.size(0))
    all_activations = t.cat(activation_samples, dim=0)[:num_samples]
    global_mean = all_activations.mean(dim=0, keepdim=True)
    global_std = all_activations.std(dim=0, keepdim=True) + 1e-8
    t.save(
        {"mean": global_mean, "std": global_std},
        os.path.join(save_dir, f"layer_{layer_number}_{layer_name}_stats.pt"),
    )
    model_activations.clear_hooks()


def compute_global_stats_for_all_layers(
    model,
    ordered_layer_names,
    num_samples_per_layer=10000,
    batch_size=128,
    num_envs=128,
    save_dir="global_stats",
):
    for layer_number, layer_name in ordered_layer_names.items():
        compute_and_save_global_stats(
            model,
            layer_number,
            layer_name,
            num_samples=num_samples_per_layer,
            batch_size=batch_size,
            num_envs=num_envs,
            save_dir=save_dir,
        )


def train_all_layers(
    model,
    ordered_layer_names,
    layer_types,
    checkpoint_dir="checkpoints",
    stats_dir="global_stats",
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

        hyperparams = get_layer_hyperparameters(layer_name, layer_types)
        d_hidden = hyperparams["d_hidden"]
        l1_coeff = hyperparams["l1_coeff"]

        layer_paths = [layer_name]

        model_activations = helpers.ModelActivations(model)
        model.eval()
        sample_activations, _, _ = generate_batch_activations_parallel(
            model,
            model_activations,
            layer_number,
            batch_size=batch_size,
            num_envs=num_envs,
            episode_length=episode_length,
        )
        print("initial size:", sample_activations.shape[-1])

        print(f"sample_activations shape before flattening: {sample_activations.shape}")
        flattened_sample_activations = sample_activations.view(
            sample_activations.size(0), -1
        )
        print(
            f"flattened_sample_activations shape: {flattened_sample_activations.shape}"
        )
        d_in = flattened_sample_activations.shape[1]
        print(f"d_in: {d_in}")
        sae_cfg = SAEConfig(
            d_in=d_in,
            d_hidden=d_hidden,
            l1_coeff=l1_coeff,
            tied_weights=False,
        )

        sae_model = SAE(sae_cfg, device)

        (
            loss_history,
            L_reconstruction_history,
            L_sparsity_history,
            variance_explained_history,
        ) = train_sae(
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

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(loss_history)
        plt.title(f"Layer {layer_number} Total Loss")

        plt.subplot(1, 3, 2)
        plt.plot(L_reconstruction_history)
        plt.title(f"Layer {layer_number} Reconstruction Loss")

        plt.subplot(1, 3, 3)
        plt.plot(L_sparsity_history)
        plt.title(f"Layer {layer_number} Sparsity Loss")

        plt.tight_layout()
        plt.show()

        model_activations.clear_hooks()
