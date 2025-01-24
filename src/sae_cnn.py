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
        return t.device("cpu")
    else:
        return t.device("cpu")

device = get_device()

################################################################################
# LAYER INFO & HYPERPARAMETERS
################################################################################

"""
We define the layer ordering (used for hooking up to your RL CNN),
plus a dictionary that maps each layer to a short string (useful for hooking)
and we define layer-specific hyperparams for the SAE:
 - expansion_factor: How much bigger the hidden dimension is relative to input dim
 - l1_coeff: L1 coefficient controlling the sparsity pressure
You can adjust these as needed.
"""

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

# High-level: "conv", "pool", "fc", or "dropout"
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


# You can fine-tune these hyperparameters to taste. Example schedule:
#  - Early conv layers: expansion_factor ~2 or 4, l1_coeff ~1e-7 to 1e-6
#  - Later conv or FC layers: expansion_factor ~4 to 8, l1_coeff ~5e-6 to 1e-5
#  - Pooling/Dropout layers often don't have typical learned weights. We might skip them or use small expansions

layer_sae_hparams = {
    "conv1a":   {"expansion_factor": 2, "l1_coeff": 1e-7},
    "conv2a":   {"expansion_factor": 4, "l1_coeff": 5e-7},
    "conv2b":   {"expansion_factor": 4, "l1_coeff": 5e-7},
    "conv3a":   {"expansion_factor": 4, "l1_coeff": 5e-6},
    "conv4a":   {"expansion_factor": 6, "l1_coeff": 1e-5},
    "fc1":      {"expansion_factor": 4, "l1_coeff": 1e-5},
    "fc2":      {"expansion_factor": 4, "l1_coeff": 1e-5},
    "fc3":      {"expansion_factor": 4, "l1_coeff": 1e-5},
    "value_fc": {"expansion_factor": 4, "l1_coeff": 1e-5},
    "pool1":    {"expansion_factor": 2, "l1_coeff": 1e-7},
    "pool2":    {"expansion_factor": 2, "l1_coeff": 1e-7},
    "pool3":    {"expansion_factor": 2, "l1_coeff": 1e-7},
    "pool4":    {"expansion_factor": 2, "l1_coeff": 1e-7},
    "dropout_conv": {"expansion_factor": 2, "l1_coeff": 1e-7},
    "dropout_fc":   {"expansion_factor": 2, "l1_coeff": 1e-7},
}


def get_layer_hyperparameters(layer_name: str):
    """
    Returns (expansion_factor, l1_coeff) for the given layer name
    using layer_sae_hparams. You can customize or override here.
    """
    if layer_name not in layer_sae_hparams:
        return (2, 1e-6)  # default fallback
    ef = layer_sae_hparams[layer_name]["expansion_factor"]
    lc = layer_sae_hparams[layer_name]["l1_coeff"]
    return (ef, lc)


################################################################################
# REPLAY BUFFER
################################################################################

class ReplayBuffer:
    """
    Stores activations + observations, then sample them in random mini-batches.

    Optional: oversample large activations. 
    - If 'oversample_large_activations' is True, we sample from the buffer
      with probabilities proportional to the L2 norm of activation examples.
    """
    def __init__(
        self, 
        capacity, 
        activation_shape, 
        observation_shape, 
        device,
        oversample_large_activations=True
    ):
        self.capacity = capacity
        self.activation_shape = activation_shape  
        self.observation_shape = observation_shape
        self.device = device
        self.activations = t.empty((capacity, *activation_shape), device=device)
        self.observations = t.empty((capacity, *observation_shape), device=device)
        self.norms = t.empty((capacity,), device=device)  # for oversampling
        self.size = 0
        self.position = 0
        self.oversample = oversample_large_activations

    def add(self, activations, observations):
        """
        Add a single (activation, observation) to the buffer.
        You could also add in batch form if desired.
        """
        act = activations.to(self.device)
        obs = observations.to(self.device)

        self.activations[self.position] = act
        self.observations[self.position] = obs

        if self.oversample:
            # store L2 norm for oversampling
            self.norms[self.position] = act.float().norm(p=2).item()

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample random or norm-proportional if oversample is True
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        if self.oversample:
            # Weighted sampling by norms
            norms_np = self.norms[:self.size].detach().cpu().numpy()
            probs = norms_np / (norms_np.sum() + 1e-8)
            indices = np.random.choice(self.size, size=batch_size, replace=True, p=probs)
        else:
            indices = np.random.choice(self.size, size=batch_size, replace=True)

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
        if self.oversample:
            self.norms = t.empty((self.capacity,), device=self.device)


################################################################################
# SPARSE AUTOENCODER ARCHITECTURE
################################################################################

@dataclass
class SAEConfig:
    """
    Basic configuration for the SAE.
    """
    d_in: int = None
    d_hidden: int = 128
    l1_coeff: float = 1e-5
    tied_weights: bool = False
    # set to True if you want to experiment with decoder norm scaling
    scale_by_decoder_norm: bool = True


class ConvSAE(nn.Module):
    """
    A "convolutional" SAE that uses 1x1 convs to encode & decode spatial features.
    Preserves shape (B, C, H, W) throughout, so no flattening is needed.
    """
    def __init__(self, in_channels, hidden_channels, l1_coeff=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.l1_coeff = l1_coeff

        # Encoder: 1x1 conv transforms (C -> hidden_channels)
        self.conv_enc = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True).to(device)
        nn.init.kaiming_uniform_(self.conv_enc.weight, nonlinearity="relu")

        # Decoder: 1x1 conv transforms (hidden_channels -> C)
        self.conv_dec = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True).to(device)
        nn.init.kaiming_uniform_(self.conv_dec.weight, nonlinearity="relu")

    def forward(self, x, current_l1_coeff=None):
        """
        Args:
          x: (B, in_channels, H, W)
          current_l1_coeff: if you have a warmup or dynamic L1, pass it here.
        Returns:
          (sparsity_loss, recon_loss, acts, x_reconstructed)
        """
        if current_l1_coeff is None:
            current_l1_coeff = self.l1_coeff

        # Encode
        z = self.conv_enc(x)            # shape: (B, hidden_channels, H, W)
        acts = F.relu(z)                # shape: (B, hidden_channels, H, W)

        # Decode
        x_hat = self.conv_dec(acts)     # shape: (B, in_channels, H, W)

        # Reconstruction Loss (pixelwise MSE)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        # L1 penalty on hidden activations
        l1_loss = acts.abs().mean() * current_l1_coeff
        return l1_loss, recon_loss, acts, x_hat


################################################################################
# ACTIVATION COLLECTION & SAMPLING
################################################################################

def collect_activations_into_replay_buffer(
    model,
    model_activations,
    layer_number,
    replay_buffer,
    num_envs=8,
    episode_length=150,
):
    """
    Roll out episodes in parallel and store all activations/observations
    into the replay buffer. Continues until buffer is full or we do 1 episode/env.
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

    layer_name = ordered_layer_names[layer_number]

    while not replay_buffer.is_full():
        observations = t.tensor(obs, dtype=t.float32).to(device)
        observations = einops.rearrange(observations, "b h w c -> b c h w")

        with t.no_grad():
            outputs, activations = model_activations.run_with_cache(observations, layer_name)

        layer_activation = activations[layer_name.replace(".", "_")]

        # If the layer_activation is a tuple, handle each env separately
        # (Some conv nets have multi-part returns from forward hooks)
        if isinstance(layer_activation, tuple):
            for i in range(num_envs):
                activation_lists[i].append(layer_activation[i].cpu())
                observation_lists[i].append(observations[i].cpu())
        else:
            for i in range(num_envs):
                activation_lists[i].append(layer_activation[i].cpu())
                observation_lists[i].append(observations[i].cpu())

        # pick actions (multinomial from policy)
        logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        probabilities = t.softmax(logits, dim=-1)
        actions = t.multinomial(probabilities, num_samples=1).squeeze(1).cpu().numpy()

        obs, rewards, dones_env, infos = venv.step(actions)
        steps += 1

        for i, done in enumerate(dones_env):
            if done or steps[i] >= episode_length:
                # push entire trajectory from this environment i
                acts_tensor = t.stack(activation_lists[i], dim=0)
                obs_tensor = t.stack(observation_lists[i], dim=0)
                for t_step in range(acts_tensor.size(0)):
                    replay_buffer.add(acts_tensor[t_step], obs_tensor[t_step])

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
    Simple multi-env rollout for a single batch of activations, rather
    than filling a replay buffer. Typically used for shape sampling or debugging.
    """
    venv = heist.create_venv(
        num=num_envs, num_levels=0, start_level=random.randint(1, 100000)
    )
    obs = venv.reset()

    activation_buffer = []
    obs_buffer = []

    layer_name = ordered_layer_names[layer_number]
    steps = 0

    while len(activation_buffer) * num_envs < batch_size:
        observations = t.tensor(obs, dtype=t.float32).to(device)
        observations = einops.rearrange(observations, "b h w c -> b c h w")

        with t.no_grad():
            outputs, activations = model_activations.run_with_cache(observations, layer_name)
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
        print(f"Warning: Collected only {activations_tensor.size(0)} activations, expected {batch_size}.")

    observations_tensor = t.cat(obs_buffer, dim=0)[:batch_size]

    return activations_tensor, observations_tensor, None


def find_latest_checkpoint(checkpoint_dir):
    """
    Returns (path_to_latest_checkpoint, step_number) or (None, None) if not found
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "sae_checkpoint_step_*.pt"))
    if not checkpoint_files:
        return None, None
    steps = [
        int(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
        for f in checkpoint_files
    ]
    latest_step = max(steps)
    latest_checkpoint = os.path.join(checkpoint_dir, f"sae_checkpoint_step_{latest_step}.pt")
    return latest_checkpoint, latest_step


def get_module_by_path(model, layer_path):
    """
    Recursively get a submodule from a model using dot/bracket notation.
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
    Optional utility to quickly measure average reward across multiple test episodes.
    """
    venv = heist.create_venv(
        num=num_envs, num_levels=0, start_level=random.randint(1, 100000)
    )
    model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
    total_rewards, frames, observations = helpers.run_episode_and_save_as_gif(
        venv, model, filepath='episode_mod.gif', 
        save_gif=False, episode_timeout=episode_length, 
        is_procgen_env=True
    )
    # total_rewards will be the sum for that run; you can gather stats, etc.
    return total_rewards


################################################################################
# TRAINING LOOP
################################################################################

def train_sae(
    sae,
    model,
    model_activations,
    layer_number,
    layer_name,
    ordered_layer_names, 
    batch_size=128,
    steps=200,
    lr=1e-3,
    num_envs=8,
    episode_length=150,
    log_freq=10,
    checkpoint_dir="checkpoints",
    stats_dir="global_stats",
    wandb_project="SAE_training",
    oversample_large_activations=False,
    l1_warmup_steps=10000,
):
    """
    Main training loop for the Sparse Autoencoder (SAE).
    1) Creates or loads from checkpoint
    2) Creates replay buffer
    3) Periodically collects data from environment
    4) Minimizes reconstruction + (downstream reconstruction) + KL(prob) + L1
    5) Logs to wandb
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


    # We'll do an initial sample to figure out d_in, etc:
    sample_activations, sample_observations, _ = generate_batch_activations_parallel(
        model, model_activations, layer_number,
        batch_size=batch_size,
        num_envs=num_envs,
        episode_length=episode_length,
    )
    activation_shape = sample_activations.shape[1:]
    observation_shape = sample_observations.shape[1:]

    # Build a replay buffer with some large capacity
    buffer_capacity = batch_size * 50
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity,
        activation_shape=activation_shape,
        observation_shape=observation_shape,
        device=device,
        oversample_large_activations=oversample_large_activations
    )

    # Collect some data so we can start sampling right away
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)

    sae = sae.to(device)
    for param in sae.parameters():
        param.data = param.data.to(device)

    
    while replay_buffer.size < replay_buffer.capacity * 0.2:
        collect_activations_into_replay_buffer(
            model,
            model_activations,
            layer_number,
            replay_buffer,
            num_envs=num_envs,
            episode_length=episode_length,
        )

    # Precompute buffer mean/std for normalizing
    all_activations = replay_buffer.activations[: replay_buffer.size]
    buffer_mean = t.mean(all_activations, dim=0)
    buffer_std = t.std(all_activations, dim=0) + 1e-8

    progress_bar = tqdm(range(start_step, steps), desc=f"Training {layer_identifier}")
    for step in progress_bar:
        # Periodically refill buffer
        if step % 250 == 0:
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
            buffer_std = t.std(all_activations, dim=0) + 1e-8

        try:
            activations_unflat, observations = replay_buffer.sample(batch_size)
        except Exception as e:
            print(f"Error sampling from buffer: {e}")
            continue

        # Normalize
        activations = (activations_unflat - buffer_mean) / buffer_std
        # if len(activations_unflat.shape) > 2:
        #     activations = activations_unflat.view(batch_size, -1).to(device)
        # else:
        #     activations = activations_unflat.to(device)
        observations = observations.to(device)

        # L1 warmup schedule
        if step < l1_warmup_steps:
            current_l1_coeff = (step / l1_warmup_steps) * sae.l1_coeff
        else:
            current_l1_coeff = sae.l1_coeff

        # Forward pass
        optimizer.zero_grad()
        sparsity_loss, L_reconstruction, acts, h_reconstructed = sae(
            activations, current_l1_coeff
        )

        # [Optional] Compute "downstream" losses for logit or next-layer reconstruction
        # For example, hooking the model with replaced activations. 
        # We'll skip that here to keep it simpler, but you could replicate 
        # the approach in your existing code if desired.

        # Just use standard autoencoder losses:
        total_loss = (sparsity_loss + L_reconstruction)

        total_loss.backward()
        optimizer.step()

        # Some stats
        with t.no_grad():
            # 1) Fraction active: flatten acts just for threshold counting
            b = acts.shape[0]
            acts_flat = acts.view(b, -1)  # (batch_size, hidden_channels*H*W)
            frac_active = (acts_flat > 1e-5).float().mean().item()

            # 2) Flatten both for MSE or do a sum over dims
            diff = (activations - h_reconstructed)  # both (B, C, H, W)

            # MSE
            diff_flat = diff.view(b, -1)        # shape (batch_size, C*H*W)
            orig_flat = activations.view(b, -1) # shape (batch_size, C*H*W)

            numer = t.sum(diff_flat**2, dim=1)
            denom = t.sum(orig_flat**2, dim=1) + 1e-8
            var_explained = (1 - numer/denom).mean().item()

        if step % log_freq == 0 or step == steps - 1:
            progress_bar.set_postfix({
                "step": step,
                "sparsity_loss": f"{sparsity_loss.item():.4e}",
                "recon_loss": f"{L_reconstruction.item():.4e}",
                "frac_active": f"{frac_active:.3f}",
                "var_expl": f"{var_explained:.3f}",
            })

            wandb.log({
                "step": step,
                "total_loss": total_loss.item(),
                "sparsity_loss": sparsity_loss.item(),
                "reconstruction_loss": L_reconstruction.item(),
                "frac_active": frac_active,
                "variance_explained": var_explained,
                "l1_coeff_current": current_l1_coeff,
            }, step=step)

            if step % 1000 == 0:
                r_no_sae = run_test_episodes(model, num_envs=8, episode_length=150)
                # temporarily replace the layer with SAE
                handle = replace_layer_with_sae(model, sae, layer_number)
                model.to(device)
                r_sae = run_test_episodes(model, num_envs=8, episode_length=150)
                handle.remove()
                wandb.log({
                    "test_reward_no_sae": r_no_sae,
                    "test_reward_with_sae": r_sae,
                }, step=step)

        # Checkpoint save
        if (step + 1) % 1000 == 0 or step == steps - 1:
            checkpoint_path = os.path.join(layer_checkpoint_dir, f"sae_checkpoint_step_{step+1}.pt")
            t.save({
                "step": step,
                "model_state_dict": sae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss.item(),
            }, checkpoint_path)

    wandb.finish()
    return


################################################################################
# TOP-LEVEL HELPER FOR TRAINING A LAYER
################################################################################

def train_layer(
    model,
    layer_name,
    layer_number,
    steps=10000,
    batch_size=128,
    lr=1e-3,
    num_envs=64,
    episode_length=150,
    log_freq=10,
    checkpoint_dir="checkpoints",
    stats_dir="global_stats",
    wandb_project="SAE_training",
    oversample_large_activations=True,
    l1_warmup_steps=10000,
):
    """
    Convenience wrapper to train a single layer's SAE
    from scratch or resume from checkpoint.

    Example usage:
        train_layer(model, "conv2a", layer_number=3, steps=20000, ...)
    """
    print(f"\n=== Training SAE for Layer: {layer_name} (#{layer_number}) ===")

    expansion_factor, l1_coeff = get_layer_hyperparameters(layer_name)

    # Get sample to figure out d_in
    model_activations = helpers.ModelActivations(model)
    sample_acts, _, _ = generate_batch_activations_parallel(
        model, model_activations, layer_number,
        batch_size=batch_size,
        num_envs=num_envs,
        episode_length=episode_length,
    )
    d_in = sample_acts.shape[1]
    # Build SAE with those hyperparams
    d_hidden = int(expansion_factor * d_in)
    sae_cfg = SAEConfig(
        d_in=d_in,
        d_hidden=d_hidden,
        l1_coeff=l1_coeff,
        tied_weights=False,
        scale_by_decoder_norm=True  # switch to True if you want that experiment
    )
    sae = ConvSAE(
    in_channels=sae_cfg.d_in,
    hidden_channels=sae_cfg.d_hidden,
    l1_coeff=sae_cfg.l1_coeff
)
    sae.to(device)

    train_sae(
        sae=sae,
        model=model,
        model_activations=model_activations,
        layer_number=layer_number,
        layer_name=layer_name,
        ordered_layer_names=ordered_layer_names,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        num_envs=num_envs,
        episode_length=episode_length,
        log_freq=log_freq,
        checkpoint_dir=checkpoint_dir,
        stats_dir=stats_dir,
        wandb_project=wandb_project,
        oversample_large_activations=oversample_large_activations,
        l1_warmup_steps=l1_warmup_steps,
    )

    model_activations.clear_hooks()
    print(f"Done training SAE for layer {layer_name}.\n")


################################################################################
# OPTIONAL: GLOBAL STATS
################################################################################
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
    num_envs=128,
    save_dir="global_stats",
):
    """
    Computes and saves the mean/std for the given layer’s activations
    across num_samples images from the environment. 
    """
    os.makedirs(save_dir, exist_ok=True)
    model_activations = helpers.ModelActivations(model)
    activation_samples = []
    total_samples = 0

    with tqdm(total=num_samples, desc=f"Computing global stats for {layer_name}") as pbar:
        while total_samples < num_samples:
            acts, _, _ = generate_batch_activations_parallel(
                model, model_activations, layer_number,
                batch_size=batch_size, num_envs=num_envs
            )
            activation_samples.append(acts)
            total_samples += acts.size(0)
            pbar.update(acts.size(0))

    all_acts = t.cat(activation_samples, dim=0)[:num_samples]
    global_mean = all_acts.mean(dim=0, keepdim=True)
    global_std = all_acts.std(dim=0, keepdim=True) + 1e-8

    t.save({"mean": global_mean, "std": global_std},
           os.path.join(save_dir, f"layer_{layer_number}_{layer_name}_stats.pt"))
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
