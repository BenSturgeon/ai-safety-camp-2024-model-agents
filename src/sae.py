#%%
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
import einops
import numpy as np
import torch as t
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
import notebooks.helpers as helpers
import notebooks.heist as heist
import random


device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

ordered_layer_names  = {
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


ENTITY_COLORS = {
    "blue": 0,
    "green": 1,
    "red": 2
}

ENTITY_TYPES = {
    "key": 2,
    "lock": 1,
    "gem": 9,
    "player": 0
}

def constant_lr(*_):
    return 1.0

# %%
@dataclass
class Config:
    # We optimize n_inst models in a single training loop to let us sweep over sparsity or importance
    # curves efficiently. You should treat the number of instances `n_inst` like a batch dimension, 
    # but one which is built into our training setup. Ignore the latter 3 arguments for now, they'll
    # return in later exercises.
    n_inst: int
    n_features: int = 5
    d_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feat_mag_distn: Literal["unif", "jump"] = "unif"

@dataclass
class SAEConfig:
    n_inst: int
    d_in: int
    d_sae: int
    l1_coeff: float = 0.2
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    architecture: Literal["standard", "gated"] = "standard"


class SAE(nn.Module):
    W_enc: Tensor
    _W_dec: Tensor
    b_enc: Tensor
    b_dec: Tensor

    def __init__(self, cfg: SAEConfig, model):
        super().__init__()
        # assert cfg.d_in == model.cfg.d_hidden, "Model's hidden dim doesn't match SAE input dim"
        self.cfg = cfg
        self.model = model.requires_grad_(False)

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_in, cfg.d_sae)))
        )
        self._W_dec = (
            None
            if self.cfg.tied_weights
            else nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_sae, cfg.d_in))))
        )
        self.b_enc = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_sae))
        self.b_dec = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_in))

        self.to(device)

    @property
    def W_dec(self) -> Tensor:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    def W_dec_normalized(self) -> Tensor:
        """Returns decoder weights, normalized over the autoencoder input dimension."""
        return self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    def generate_batch(self, batch_size: int) -> Tensor:
        """
        Generates a batch of hidden activations from our model.
        """
        return einops.einsum(
            self.model.generate_batch(batch_size),
            self.model.W,
            "batch inst feats, inst d_in feats -> batch inst d_in",
        )
    def forward(self, h: Tensor):
        """
        Forward pass on the autoencoder.

        Args:
            h: hidden layer activations of model

        Returns:
            loss_dict: dict of different loss function term values, for every (batch elem, instance)
            loss: scalar total loss (summed over instances & averaged over batch dim)
            acts: autoencoder feature activations
            h_reconstructed: reconstructed autoencoder input
        """
        h_cent = h - self.b_dec

        acts = einops.einsum(
            h_cent, self.W_enc, "batch inst d_in, inst d_in d_sae -> batch inst d_sae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = (
            einops.einsum(
                acts, self.W_dec, "batch inst d_sae, inst d_sae d_in -> batch inst d_in"
            )
            + self.b_dec
        )

        # Compute loss terms
        L_reconstruction = (h_reconstructed - h).pow(2).mean(-1)
        L_sparsity = acts.abs().sum(-1)
        loss_dict = {
            "L_reconstruction": L_reconstruction,
            "L_sparsity": L_sparsity,
        }
        loss = (L_reconstruction + self.cfg.l1_coeff * L_sparsity).mean(0).sum()

        return loss_dict, loss, acts, h_reconstructed

    def optimize(
            self,
            batch_size: int = 1024,
            steps: int = 10_000,
            log_freq: int = 50,
            lr: float = 1e-3,
            lr_scale: Callable[[int, int], float] = constant_lr,
            resample_method: Literal["simple", "advanced", None] = None,
            resample_freq: int = 2500,
            resample_window: int = 500,
            resample_scale: float = 0.5,
        ):
            """
            Optimizes the autoencoder using the given hyperparameters.

            Args:
                model:              we reconstruct features from model's hidden activations
                batch_size:         size of batches we pass through model & train autoencoder on
                steps:              number of optimization steps
                log_freq:           number of optimization steps between logging
                lr:                 learning rate
                lr_scale:           learning rate scaling function
                resample_method:    method for resampling dead latents
                resample_freq:      number of optimization steps between resampling dead latents
                resample_window:    number of steps needed for us to classify a neuron as dead
                resample_scale:     scale factor for resampled neurons

            Returns:
                data_log:               dictionary containing data we'll use for visualization
            """
            assert resample_window <= resample_freq

            optimizer = t.optim.Adam(list(self.parameters()), lr=lr, betas=(0.0, 0.999))
            frac_active_list = []
            progress_bar = tqdm(range(steps))

            # Create lists to store data we'll eventually be plotting
            data_log = {"steps": [], "W_enc": [], "W_dec": [], "frac_active": []}

            for step in progress_bar:
                # Resample dead latents
                if (resample_method is not None) and ((step + 1) % resample_freq == 0):
                    frac_active_in_window = t.stack(frac_active_list[-resample_window:], dim=0)
                    if resample_method == "simple":
                        self.resample_simple(frac_active_in_window, resample_scale)
                    elif resample_method == "advanced":
                        self.resample_advanced(frac_active_in_window, resample_scale, batch_size)

                # Update learning rate
                step_lr = lr * lr_scale(step, steps)
                for group in optimizer.param_groups:
                    group["lr"] = step_lr

                # Get a batch of hidden activations from the model
                with t.inference_mode():
                    h = self.generate_batch(batch_size)

                # Optimize
                loss_dict, loss, acts, _ = self.forward(h)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Normalize decoder weights by modifying them inplace (if not using tied weights)
                if not self.cfg.tied_weights:
                    self.W_dec.data = self.W_dec_normalized

                # Calculate the mean sparsities over batch dim for each feature
                frac_active = (acts.abs() > 1e-8).float().mean(0)
                frac_active_list.append(frac_active)

                # Display progress bar, and append new values for plotting
                if step % log_freq == 0 or (step + 1 == steps):
                    progress_bar.set_postfix(
                        lr=step_lr,
                        frac_active=frac_active.mean().item(),
                        **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
                    )
                    data_log["W_enc"].append(self.W_enc.detach().cpu().clone())
                    data_log["W_dec"].append(self.W_dec.detach().cpu().clone())
                    data_log["frac_active"].append(frac_active.detach().cpu().clone())
                    data_log["steps"].append(step)

            return data_log

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float,
        resample_scale: float,
    ) -> None:
        """
        Resamples dead latents, by modifying the model's weights and biases inplace.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (d_in,), and normalize these vectors
            - Set new values of W_dec and W_enc to be these normalized vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron

        This function performs resampling over all instances at once, using batched operations.
        """
        # Get a tensor of dead latents
        dead_latents_mask = (frac_active_in_window < 1e-8).all(dim=0)  # [instances d_sae]
        n_dead = int(dead_latents_mask.int().sum().item())

        # Get our random replacement values of shape [n_dead d_in], and scale them
        replacement_values = t.randn((n_dead, self.cfg.d_in), device=self.W_enc.device)
        replacement_values_normed = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )

        # Change the corresponding values in W_enc, W_dec, and b_enc
        self.W_enc.data.transpose(-1, -2)[dead_latents_mask] = resample_scale * replacement_values_normed
        self.W_dec.data[dead_latents_mask] = replacement_values_normed
        self.b_enc.data[dead_latents_mask] = 0.0


# %%
# d_hidden = d_in = 2
# n_features = d_sae = 5
# n_inst = 8

# cfg = Config(n_inst=n_inst, n_features=n_features, d_hidden=d_hidden)


# model_path = "../model_interpretable.pt"
# model = helpers.load_interpretable_model(model_path=model_path)
# # model = Model(cfg=cfg, device=device)
# # model.optimize(steps=10_000)

# sae = SAE(cfg=SAEConfig(n_inst=n_inst, d_in=d_in, d_sae=d_sae), model=model)



# data_log = sae.optimize(steps=25_000)
# %%

def generate_batch_activations(model, layer_number, batch_size=32):
    # Create the environment
    venv = heist.create_venv(num=1, num_levels=batch_size, start_level=random.randint(1, 100000))
    
    # Get the layer name
    layer_name = ordered_layer_names[layer_number]
    renamed_layer = helpers.rename_path(layer_name)
    
    # Initialize ModelActivations
    model_activations = helpers.ModelActivations(model)
    
    # Generate batch of observations
    observations = []
    for _ in range(batch_size):
        obs = venv.reset()
        done = False
        episode_observations = []
        while not done:
            episode_observations.append(helpers.observation_to_rgb(obs))
            observations.extend(episode_observations)
            state = heist.state_from_venv(venv, 0)

            observation = np.squeeze(observation)
            observation = np.transpose(observation, (1, 2, 0))
            converted_obs = helpers.observation_to_rgb(observation)
            action = helpers.generate_action(model, converted_obs, is_procgen_env=True)

            observation, reward, done, info = venv.step(action)
            total_reward += reward
    
    # Convert observations to tensor
    obs_tensor = t.stack([t.tensor(obs) for obs in observations]).to(device)
    
    # Get activations
    model_activations.clear_hooks()
    _, activations = model_activations.run_with_cache(obs_tensor, [layer_name])
    
    return activations[renamed_layer]

# Example usage:
model_path = "../model_interpretable.pt"
model = helpers.load_interpretable_model(model_path=model_path)
layer_number = 8  # Example layer number
batch_activations = generate_batch_activations(model, layer_number)
print(f"Shape of batch activations: {batch_activations.shape}")