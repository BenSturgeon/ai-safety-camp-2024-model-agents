# %%

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
import sae

import autoreload

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda" if t.cuda.is_available() else "cpu"
)
sys.path.append("../")  # Adjust the path if necessary to import your modules

from src.utils import helpers, heist

# Set device
# device = t.device("cuda" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

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


def constant_lr(*_):
    return 1.0


# Load the model
model = sae.load_interpretable_model()
model.to(device)
model.eval()  # Set model to evaluation mode

ordered_layer_names = {
    # 1: 'conv1a',
    # 2: 'pool1',
    # 3: 'conv2a',
    # 4: 'conv2b',
    # 5: 'pool2',
    # 6: "conv3a",
    # 7: 'pool3',
    # 8: 'conv4a',
    # 9: 'pool4',
    10: 'fc1',
    11: 'fc2',
    12: 'fc3',
    13: 'value_fc',
    # 14: 'dropout_conv',
    # 15: 'dropout_fc'
}
# Execute training for all layers
# sae.train_all_layers(
#     model=model,
#     ordered_layer_names=ordered_layer_names,
#     layer_types=layer_types,
#     checkpoint_dir='checkpoints',
#     wandb_project="SAE_training",
#     steps_per_layer=5000,  # Reduced steps
#     batch_size=64,        # Adjust based on your hardware
#     lr=2e-4,
#     num_envs=4,           # Number of parallel environments
#     episode_length=150,
#     log_freq=10,
# )


# def compute_global_stats_for_all_layers(
#     model,
#     ordered_layer_names,
#     num_samples_per_layer=10000,
#     batch_size=64,
#     num_envs=8,
#     save_dir="global_stats",
# ):
#     for layer_number, layer_name in ordered_layer_names.items():
#         sae.compute_and_save_global_stats(
#             model,
#             layer_number,
#             layer_name,
#             num_samples=num_samples_per_layer,
#             batch_size=batch_size,
#             num_envs=num_envs,
#             save_dir=save_dir,
#         )


# # Load the model
# model = sae.load_interpretable_model()
# model.to(device)
# model.eval()

# # Compute global statistics for all layers
# compute_global_stats_for_all_layers(
#     model,
#     ordered_layer_names,
#     num_samples_per_layer=10000,
#     batch_size=64,
#     num_envs=8,
#     save_dir="global_stats",
# )
# %%
layer_number = 10
layer_name = ordered_layer_names[layer_number]
sae.train_layer(
    model,
    layer_name,
    layer_number,
    layer_types,
    checkpoint_dir="checkpoints",
    stats_dir="global_stats",
    wandb_project="SAE_training",
    steps=210050,
    batch_size=128,
    lr=1e-4,
    num_envs=8,
    episode_length=150,
    log_freq=100,
)

# %%
model = helpers.load_interpretable_model()
model.to(device)
model.eval()
# Test data collection
model.eval()
layer_number = 6
model_activations = helpers.ModelActivations(model)
replay_buffer = sae.ReplayBuffer(capacity=10000)
sae.generate_episodes_and_fill_buffer(
    model,
    model_activations,
    layer_number,
    replay_buffer,
    num_envs=32,
    max_steps_per_episode=120,
)

print(f"Replay buffer size: {len(replay_buffer)}")


# %%
import autoreload

autoreload.reload_all()
# %%
