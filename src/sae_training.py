# %%
import sys, os

sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
import einops
import numpy as np
import torch as t

from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
import src.sae_cnn as sae_cnn

device = t.device(
    "cpu"
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
model = sae_cnn.load_interpretable_model()
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
    8: 'conv4a',
    # 9: 'pool4',
    # 10: 'fc1',
    # 11: 'fc2',
    # 12: 'fc3',
    # 13: 'value_fc',
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
for layer_number, layer_name in ordered_layer_names.items():
    print(f"--- Training SAE for layer {layer_number}: {layer_name} ---")
    sae_cnn.train_layer(
        model,
        layer_name,
        layer_number,
        steps=15000000,  
        batch_size=64,
        lr=1e-5,       
        num_envs=8,
        episode_length=150,
        log_freq=1000,
        checkpoint_dir="checkpoints",
        stats_dir="global_stats",
        wandb_project="SAE_training",
        # Potentially add layer-specific wandb run names or tags here
    )
    print(f"--- Finished training SAE for layer {layer_number}: {layer_name} ---")

# %%
# model = helpers.load_interpretable_model()
# model.to(device)
# model.eval()
# # Test data collection
# model_activations = helpers.ModelActivations(model)
# layer_number = 8
# layer_name = ordered_layer_names[layer_number]

# # 1) Hook model to find activation shape:
# model_activations = helpers.ModelActivations(model)
# dummy_obs = t.zeros((1, 64, 64, 3), dtype=t.float32, device=device)
# dummy_obs = einops.rearrange(dummy_obs, "b h w c -> b c h w")
# with t.no_grad():
#     _, act_dict = model_activations.run_with_cache(dummy_obs, layer_name)
# layer_activation = act_dict[layer_name.replace(".", "_")]
# activation_shape = tuple(layer_activation.shape[1:])  # all but batch
# observation_shape = (3, 64, 64)

# # 2) Create replay buffer
# replay_buffer = sae_cnn.ReplayBuffer(
#     capacity=10000,
#     activation_shape=activation_shape,
#     observation_shape=observation_shape,
#     device=device,
#     oversample_large_activations=False
# )

# # 3) Collect data
# sae_cnn.collect_activations_into_replay_buffer(
#     model,
#     model_activations,
#     layer_number=layer_number,
#     replay_buffer=replay_buffer,
#     num_envs=32,
#     episode_length=120,
# )






# # %%
# import autoreload

# autoreload.reload_all()
# # %%
