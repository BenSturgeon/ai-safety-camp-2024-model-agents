import torch.nn as nn
import torch as t
import imageio
from torch.nn import functional as F
import re
import glob
import os

import gym

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import math
import imageio
import random
import sys

sys.path.append("../")  # This is added so we can import from the source folder
from policies_impala import ImpalaCNN
from interpretable_impala import CustomCNN as interpretable_CNN

# from src.policies_modified import ImpalaCNN
from visualisation_functions import *

from utils import heist


def get_device():
    if t.cuda.is_available():
        return t.device("cuda")
    elif hasattr(t.backends, "mps") and t.backends.mps.is_available():
        return t.device("cpu")
    else:
        return t.device("cpu")

device = get_device()

ordered_layer_names = {
    0: "conv_seqs",
    1: "conv_seqs.0",
    2: "conv_seqs.0.conv",
    3: "conv_seqs.0.max_pool2d",
    4: "conv_seqs.0.res_block0",
    5: "conv_seqs.0.res_block0.conv0",
    6: "conv_seqs.0.res_block0.conv1",
    7: "conv_seqs.0.res_block1",
    8: "conv_seqs.0.res_block1.conv0",
    9: "conv_seqs.0.res_block1.conv1",
    10: "conv_seqs.1",
    11: "conv_seqs.1.conv",
    12: "conv_seqs.1.max_pool2d",
    13: "conv_seqs.1.res_block0",
    14: "conv_seqs.1.res_block0.conv0",
    15: "conv_seqs.1.res_block0.conv1",
    16: "conv_seqs.1.res_block1",
    17: "conv_seqs.1.res_block1.conv0",
    18: "conv_seqs.1.res_block1.conv1",
    19: "conv_seqs.2",
    20: "conv_seqs.2.conv",
    21: "conv_seqs.2.max_pool2d",
    22: "conv_seqs.2.res_block0",
    23: "conv_seqs.2.res_block0.conv0",
    24: "conv_seqs.2.res_block0.conv1",
    25: "conv_seqs.2.res_block1",
    26: "conv_seqs.2.res_block1.conv0",
    27: "conv_seqs.2.res_block1.conv1",
    28: "hidden_fc",
    29: "logits_fc",
    30: "value_fc",
}


def get_ordered_layer_names():
    return ordered_layer_names


class ModelActivations:
    def __init__(self, model):
        self.activations = {}
        self.model = model
        self.hooks = []  # To keep track of hooks

    def clear_hooks(self):
        # Remove all previously registered hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def get_activation(self, name):
        def hook(model, input, output):
            processed_output = []
            for item in output:
                if isinstance(item, t.Tensor):
                    processed_output.append(item.detach())
                elif isinstance(item, t.distributions.Categorical):
                    processed_output.append(item.logits.detach())
                else:
                    processed_output.append(item)
            self.activations[name] = t.stack(processed_output) if len(processed_output) > 1 else processed_output[0]

        return hook

    def register_hook_by_path(self, path, name):
        elements = path.split(".")
        model = self.model
        for i, element in enumerate(elements):
            if "[" in element:
                base, index = element.replace("]", "").split("[")
                index = int(index)
                model = getattr(model, base)[index]
            else:
                model = getattr(model, element)
            if i == len(elements) - 1:
                hook = model.register_forward_hook(self.get_activation(name))
                self.hooks.append(hook)  # Keep track of the hook

    def run_with_cache(self, input, layer_paths):
        self.clear_hooks()  # Clear any existing hooks
        self.activations = {}  # Reset activations
        if isinstance(layer_paths, str):
            # Handle case where layer_paths is a single string
            self.register_hook_by_path(layer_paths, layer_paths.replace(".", "_"))
        elif isinstance(layer_paths, list):
            # Handle case where layer_paths is a list

            for path in layer_paths:
                self.register_hook_by_path(path, path.replace(".", "_"))
        else:
            raise ValueError("layer_paths must be a string or a list of strings")
        if not isinstance(input, t.Tensor):
            input = t.tensor(input, dtype=t.float32)
        input = input.to(device)
        output = self.model(input)
        return output, self.activations


def generate_action(model, obs, is_procgen_env=True):
    with t.no_grad():
        if len(obs.shape) == 3:
            obs = np.expand_dims(obs, axis=0).to(device)
        if isinstance(obs, np.ndarray):
            obs = t.from_numpy(obs).float().to(device)
        elif isinstance(obs, t.Tensor):
            obs = obs.float().to(device)
        model.to(device)
        outputs = model(obs)
    logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    actions = t.multinomial(probabilities, num_samples=1).squeeze(-1)
    if is_procgen_env:
        return actions.cpu().numpy()
    return actions.cpu()


def find_latest_model_checkpoint(base_dir):
    """Finds the latest base model checkpoint based on step count or modification time."""
    if not os.path.isdir(base_dir):
        print(f"Warning: Model checkpoint base directory not found: {base_dir}")
        return None

    # Search potentially nested directories, e.g., models/maze_I/
    checkpoints = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
    if not checkpoints:
        print(f"Warning: No .pt files found in {base_dir} or its subdirectories.")
        return None

    latest_checkpoint = None
    max_steps = -1

    # Adjust regex if model checkpoints have a different naming pattern
    # Example: checkpoint_78643200_steps.pt
    step_pattern = re.compile(r"checkpoint_(\d+)_steps\.pt$")

    for ckpt_path in checkpoints:
        match = step_pattern.search(os.path.basename(ckpt_path))
        if match:
            steps = int(match.group(1))
            if steps > max_steps:
                max_steps = steps
                latest_checkpoint = ckpt_path

    # Fallback if no files match the step pattern
    if latest_checkpoint is None:
         print(f"Warning: Could not parse step count from model checkpoint names in {base_dir}. Using file with latest modification time.")
         try:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
         except Exception as e:
             print(f"Error finding latest model checkpoint by time: {e}")
             latest_checkpoint = checkpoints[-1]

    return latest_checkpoint

def load_interpretable_model(
    ImpalaCNN=interpretable_CNN, model_path="../model_interpretable.pt"
):
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
    model = ImpalaCNN(observation_space, action_space)
    model.load_from_file(model_path, device="cpu")
    return model


def load_model(ImpalaCNN=ImpalaCNN, model_path="../model_1400_latest.pt"):
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
    model = ImpalaCNN(observation_space, action_space)
    model.load_from_file(model_path, device="cpu")
    return model


def get_model_layer_names(model):
    layer_names = [
        name for name, _ in model.named_modules() if isinstance(_, nn.Module)
    ]
    return layer_names[1 : len(layer_names)]


def plot_activations_for_layers(activations, layer_paths, save_filename_prefix=None):
    for layer_name in layer_paths:
        print(layer_name)
        # Check if the specified layer's activations are available
        if layer_name not in activations:
            print(f"No activations found for layer: {layer_name}")
            continue

        # Extract the activation tensor for the specified layer from the tuple
        activation_tensor = activations[layer_name][0]

        # The tensor is 3-dimensional [channels, height, width]
        num_activations = activation_tensor.shape[0]  # Number of activation maps

        # Calculate grid size
        grid_size = math.ceil(math.sqrt(num_activations))

        # Create a figure with dynamic subplots based on the number of activations
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
        )
        if grid_size == 1:
            axes = np.array([[axes]])  # Ensure axes can be indexed with two dimensions

        # Initialize an index for activation maps
        activation_idx = 0

        for i in range(grid_size):
            for j in range(grid_size):
                ax = axes[i, j]

                # Plot the activation map if we haven't gone through all of them yet
                if activation_idx < num_activations:
                    ax.imshow(
                        activation_tensor[activation_idx, :, :],
                        cmap="viridis",
                        aspect="auto",
                    )
                    ax.set_title(f"Filter {activation_idx+1} {layer_name}", fontsize=8)
                    activation_idx += 1
                else:
                    ax.axis("off")  # Hide axes without data

                ax.axis("off")  # Hide axes for all plots for a cleaner look

        plt.tight_layout()

        # Save or show the plot
        if save_filename_prefix:
            save_filename = f"{save_filename_prefix}_{layer_name}.png"
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()


def plot_activations_for_channel(activations, layer_name, save_filename_prefix=None):
    # Check if the specified layer's activations are available
    if layer_name not in activations:
        print(f"No activations found for layer: {layer_name}")
        return None

    # Extract the activation tensor for the specified layer from the tuple
    activation_tensor = activations[layer_name][0]

    # The tensor is 3-dimensional [channels, height, width]
    num_activations = activation_tensor.shape[0]  # Number of activation maps

    for activation_idx in range(num_activations):
        # Create a new figure for each channel
        fig, ax = plt.subplots(figsize=(3, 3))

        # Plot the activation map
        im = ax.imshow(
            activation_tensor[activation_idx, :, :], cmap="viridis", aspect="auto"
        )
        # ax.set_title(f'Filter {activation_idx} {layer_name}', fontsize=10)
        ax.axis("off")  # Hide axes for a cleaner look

        # Add colorbar
        # plt.colorbar(im, ax=ax)

        plt.tight_layout()

        # Save or show the plot
        if save_filename_prefix:
            save_filename = (
                f"{save_filename_prefix}_{layer_name}_filter{activation_idx+1}.png"
            )
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()

    # Save or show the plot
    if save_filename_prefix:
        save_filename = f"{save_filename_prefix}_{layer_name}.png"
        plt.savefig(save_filename)
        plt.close()
    else:
        plt.show()


def plot_activations_for_layers_rb_max(
    activations, layer_paths=None, save_filename_prefix=None, plot_scale_max=1
):
    plt.rcParams["image.cmap"] = (
        "RdBu_r"  # Set the reversed default colormap to 'RdBu_r' for all plots
    )

    if layer_paths is None:
        layer_paths = list(activations.keys())
    for layer_name in layer_paths:
        if layer_name not in list(activations.keys()):
            print(f"No activations found for layer: {layer_name}")
            continue

        # Extract the activation tensor for the specified layer from the tuple
        if isinstance(activations[layer_name], tuple):
            activation_tensor = activations[layer_name][0]
        else:
            activation_tensor = activations[layer_name]
        num_activations = activation_tensor.shape[0]
        grid_size = math.ceil(math.sqrt(num_activations))

        # Create a figure with GridSpec to manage space between image and color bar
        fig = plt.figure(
            figsize=(grid_size * 2.5, grid_size * 2)
        )  # Adjust figure size to better accommodate color bars
        gs = gridspec.GridSpec(
            grid_size,
            grid_size,
            width_ratios=[1] * grid_size,
            height_ratios=[1] * grid_size,
        )

        activation_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                ax = fig.add_subplot(gs[i, j])
                if activation_idx < num_activations:
                    if activation_tensor.ndim == 3:  # Typical for conv layers
                        data = activation_tensor[activation_idx, :, :]
                    # elif activation_tensor.ndim == 2:  # Typical for flattened or dense layers
                    #     data = np.tile(activation_tensor[activation_idx, :], (10, 1))  # Expand vertically
                    # elif activation_tensor.ndim == 1:  # Directly dense layer, rare case
                    #     data = np.tile(activation_tensor[:, np.newaxis], (1, 10))  # Expand horizontally
                    else:
                        raise ValueError(
                            f"Unsupported tensor dimension {activation_tensor.ndim}: must be 3"
                        )

                    im = ax.imshow(
                        data, aspect="auto", vmin=-plot_scale_max, vmax=plot_scale_max
                    )
                    ax.set_title(
                        f"Filter {activation_idx + 1} {layer_name}", fontsize=8
                    )

                    # Create a new axis for the color bar next to the current axis
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)

                    activation_idx += 1
                else:
                    ax.axis("off")
                ax.axis("off")  # Maintain a clean look by hiding axis ticks and labels

        plt.tight_layout()

        if save_filename_prefix:
            save_filename = f"{save_filename_prefix}_{layer_name}.png"
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()


def plot_four_activations_for_layers(
    activations, layer_paths, save_filename_prefix=None
):
    for layer_name in layer_paths:
        # Check if the specified layer's activations are available
        if layer_name not in activations:
            print(f"No activations found for layer: {layer_name}")
            continue

        # Extract the activation tensor for the specified layer from the tuple
        activation_tensor = activations[layer_name][0]

        # The tensor is 3-dimensional [channels, height, width]
        num_activations = activation_tensor.shape[0]  # Number of activation maps

        # Select 4 random activation indices if there are at least 4 activations available
        if num_activations >= 4:
            selected_indices = random.sample(range(num_activations), 4)
        else:
            selected_indices = range(num_activations)

        # We will plot only 4 activation maps
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # Always 2x2 grid for 4 plots
        axes = axes.flatten()  # Flatten to simplify the indexing

        # Plot each of the randomly selected activation maps
        for idx, activation_idx in enumerate(selected_indices):
            ax = axes[idx]
            ax.imshow(
                activation_tensor[activation_idx, :, :], cmap="viridis", aspect="auto"
            )
            ax.set_title(f"Filter {activation_idx + 1} {layer_name}", fontsize=8)
            ax.axis("off")  # Hide axes for a cleaner look

        plt.tight_layout()

        # Save or show the plot
        if save_filename_prefix:
            save_filename = f"{save_filename_prefix}_{layer_name}.png"
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()

def plot_layer_channels(activations, layer_name, channel_indices=None, channels_per_plot=None,
                       cmap='viridis', save_prefix=None, figsize=(12, 10), layout=None, return_image=False):
    """
    Plot channels of a layer's activations with flexible layout options.
    
    Args:
        activations: Dictionary of activations
        layer_name: Name of the layer to visualize (e.g., 'conv4a')
        channel_indices: List of specific channel indices to plot. If None, plot all channels.
        channels_per_plot: Maximum number of channels to show in a single figure.
                          If None, uses all channels.
        cmap: Colormap to use for the visualization
        save_prefix: If provided, save plots with this prefix instead of displaying
        figsize: Size of the figure (width, height) in inches
        layout: Tuple (rows, cols) for the grid layout. If None, calculated automatically.
        return_image: If True, returns the figure instead of displaying/saving it
        
    Returns:
        If return_image is True, returns the matplotlib figure object
        Otherwise returns None
    """
    if layer_name not in activations:
        print(f"Layer '{layer_name}' not found in activations!")
        return None
    
    # Get activation tensor for the layer
    activation = activations[layer_name]
    print(f"Layer: {layer_name}, Shape: {activation.shape}")
    
    # Handle different possible shapes
    if len(activation.shape) == 3:  # Shape is [channels, height, width]
        num_channels, height, width = activation.shape
        
        # Determine which channels to plot
        if channel_indices is None:
            channel_indices = list(range(num_channels))
        else:
            # Make sure indices are valid
            channel_indices = [i for i in channel_indices if 0 <= i < num_channels]
            if not channel_indices:
                print(f"No valid channel indices provided for layer {layer_name}")
                return None
        
        # Determine layout constraints
        if layout is not None:
            rows, cols = layout
            # If layout is specified, it determines channels_per_plot
            max_channels_per_plot = rows * cols
        else:
            # Default layout calculation
            if channels_per_plot is None:
                # If no layout or channels_per_plot is specified, show all channels
                channels_per_plot = len(channel_indices)
            
            # Calculate a reasonable layout for channels_per_plot
            cols = math.ceil(math.sqrt(min(channels_per_plot, len(channel_indices))))
            rows = math.ceil(min(channels_per_plot, len(channel_indices)) / cols)
            max_channels_per_plot = rows * cols
        
        # Calculate number of plots needed
        num_plots = math.ceil(len(channel_indices) / max_channels_per_plot)
        
        # Generate each plot
        for plot_idx in range(num_plots):
            start_idx = plot_idx * max_channels_per_plot
            end_idx = min(start_idx + max_channels_per_plot, len(channel_indices))
            curr_indices = channel_indices[start_idx:end_idx]
            
            # Adjust layout for the last plot if it's not full
            if plot_idx == num_plots - 1 and len(curr_indices) < max_channels_per_plot and layout is None:
                # Recalculate layout for the last plot if it's not full
                cols = math.ceil(math.sqrt(len(curr_indices)))
                rows = math.ceil(len(curr_indices) / cols)
            
            # Create figure and subplots
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            
            # Handle case of single subplot
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Plot each channel
            for i, channel_idx in enumerate(curr_indices):
                row, col = i // cols, i % cols
                ax = axes[row, col]
                
                # Get channel activation and plot
                channel_data = activation[channel_idx].detach().cpu().numpy()
                im = ax.imshow(channel_data, cmap=cmap)
                ax.set_title(f"Channel {channel_idx}", fontsize=10)
                ax.axis('off')
            
            # Hide any empty subplots
            for i in range(len(curr_indices), rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')
            
            # Add a colorbar to the figure
            plt.tight_layout()
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            
            plt.suptitle(f"Layer: {layer_name} (Plot {plot_idx+1}/{num_plots})", 
                         fontsize=16, y=0.98)
            
            if return_image:
                return fig
            elif save_prefix:
                plt.savefig(f"{save_prefix}_{layer_name}_plot{plot_idx+1}.png", bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    else:
        print(f"Unexpected shape for layer {layer_name}: {activation.shape}")
        print("This function expects 3D activations with shape [channels, height, width]")

def compute_activation_differences(activations1, activations2):
    differences = {}
    for key in activations1:
        # Compute the difference tensor for the current key
        difference = activations2[key][0] - activations1[key][0]

        # Store the difference tensor in a tuple
        differences[key] = (difference,)

        # Check if there are any non-zero differences
        has_non_zero = t.any(difference != 0)

        if has_non_zero:
            print(f"Key: {key} has non-zero differences.")
        else:
            print(f"Key: {key} has only zero differences.")
    return differences


def plot_activations_for_layers_side_by_side(
    activations1, activations2, layer_paths, save_filename_prefix=None
):
    for layer_name in layer_paths:
        # Check if the specified layer's activations are available in both sets
        if layer_name not in activations1 or layer_name not in activations2:
            print(f"No activations found for layer: {layer_name}")
            continue

        # Extract the activation tensors for the specified layer from both sets
        activation_tensor1 = activations1[layer_name][0]
        activation_tensor2 = activations2[layer_name][0]

        # The tensors are 3-dimensional [channels, height, width]
        num_activations = activation_tensor1.shape[0]  # Number of activation maps

        # Calculate grid size
        grid_size = math.ceil(math.sqrt(num_activations))

        # Create a figure with dynamic subplots based on the number of activations
        fig, axes = plt.subplots(
            grid_size, grid_size * 2, figsize=(grid_size * 4, grid_size * 2)
        )
        if grid_size == 1:
            axes = np.array(
                [[axes[0], axes[1]]]
            )  # Ensure axes can be indexed with two dimensions

        # Initialize an index for activation maps
        activation_idx = 0
        for i in range(grid_size):
            for j in range(grid_size * 2):
                ax = axes[i, j]
                # Plot the activation maps side by side if we haven't gone through all of them yet
                if activation_idx < num_activations:
                    if j % 2 == 0:
                        ax.imshow(
                            activation_tensor1[activation_idx, :, :],
                            cmap="viridis",
                            aspect="auto",
                        )
                        ax.set_title(f"Filter {activation_idx+1} (Set 1)", fontsize=8)
                    else:
                        ax.imshow(
                            activation_tensor2[activation_idx, :, :],
                            cmap="viridis",
                            aspect="auto",
                        )
                        ax.set_title(f"Filter {activation_idx+1} (Set 2)", fontsize=8)
                        activation_idx += 1
                else:
                    ax.axis("off")  # Hide axes without data
                ax.axis("off")  # Hide axes for all plots for a cleaner look

        plt.tight_layout()

        # Save or show the plot
        if save_filename_prefix:
            save_filename = f"{save_filename_prefix}_{layer_name}.png"
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()


def plot_layer_activations_dynamic_grid(
    activations, layer_name, save_filename=None, observation=None
):
    if layer_name not in activations:
        print(f"No activations found for layer: {layer_name}")
        return

    activation_tensor = activations[layer_name][0].cpu().numpy()

    if activation_tensor.ndim == 3:
        activation_tensor = activation_tensor[np.newaxis, :]

    num_activations = activation_tensor.shape[1]

    grid_size = math.ceil(math.sqrt(num_activations + 1))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
    )

    if observation is not None:
        obs_for_plot = observation.squeeze().numpy()
        axes[0, 0].imshow(obs_for_plot)
        axes[0, 0].set_title("Observation", fontsize=8)
        axes[0, 0].axis("off")
        start_idx = 1
    else:
        start_idx = 0

    activation_idx = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if i == 0 and j == 0 and observation is not None:
                continue
            ax = axes[i, j]

            if activation_idx < num_activations:
                ax.imshow(
                    activation_tensor[0, activation_idx, :, :],
                    cmap="viridis",
                    aspect="auto",
                )
                ax.set_title(f"Filter {activation_idx + start_idx}", fontsize=8)
                activation_idx += 1
            else:
                ax.axis("off")

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)
        plt.close()
    else:
        plt.show()


def plot_single_observation(observation):
    # Convert (1, 3, 64, 64) or (3, 64, 64) to (64, 64, 3) if necessary
    if observation.shape == (1, 3, 64, 64):
        observation = observation.squeeze().transpose(1, 2, 0)
    elif observation.shape == (3, 64, 64):
        observation = observation.transpose(1, 2, 0)

    plt.imshow(observation)
    # plt.title("Observation")
    plt.axis("off")
    plt.show()


def plot_multiple_observations(observation_list):
    num_observations = len(observation_list)
    grid_size = math.ceil(math.sqrt(num_observations))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
    )

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_observations:
                ax = axes[i, j]
                ax.imshow(observation_list[idx])
                ax.set_title(f"Observation {idx}")
                ax.axis("off")
            else:
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def rgb_to_observation(rgb_image):
    # Ensure the rgb_image is a 64x64x3 tensor
    # assert rgb_image.shape == (64, 64, 3), "RGB image must be a 64x64x3 tensor" #Do we need this?

    # Scale the rgb_image values to the range of 0 to 1
    if isinstance(rgb_image, np.ndarray):
        observation = rgb_image / 255.0
        observation = observation.astype(np.float32)
    elif t.is_tensor(rgb_image):
        observation = rgb_image.float() / 255.0
    else:
        raise TypeError("RGB image must be a numpy array or a Pyt tensor.")
    return observation


def observation_to_rgb(observation):
    # Ensure the observation is a 64x64x3 tensor
    # assert observation.shape == (64, 64, 3), "Observation must be a 64x64x3 tensor" #Do we need this?

    # Scale the observation values to the range of 0 to 255
    if isinstance(observation, np.ndarray):
        rgb_image = observation * 255
        rgb_image = rgb_image.astype(np.uint8)
    elif t.is_tensor(observation):
        rgb_image = observation * 255
        rgb_image = rgb_image.byte()
    else:
        raise TypeError("Observation must be a numpy array or a Pyt tensor.")
    return rgb_image


def tensor_to_image(tensor):
    return tensor.squeeze().transpose(1, 2, 0)


def rename_paths(paths):
    return [s.replace(".", "_") for s in paths]


def rename_path(path):
    return path.replace(".", "_")


def plot_confusion_matrix(conf_matrix, labels_dict):
    # Define the size of the figure
    plt.figure(figsize=(3, 3))

    # Convert labels_dict keys to a list for x and y axis labels
    labels = list(labels_dict.keys())

    # Plotting using seaborn
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )

    # Adding labels and title for clarity
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix Visualization")
    plt.show()


def run_episode_and_save_as_gif(
    env,
    model,
    filepath="../gifs/run.gif",
    save_gif=False,
    episode_timeout=200,
    is_procgen_env=True,
):
    model=model.to(device)
    observations = []
    observation = env.reset()
    done = False
    if isinstance(observation, np.ndarray) and observation.ndim > 3:
        done = t.zeros(observation.shape[0], dtype=t.bool)
    else:
        done = False
    total_reward = 0
    frames = []

    # observation = colour_swap(observation)
    count = 0
    while not done.all():  # Check if all environments in the batch are done
        if save_gif:
            frames.append(env.render(mode="rgb_array"))

        if observation.ndim == 4:  # If there's a batch dimension
            observation = np.transpose(
                observation, (0, 2, 3, 1)
            )  # (batch, height, width, channels)
        else:
            observation = np.transpose(
                observation, (1, 2, 0)
            )  # (height, width, channels)
        converted_obs = observation_to_rgb(observation)

        action = generate_action(model, converted_obs, is_procgen_env=is_procgen_env)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        observations.append(converted_obs)
        count += 1
        if count >= episode_timeout:
            break

    if save_gif:
        imageio.mimsave(filepath, frames, fps=30)
        print(f"saved gif at {filepath}!")

    return total_reward, frames, observations


def run_episode_with_steering_and_save_as_gif(
    env,
    model,
    steering_vector,
    steering_layer,
    modification_value,
    filepath="../gifs/run.gif",
    save_gif=False,
    episode_timeout=400,
    is_procgen_env=True,
):
    observations = []
    observation = env.reset()
    # plot_single_observation(observation.squeeze().transpose(1,2,0))

    done = False
    total_reward = 0
    frames = []
    activations = {}
    # observation = colour_swap(observation)
    count = 0
    while not done:
        if save_gif:
            frames.append(env.render(mode="rgb_array"))
        observation = np.squeeze(observation)
        observation = np.transpose(observation, (1, 2, 0))
        converted_obs = observation_to_rgb(observation)
        action = generate_action_with_steering(
            model,
            converted_obs,
            steering_vector,
            steering_layer,
            modification_value,
            is_procgen_env,
        )

        observation, reward, done, info = env.step(action)
        total_reward += reward
        observations.append(converted_obs)
        count += 1
        if count >= episode_timeout:
            break
    if save_gif:
        imageio.mimsave(filepath, frames, fps=30)
        print("Saved gif!")

    return total_reward, frames, observations


def run_episode_with_steering_and_check_target_acquisition(
    env,
    model,
    steering_vector,
    steering_layer,
    modification_value,
    filepath="../gifs/run.gif",
    save_gif=False,
    episode_timeout=400,
    is_procgen_env=True,
):
    observations = []
    observation = env.reset()

    done = False
    total_reward = 0
    frames = []
    activations = {}
    state = heist.state_from_venv(env, 0)

    lock_positions_before = heist.get_lock_positions(state)
    num_changes_expected = len(lock_positions_before)
    num_changes_counted = 0
    count = 0
    while not done:
        if save_gif:
            frames.append(env.render(mode="rgb_array"))
        observation = np.squeeze(observation)
        observation = np.transpose(observation, (1, 2, 0))
        converted_obs = observation_to_rgb(observation)
        action = generate_action_with_steering(
            model,
            converted_obs,
            steering_vector,
            steering_layer,
            modification_value,
            is_procgen_env,
        )

        observation, reward, done, info = env.step(action)

        lock_positions_after = get_lock_positions(heist.state_from_venv(env, 0))
        if lock_positions_before != lock_positions_after:
            num_changes_counted += 1
        total_reward += reward
        observations.append(converted_obs)
        count += 1
        if count >= episode_timeout:
            break
    if save_gif:
        imageio.mimsave(filepath, frames, fps=30)
        print("Saved gif!")

    if num_changes_counted == num_changes_expected and total_reward == 0:
        return True
    else:
        return False


def create_objective_activation_dataset(dataset, model, layer_paths):
    activation_dataset = {
        "gem": [],
        "blue_key": [],
        "green_key": [],
        "red_key": [],
        "blue_lock": [],
        "green_lock": [],
        "red_lock": [],
    }

    for category in dataset:
        for obs in dataset[category]:
            obs_rgb = observation_to_rgb(obs)
            model_activations = ModelActivations(model)
            _, activations = model_activations.run_with_cache(obs_rgb, layer_paths)
            model_activations.clear_hooks()

            activation_dataset[category].append(activations)

    return activation_dataset



@t.no_grad()
def generate_action_with_patching(
    model, observation, patched_vector, steering_layer, is_procgen_env=False
):
    # Check for available devices
    device = None
    xm = None

    if device is None:
        if t.cuda.is_available():
            device = t.device("cuda")
        elif t.backends.mps.is_available():
            device = t.device("mps")
        else:
            device = t.device("cpu")

    observation = t.tensor(observation, dtype=t.float32).unsqueeze(0)

    # Define the steering hook function
    def steering_hook(module, input, output):
        # Add the steering vector to the output activations
        modified_output = output + (patched_vector.unsqueeze(0))
        return modified_output

    # Register the steering hook to the specified layer
    named_modules_dict = dict(model.named_modules())
    target_layer = named_modules_dict[steering_layer]
    steering_handle = target_layer.register_forward_hook(steering_hook)

    # Forward pass with steering
    model_output = model(observation)

    # Remove the steering hook
    steering_handle.remove()

    logits = model_output[
        0
    ].logits  # discard the output of the critic in our actor critic network
    probabilities = t.softmax(logits, dim=-1)
    action = t.multinomial(probabilities, 1).item()

    # If using TPU, we need to explicitly synchronize the device
    if xm is not None:
        xm.mark_step()

    if is_procgen_env:
        return np.array([action])
    return action


@t.no_grad()
def generate_action_with_steering(
    model,
    observation,
    steering_vector,
    steering_layer,
    modification_value,
    is_procgen_env=False,
):
    # Check for available devices
    device = None
    xm = None

    # try:
    #     import t_xla.core.xla_model as xm
    #     if xm.xrt_world_size() > 0:
    #         device = xm.xla_device()
    #         # print("Running on TPU")
    # except ImportError:
    #     print("Pyt XLA not available. Will use CPU/GPU if available.")

    if device is None:
        if t.cuda.is_available():
            device = t.device("cuda")
        elif t.backends.mps.is_available():
            device = t.device("mps")
        else:
            device = t.device("cpu")

    # Move model to the appropriate device
    # model = model.to(device)

    observation = t.tensor(observation, dtype=t.float32).unsqueeze(0)
    steering_vector = steering_vector

    # Define the steering hook function
    def steering_hook(module, input, output):
        # Add the steering vector to the output activations
        modified_output = output + (steering_vector.unsqueeze(0) * modification_value)
        return modified_output

    # Register the steering hook to the specified layer
    named_modules_dict = dict(model.named_modules())
    target_layer = named_modules_dict[steering_layer]
    steering_handle = target_layer.register_forward_hook(steering_hook)

    # Forward pass with steering
    model_output = model(observation)

    # Remove the steering hook
    steering_handle.remove()

    logits = model_output[
        0
    ].logits  # discard the output of the critic in our actor critic network
    probabilities = t.softmax(logits, dim=-1)
    action = t.multinomial(probabilities, 1).item()

    # If using TPU, we need to explicitly synchronize the device
    if xm is not None:
        xm.mark_step()

    if is_procgen_env:
        return np.array([action])
    return action


def create_activation_dataset(dataset, model, layer_paths, categories):
    activation_dataset = categories

    for category in dataset:
        for obs in dataset[category]:
            obs_rgb = observation_to_rgb(obs)
            model_activations = ModelActivations(model)
            _, activations = model_activations.run_with_cache(obs_rgb, layer_paths)
            model_activations.clear_hooks()

            activation_dataset[category].append(activations)

    return activation_dataset


def run_gem_steering_experiment(
    model_path,
    layer_number,
    modification_value,
    num_levels=1,
    start_level=5,
    episode_timeout=200,
    save_gif=False,
    gif_filepath="steering_gif.gif",
):
    start_level = random.randint(1, 10000)
    venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
    state = heist.state_from_venv(venv, 0)
    unchanged_obs = venv.reset()

    unchanged_obs = venv.reset()
    state_values = state.state_vals

    for ents in state_values["ents"]:
        if ents["image_type"].val == 9:
            gem_x = ents["x"].val
            gem_y = ents["y"].val

    state.remove_gem()

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

    state = heist.state_from_venv(venv, 0)

    state.set_gem_position(gem_y - 0.5, gem_x - 0.5)

    state_bytes = state.state_bytes

    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
    # Load model and calculate steering vector
    model = load_model(model_path=model_path)
    layer_names = get_model_layer_names(model)
    steering_layer_unchanged = ordered_layer_names[layer_number]
    steering_layer = rename_path(steering_layer_unchanged)

    model_activations = ModelActivations(model)
    model_activations.clear_hooks()
    output1, unmodified_activations = model_activations.run_with_cache(
        observation_to_rgb(unchanged_obs), layer_names
    )
    model_activations.clear_hooks()
    output2, modified_obs_activations = model_activations.run_with_cache(
        observation_to_rgb(modified_obs), layer_names
    )

    steering_vector = (
        unmodified_activations[steering_layer][0]
        - modified_obs_activations[steering_layer][0]
    )

    # Run episode with steering
    total_reward_steering, frames_steering, observations_steering = (
        run_episode_with_steering_and_save_as_gif(
            venv,
            model,
            steering_vector,
            steering_layer=ordered_layer_names[layer_number],
            modification_value=modification_value,
            filepath=gif_filepath,
            save_gif=save_gif,
            episode_timeout=episode_timeout,
        )
    )

    return total_reward_steering


### Functions for generating/plotting activations for custom mazes


def make_mazes_with_entities_removed(venv, list_of_entities_to_remove: list):
    # Takes in maze/venv with a list of lists of entities to remove.
    # For each list of entities to remove, creates obs+frame with those entities removed from maze

    # Remove entities not in entity list
    def remove_entities(state, entities):
        if entities == ["all"]:
            state.remove_all_entities()
            return state
        else:
            if "gem" in entities:
                state.remove_gem()
            if "player" in entities:
                state.remove_player()
            if "blue_key" in entities:
                state.delete_specific_keys([0])
            if "green_key" in entities:
                state.delete_specific_keys([1])
            if "red_key" in entities:
                state.delete_specific_keys([2])
            if "blue_lock" in entities:
                state.delete_specific_locks([0])
            if "green_lock" in entities:
                state.delete_specific_locks([1])
            if "red_lock" in entities:
                state.delete_specific_locks([2])
            return state

    obs_list = []
    frames_list = []

    # Save original state
    original_state = heist.state_from_venv(venv, 0)

    # Save original obs and frame
    original_obs = venv.reset()
    original_frame = venv.render(mode="rgb_array")

    obs_list.append(original_obs)
    frames_list.append(original_frame)

    for entities_to_remove in list_of_entities_to_remove:
        # Reset state
        venv.env.callmethod("set_state", [original_state.state_bytes])
        state = heist.state_from_venv(venv, 0)

        # Remove entities
        state = remove_entities(state, entities_to_remove)

        # Update environment using new state
        state_bytes = state.state_bytes
        if state_bytes is not None:
            venv.env.callmethod("set_state", [state_bytes])
            obs = venv.reset()
            frame = venv.render(mode="rgb_array")
        else:
            raise ValueError("State bytes is None")

        obs_list.append(obs)
        frames_list.append(frame)

    return obs_list, frames_list


def calc_activations_for_obs_list(
    model_activations, obs_list: list, layer_names
) -> list:
    # Runs model and collects activations for a list of observations
    activations_list = []
    for i, obs in enumerate(obs_list):
        output, activations = model_activations.run_with_cache(
            observation_to_rgb(obs), layer_names
        )
        activations_list.append(activations)
    return activations_list


def calc_weighted_activations(
    activations_list: list, activation_weightings: list = [1, -1]
) -> dict:
    # Calculates a weighted sum of activations for a list of activations
    weighted_activations = {}

    for layer in activations_list[0].keys():
        for i, activations in enumerate(activations_list):
            if layer not in weighted_activations.keys():
                if isinstance(activations[layer], tuple):
                    weighted_activations[layer] = (
                        activation_weightings[i] * activations[layer][0]
                    )
                else:
                    weighted_activations[layer] = (
                        activation_weightings[i] * activations[layer]
                    )
            else:
                if isinstance(activations[layer], tuple):
                    weighted_activations[layer] += (
                        activation_weightings[i] * activations[layer][0]
                    )
                else:
                    weighted_activations[layer] += (
                        activation_weightings[i] * activations[layer]
                    )
        # weighted_activations[layer] = (weighted_activations[layer], )

    return weighted_activations


def get_objective_activations(model_activations, layer_paths, num_samples=16) -> dict:
    """Run observations for different objectives through model and collect activations for each layer."""

    # Create and update datasets
    dataset = heist.create_classified_dataset(
        num_samples_per_category=num_samples, num_levels=0
    )
    empty_dataset = heist.create_empty_maze_dataset(
        num_samples_per_category=num_samples, num_levels=0
    )
    dataset.update(empty_dataset)

    # Initialize dictionaries for activations and class vectors
    objective_activations = defaultdict(dict)

    # Process each objective in the dataset
    for objective, data in dataset.items():
        # Stack and convert the dataset to a tensor
        dataset_tensor = np.stack(data)

        # Run the model to get output and activations
        _, activations = model_activations.run_with_cache(
            observation_to_rgb(dataset_tensor), layer_paths
        )
        objective_activations[objective] = activations

    return objective_activations


def create_objective_vectors(model_activations, layer_paths, num_samples=16) -> dict:
    # Create objective vectors that are the mean of activations for obs that correspond to model going for a specific objective.

    # Create and update datasets
    dataset = heist.create_classified_dataset(
        num_samples_per_category=num_samples, num_levels=0
    )
    empty_dataset = heist.create_empty_maze_dataset(
        num_samples_per_category=num_samples, num_levels=0
    )
    dataset.update(empty_dataset)

    # Initialize dictionaries for activations and class vectors
    objective_vectors = defaultdict(dict)

    # Process each objective in the dataset
    for objective, data in dataset.items():
        # Stack and convert the dataset to a tensor
        dataset_tensor = np.stack(data)

        # Run the model to get output and activations
        _, activations = model_activations.run_with_cache(
            observation_to_rgb(dataset_tensor), layer_paths
        )
        objective_vectors[objective] = activations

        # Calculate the mean of activations for each layer
        for layer, activation in activations.items():
            objective_vectors[objective][layer] = t.stack(activation).mean(dim=0)

    return objective_vectors


@t.no_grad()
def batch_generate_action_with_cache(
    model, observations, model_activations, layer_paths, is_procgen_env=False
):
    """
    Generates actions and captures activations for a batch of observations.

    Args:
        model (nn.Module): The policy model.
        observations (list or np.ndarray): List or array of observations.
        model_activations (ModelActivations): Instance of ModelActivations.
        layer_paths (list of str): Layers from which to capture activations.
        is_procgen_env (bool): Flag indicating if using Procgen environments.

    Returns:
        actions (np.ndarray): Array of actions.
        activations (dict): Dictionary of captured activations.
    """
    # Convert observations to tensor and preprocess
    if isinstance(observations, list):
        observations = np.stack(observations)
    observations = observation_to_rgb(observations)  # Ensure correct preprocessing
    observations = t.tensor(observations, dtype=t.float32)

    # Run the model and capture activations
    outputs, activations = model_activations.run_with_cache(observations, layer_paths)

    # Extract logits (assuming model outputs a tuple where the first element has 'logits')
    if isinstance(outputs, tuple):
        logits = outputs[0].logits
    else:
        logits = outputs.logits

    # Compute action probabilities
    probabilities = t.softmax(logits, dim=-1)

    # Sample actions
    actions = t.multinomial(probabilities, num_samples=1).squeeze(1).cpu().numpy()

    if is_procgen_env:
        actions = actions.reshape(-1, 1)

    return actions, activations
