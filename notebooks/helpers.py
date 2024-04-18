import torch
import torch.nn as nn
import imageio

import sys
sys.path.append('../') #This is added so we can import from the source folder
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, DummyVecEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import imageio


from src.policies_impala import ImpalaCNN
# from src.policies_modified import ImpalaCNN
from src.visualisation_functions import *

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
                if isinstance(item, torch.Tensor):
                    processed_output.append(item.detach())
                elif isinstance(item, torch.distributions.Categorical):
                    processed_output.append(item.logits.detach())
                else:
                    processed_output.append(item)
            self.activations[name] = tuple(processed_output)
        return hook

    def register_hook_by_path(self, path, name):
        elements = path.split('.')
        model = self.model
        for i, element in enumerate(elements):
            if '[' in element:
                base, index = element.replace(']', '').split('[')
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
        for path in layer_paths:
            self.register_hook_by_path(path, path.replace('.', '_'))
        output = self.model(input)
        return output, self.activations



@torch.no_grad()
def generate_action(model, observation, is_procgen_env=False):
    observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    model_output = model(observation)
    
    logits = model_output[0].logits  # discard the output of the critic in our actor critic network
    
    probabilities = torch.softmax(logits, dim=-1)
    
    action = torch.multinomial(probabilities, 1).item() 
    if is_procgen_env:
        return np.array([action])
    return action

def load_model(ImpalaCNN = ImpalaCNN, model_path ="../model_1400_latest.pt"):
    env_name = "procgen:procgen-heist-v0"  
    env = gym.make(env_name, start_level=100, num_levels=200, render_mode="rgb_array", distribution_mode="easy") 
    observation_space = env.observation_space
    action_space = env.action_space.n
    model = ImpalaCNN(observation_space, action_space)
    model.load_from_file(model_path, device="cpu")
    return model

def get_model_layer_names(model):
    layer_names = [name for name, _ in model.named_modules() if isinstance(_, nn.Module)]
    return layer_names[1:len(layer_names)]

def plot_activations_for_layers(activations, layer_paths, save_filename_prefix=None):
    for layer_name in layer_paths:
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
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
        if grid_size == 1:
            axes = np.array([[axes]])  # Ensure axes can be indexed with two dimensions

        # Initialize an index for activation maps
        activation_idx = 0

        for i in range(grid_size):
            for j in range(grid_size):
                ax = axes[i, j]
                
                # Plot the activation map if we haven't gone through all of them yet
                if activation_idx < num_activations:
                    ax.imshow(activation_tensor[activation_idx, :, :], cmap='viridis', aspect='auto')
                    ax.set_title(f'Filter {activation_idx+1} {layer_name}', fontsize=8)
                    activation_idx += 1
                else:
                    ax.axis('off')  # Hide axes without data
                
                ax.axis('off')  # Hide axes for all plots for a cleaner look

        plt.tight_layout()
        
        # Save or show the plot
        if save_filename_prefix:
            save_filename = f"{save_filename_prefix}_{layer_name}.png"
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()


def compute_activation_differences(activations1, activations2):
    differences = {}
    for key in activations1:
        # Compute the difference tensor for the current key
        difference = activations2[key][0] - activations1[key][0]
        
        # Store the difference tensor in a tuple
        differences[key] = (difference,)
        print(difference.shape)

        # Check if there are any non-zero differences
        has_non_zero = torch.any(difference != 0)
        
        if has_non_zero:
            print(f"Key: {key} has non-zero differences.")
        else:
            print(f"Key: {key} has only zero differences.")
    print(differences)
    return differences

def plot_activations_for_layers_side_by_side(activations1, activations2, layer_paths, save_filename_prefix=None):
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
        fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=(grid_size * 4, grid_size * 2))
        if grid_size == 1:
            axes = np.array([[axes[0], axes[1]]])  # Ensure axes can be indexed with two dimensions

        # Initialize an index for activation maps
        activation_idx = 0
        for i in range(grid_size):
            for j in range(grid_size * 2):
                ax = axes[i, j]
                # Plot the activation maps side by side if we haven't gone through all of them yet
                if activation_idx < num_activations:
                    if j % 2 == 0:
                        ax.imshow(activation_tensor1[activation_idx, :, :], cmap='viridis', aspect='auto')
                        ax.set_title(f'Filter {activation_idx+1} (Set 1)', fontsize=8)
                    else:
                        ax.imshow(activation_tensor2[activation_idx, :, :], cmap='viridis', aspect='auto')
                        ax.set_title(f'Filter {activation_idx+1} (Set 2)', fontsize=8)
                        activation_idx += 1
                else:
                    ax.axis('off')  # Hide axes without data
                ax.axis('off')  # Hide axes for all plots for a cleaner look

        plt.tight_layout()

        # Save or show the plot
        if save_filename_prefix:
            save_filename = f"{save_filename_prefix}_{layer_name}.png"
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()

def plot_layer_activations_dynamic_grid(activations, layer_name, save_filename=None, observation=None):
    if layer_name not in activations:
        print(f"No activations found for layer: {layer_name}")
        return

    activation_tensor = activations[layer_name][0].cpu().numpy()

    if activation_tensor.ndim == 3:
        activation_tensor = activation_tensor[np.newaxis, :]

    num_activations = activation_tensor.shape[1]

    grid_size = math.ceil(math.sqrt(num_activations + 1))  
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))


    if observation is not None:
        obs_for_plot = observation.squeeze().numpy()
        axes[0, 0].imshow(obs_for_plot)
        axes[0, 0].set_title("Observation", fontsize=8)
        axes[0, 0].axis('off')
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
                ax.imshow(activation_tensor[0, activation_idx, :, :], cmap='viridis', aspect='auto')
                ax.set_title(f'Filter {activation_idx + start_idx}', fontsize=8) 
                activation_idx += 1
            else:
                ax.axis('off') 

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)
        plt.close()
    else:
        plt.show()

def plot_single_observation(observation):
    plt.imshow(observation)
    plt.title("Observation")
    plt.axis('off')  
    plt.show()


def observation_to_rgb(observation):
    # Ensure the observation is a 64x64x3 tensor
    if observation.shape == (1, 3 , 64, 64):
        observation = observation.squeeze().transpose(1,2,0)
    assert observation.shape == (64, 64, 3), "Observation must be a 64x64x3 tensor"

    # Scale the observation values to the range of 0 to 255
    rgb_image = observation * 255

    # Convert the tensor to uint8 data type
    rgb_image = rgb_image.astype(np.uint8)

    return rgb_image

def rename_paths(paths):
    return [s.replace('.', '_') for s in paths]
