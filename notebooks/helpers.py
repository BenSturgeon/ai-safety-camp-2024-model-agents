import torch
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
from src.visualisation_functions import *


@torch.no_grad()
def generate_action(model, observation):
    observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    model_output = model(observation)
    
    logits = model_output[0].logits  # discard the output of the critic in our actor critic network
    
    probabilities = torch.softmax(logits, dim=-1)
    
    action = torch.multinomial(probabilities, 1).item() 
    return action

def load_model(model_path = '../model_1400_latest.pt'):
    env_name = "procgen:procgen-heist-v0"  
    env = gym.make(env_name, start_level=100, num_levels=200, render_mode="rgb_array", distribution_mode="easy") #remove render mode argument to go faster but not produce images 
    observation_space = env.observation_space
    action_space = env.action_space.n
    model = ImpalaCNN(observation_space, action_space)
    model.load_from_file(model_path, device="cpu")
    return model


def plot_activations_for_layers(activations, layer_paths, save_filename_prefix=None):
    for layer_name in layer_paths:
        # Check if the specified layer's activations are available
        if layer_name not in activations:
            print(f"No activations found for layer: {layer_name}")
            continue

        # Extract the activation tensor for the specified layer
        activation_tensor = activations[layer_name].cpu().numpy()

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
        
        # Store the difference tensor in a new dictionary
        differences[key] = difference
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