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
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import math
import imageio
import random

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

# @torch.no_grad()
# def generate_action_with_steering(model, observation, steering_vector, is_procgen_env=True):
#     observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

#     # Define the steering hook function
#     def steering_hook(module, input, output):
#         # Add the steering vector to the output activations
#         modified_output = output + steering_vector.unsqueeze(0)
#         return modified_output

#     # Register the steering hook to the 'hidden_fc' layer
#     named_modules_dict = dict(model.named_modules())
#     hidden_fc_layer = named_modules_dict['hidden_fc']
#     steering_handle = hidden_fc_layer.register_forward_hook(steering_hook)

#     # Forward pass with steering
#     model_output = model(observation)

#     # Remove the steering hook
#     steering_handle.remove()



#     logits = model_output[0].logits  # discard the output of the critic in our actor critic network
#     probabilities = torch.softmax(logits, dim=-1)
#     action = torch.multinomial(probabilities, 1).item()
#     if is_procgen_env:
#         return np.array([action])
#     return action

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


def plot_activations_for_layers(activations, layer_paths, save_filename_prefix=None, plot_scale_max=5):
    plt.rcParams['image.cmap'] = 'RdBu_r'  # Set the reversed default colormap to 'RdBu_r' for all plots

    for layer_name in layer_paths:
        if layer_name not in activations:
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
        fig = plt.figure(figsize=(grid_size * 2.5, grid_size * 2))  # Adjust figure size to better accommodate color bars
        gs = gridspec.GridSpec(grid_size, grid_size, width_ratios=[1]*grid_size, height_ratios=[1]*grid_size)

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
                        raise ValueError(f"Unsupported tensor dimension {activation_tensor.ndim}: must be 3")

                    im = ax.imshow(data, aspect='auto', vmin=-plot_scale_max, vmax=plot_scale_max)
                    ax.set_title(f'Filter {activation_idx + 1} {layer_name}', fontsize=8)

                    # Create a new axis for the color bar next to the current axis
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)

                    activation_idx += 1
                else:
                    ax.axis('off')
                ax.axis('off')  # Maintain a clean look by hiding axis ticks and labels

        plt.tight_layout()

        if save_filename_prefix:
            save_filename = f"{save_filename_prefix}_{layer_name}.png"
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()


def plot_four_activations_for_layers(activations, layer_paths, save_filename_prefix=None):
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
            ax.imshow(activation_tensor[activation_idx, :, :], cmap='viridis', aspect='auto')
            ax.set_title(f'Filter {activation_idx + 1} {layer_name}', fontsize=8)
            ax.axis('off')  # Hide axes for a cleaner look

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
    # assert observation.shape == (64, 64, 3), "Observation must be a 64x64x3 tensor" #Do we need this?
    observation = torch.tensor(observation, dtype=torch.float32)
    # Scale the observation values to the range of 0 to 255
    if observation.dim() == 3:
        observation = observation.permute(1, 2, 0)
        rgb_image = observation * 255

    # Convert the tensor to uint8 data type
        rgb_images = rgb_image.astype(np.uint8)
    elif observation.dim() == 4:
        # observation = observation.permute(0, 2, 3, 1)
        rgb_images = observation * 255


    return rgb_images

def tensor_to_image(tensor):
    return tensor.squeeze().transpose(1,2,0)

def rename_paths(paths):
    return [s.replace('.', '_') for s in paths]

def rename_path(path):
    return path.replace('.', '_') 

def plot_confusion_matrix(conf_matrix, labels_dict):
    # Define the size of the figure
    plt.figure(figsize=(10, 7))

    # Convert labels_dict keys to a list for x and y axis labels
    labels = list(labels_dict.keys())

    # Plotting using seaborn
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)

    # Adding labels and title for clarity
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix Visualization')
    plt.show()

def run_episode_and_save_as_gif(env, model, filepath='../gifs/run.gif', save_gif=False, episode_timeout=200, is_procgen_env=True):

    observations = []
    observation = env.reset()
    done = False
    total_reward = 0
    frames=[]
    
    

    # observation = colour_swap(observation)
    count = 0
    while not done:
        if save_gif:
            frames.append(env.render(mode='rgb_array'))  

        observation= np.squeeze(observation)
        observation =np.transpose(observation, (1,2,0))
        converted_obs = observation_to_rgb(observation)
        
        action = generate_action(model, converted_obs, is_procgen_env=is_procgen_env) 
        observation, reward, done, info = env.step(action)
        total_reward += reward
        observations.append(converted_obs)
        count +=1
        if count >= episode_timeout:
            break

    if save_gif:
        imageio.mimsave(filepath, frames, fps=30) 

    return total_reward, frames, observations

def run_episode_with_steering_and_save_as_gif(env, model, steering_vector, steering_layer, modification_value,filepath='../gifs/run.gif', save_gif=False, episode_timeout=400, is_procgen_env=True):
    observations = []
    observation = env.reset()
    # plot_single_observation(observation.squeeze().transpose(1,2,0))

    done = False
    total_reward = 0
    frames=[]
    activations = {}
    # observation = colour_swap(observation)
    count = 0
    while not done:
        if save_gif:
            frames.append(env.render(mode='rgb_array'))
        observation= np.squeeze(observation)
        observation =np.transpose(observation, (1,2,0))
        converted_obs = observation_to_rgb(observation)
        action = generate_action_with_steering(model, converted_obs, steering_vector, steering_layer,modification_value, is_procgen_env)

        observation, reward, done, info = env.step(action)
        total_reward += reward
        observations.append(converted_obs)
        count +=1
        if count >= episode_timeout:
            break
    if save_gif:
        imageio.mimsave(filepath, frames, fps=30)
        print("Saved gif!")

    return total_reward, frames, observations

def create_objective_activation_dataset(dataset, model, layer_paths):
    activation_dataset = {
        "gem": [],      
        "blue_key": [],
        "green_key": [],
        "red_key": [],
        "blue_lock": [],
        "green_lock": [],
        "red_lock": []
    }

    for category in dataset:
        for obs in dataset[category]:
            obs_rgb = observation_to_rgb(obs)
            model_activations = ModelActivations(model)
            _, activations = model_activations.run_with_cache(obs_rgb, layer_paths)
            model_activations.clear_hooks()


            activation_dataset[category].append(activations)

    return activation_dataset

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

        # Handle edge case: input is not a tensor
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float32)


        # Check the shape of the input and reshape if necessary
        if input.shape == torch.Size([1, 3, 64, 64]):
            input = input.squeeze(0)  # Remove the batch dimension
        if input.shape == torch.Size([3, 64, 64]):
            input = input.permute(1, 2, 0)  # Switch dimensions to (64, 64, 3)


        # Handle edge case: empty layer_paths
        if not layer_paths:
            output = self.model(input)
            return output, self.activations

        # Register hooks for each layer path
        for path in layer_paths:
            try:
                self.register_hook_by_path(path, path.replace('.', '_'))
            except AttributeError:
                print(f"Warning: Layer '{path}' not found in the model. Skipping hook registration.")

        # Add batch dimension if missing
        if input.dim() == 3:
            input = input.unsqueeze(0)

        # Run the model with the registered hooks
        output = self.model(input)

        return output, self.activations
    



@torch.no_grad()
def generate_action_with_steering(model, observation, steering_vector,steering_layer, modification_value, is_procgen_env=False):
    observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    
    # Define the steering hook function
    def steering_hook(module, input, output):
        # Add the steering vector to the output activations
        modified_output = output + (steering_vector.unsqueeze(0) * modification_value)
        return modified_output

    # Register the steering hook to the 'hidden_fc' layer
    named_modules_dict = dict(model.named_modules())
    hidden_fc_layer = named_modules_dict[steering_layer]
    steering_handle = hidden_fc_layer.register_forward_hook(steering_hook)

    # Forward pass with steering
    model_output = model(observation)

    # Remove the steering hook
    steering_handle.remove()

    logits = model_output[0].logits  # discard the output of the critic in our actor critic network
    probabilities = torch.softmax(logits, dim=-1)
    action = torch.multinomial(probabilities, 1).item()

    if is_procgen_env:
        return np.array([action])
    return action

# def run_episode_with_steering_and_save_as_gif(env, model, steering_vector,steering_layer, filepath='../gifs/run.gif', save_gif=False, episode_timeout=200, is_procgen_env=True):
#     observations = []
#     observation = env.reset()
#     done = False
#     total_reward = 0
#     frames = []
#     count = 0

#     while not done:
#         if save_gif:
#             frames.append(env.render(mode='rgb_array'))

#         action = generate_action_with_steering(model, observation, steering_vector, steering_layer, is_procgen_env=is_procgen_env)

#         # Perform inference with the model one step
#         observation, reward, done, info = env.step(action)
#         total_reward += reward
#         observations.append(observation)
#         count += 1

#         if count >= episode_timeout:
#             break

#     if save_gif:
#         imageio.mimsave(filepath, frames, fps=30)

#     return total_reward, frames, observations