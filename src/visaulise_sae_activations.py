# %%

import procgen_tools
import utils.helpers as helpers
import utils.heist as heist
import torch
import random
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from utils.sae_utils import load_sae_from_checkpoint, ordered_layer_names, get_sae_activations, replace_layer_with_sae
from sae_cnn import ConvSAE

# %%
# Load model and setup environment
difficulty = 'easy'
model = helpers.load_interpretable_model(model_path=f"../model_interpretable.pt")
model_activations = helpers.ModelActivations(model)
layer_paths = helpers.get_model_layer_names(model)
# %%

# Main workflow example: Remove entities and plot differences
venv = heist.create_venv(1, 15, 1, 0, difficulty)
state = heist.state_from_venv(venv, 0)


# %%

env_name = "procgen:procgen-heist-v0"  

env = gym.make(env_name, start_level=100, num_levels=200, render_mode="rgb_array", distribution_mode="easy") #remove render mode argument to go faster but not produce images 

model = helpers.load_interpretable_model(model_path="../model_interpretable.pt")

# %%
layers = {
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

# %%

venv = heist.create_venv(1, random.randint(0,10000), 1)
observation = venv.reset()

model_activations = helpers.ModelActivations(model)

output1, normal_activations = model_activations.run_with_cache(helpers.observation_to_rgb(observation), list(layers.values()))

# %%

# Set the checkpoint path and layer info
sae_checkpoint_path = "../checkpoints/sae_checkpoint_step_4500000.pt"
layer_number = 8  # conv4a
layer_name = ordered_layer_names[layer_number]

# %%


# Load the SAE model
if sae_checkpoint_path and os.path.exists(sae_checkpoint_path):
    # Load the SAE model
    print(f"Loading ConvSAE model from {sae_checkpoint_path}")
    sae = load_sae_from_checkpoint(sae_checkpoint_path)
    print(f"Successfully loaded ConvSAE model")
    
    # Get a new observation
    venv = heist.create_venv(1, random.randint(0, 10000), 1)
    observation = venv.reset()

    # Prepare storage for the activations
    layer_activations = []
    
    # Define hook function to capture layer activations
    def activation_hook(module, input, output):
        layer_activations.append(output.detach())

    converted_obs = helpers.observation_to_rgb(observation)
    
    handle = replace_layer_with_sae(model, sae, 8) 

    module = model
    elements = layer_name.split(".")
    for element in elements:
        if "[" in element and "]" in element:
            base, idx = element.split("[")
            idx = int(idx[:-1])
            module = getattr(module, base)[idx]
        else:
            module = getattr(module, element)

    # Capture both original and SAE activations
    sae_activations = []
    def hook_sae_activations(module, input, output):
        # For ConvSAE, capture activations after the encoder
        with torch.no_grad():
            _, _, acts, _ = sae(output)
            sae_activations.append(acts)
        return output  # Return original output to not modify forward pass

    # Register hook and run model
    handle = module.register_forward_hook(hook_sae_activations)
    with torch.no_grad():
        outputs = model(converted_obs)
    handle.remove()  # Clean up hook

    # Plot the original observation
    plt.figure(figsize=(8, 8))
    plt.imshow(helpers.tensor_to_image(converted_obs))
    plt.title("Original Observation")
    plt.axis('off')
    plt.show()

    helpers.plot_layer_channels({f"{layer_name}_sae": sae_activations[0].squeeze()}, f"{layer_name}_sae")

    

# %%

