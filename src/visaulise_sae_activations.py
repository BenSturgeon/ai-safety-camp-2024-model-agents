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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# %%
# Load model and setup environment
difficulty = 'easy'
model = helpers.load_interpretable_model(model_path=f"../model_interpretable.pt")
model_activations = helpers.ModelActivations(model)
layer_paths = helpers.get_model_layer_names(model)
# %%

venv = heist.create_venv(1, 15, 1, 0, difficulty)
state = heist.state_from_venv(venv, 0)


# %%

env_name = "procgen:procgen-heist-v0"  

env = gym.make(env_name, start_level=100, num_levels=200, render_mode="rgb_array", distribution_mode="easy") 

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

    # converted_obs = helpers.observation_to_rgb(observation)
    print()    
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

    helpers.plot_layer_channels({f"{layer_name}_sae": sae_activations[0].squeeze()}, f"{layer_name}_sae", layout=(4,4))

    

# %%

# Run the example maze sequence
from src.utils.environment_modification_experiments import create_example_maze_sequence, run_custom_maze_sequence

# Get the maze patterns from the example
maze_patterns = create_example_maze_sequence()

# Run the sequence and get the observations
observations, venv = maze_patterns  # This unpacks what create_example_maze_sequence returns

# If you want to save the sequence as a GIF
from src.utils.environment_modification_experiments import run_custom_maze_sequence

gif_observations = run_custom_maze_sequence(
    maze_patterns=maze_patterns[0],  # The first element contains the maze patterns
    save_path="example_maze_sequence",
    render_as_gif=True,
    fps=3
)

# %%
# Convert observations to RGB frames and store in list
frames = []
for obs in observations:
    rgb_frame = helpers.tensor_to_image(obs)
    frames.append(rgb_frame)

# Display each frame in sequence
for i, frame in enumerate(frames):
    plt.figure(figsize=(8, 8))
    plt.imshow(frame)
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()
import imageio
# Save as GIF
frames_uint8 = [(frame * 255).astype('uint8') if frame.dtype == float else frame for frame in frames]
imageio.mimsave("maze_sequence.gif", frames_uint8, fps=2)
# %%







# Load the SAE model
if sae_checkpoint_path and os.path.exists(sae_checkpoint_path):
    print(f"Loading ConvSAE model from {sae_checkpoint_path}")
    sae = load_sae_from_checkpoint(sae_checkpoint_path).to(device)
    print(f"Successfully loaded ConvSAE model")

    # Get the maze patterns from the example
    maze_patterns = create_example_maze_sequence()
    observations, venv = maze_patterns

    # Prepare storage for activations
    all_sae_activations = []
    all_frames = []
    model = model.to(device)
    sae = sae.to(device)  # Move SAE to same device as model

    # Process each observation
    for obs in observations:
        # Convert observation to RGB
        # obs = torch.tensor(obs)
        converted_obs = helpers.observation_to_rgb(obs)

        all_frames.append(helpers.tensor_to_image(converted_obs))
        
        # Get SAE activations

        module = model
        elements = layer_name.split(".")
        for element in elements:
            if "[" in element and "]" in element:
                base, idx = element.split("[")
                idx = int(idx[:-1])
                module = getattr(module, base)[idx].to(device)
            else:
                module = getattr(module, element).to(device)

        sae_activations = []
        def hook_sae_activations(module, input, output):
            with torch.no_grad():
                _, _, acts, _ = sae(output)
                sae_activations.append(acts)
            return output

        # Register hook and run model
        handle = module.register_forward_hook(hook_sae_activations)
        with torch.no_grad():
            converted_obs = torch.tensor(converted_obs, dtype=torch.float32).to(device)
            model = model.to(device)
            sae = sae.to(device)  # Move SAE to same device as model
            outputs = model(converted_obs)
        handle.remove()
        
        all_sae_activations.append(sae_activations[0].squeeze())

    # Create visualization frames
    viz_frames = []
    for i, (frame, sae_act) in enumerate(zip(all_frames, all_sae_activations)):
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot original frame
        ax1.imshow(frame)
        ax1.set_title("Original Observation")
        ax1.axis('off')
        
        # Plot SAE activations
        helpers.plot_layer_channels({f"{layer_name}_sae": sae_act}, f"{layer_name}_sae", )
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        viz_frames.append(image)
        plt.close()

    # Save as GIF
    imageio.mimsave("maze_sequence_with_activations.gif", viz_frames, fps=2)
    plt.close('all')
# %%


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from src.utils.environment_modification_experiments import (
    create_example_maze_sequence,
    run_custom_maze_sequence,
)


# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = helpers.load_interpretable_model(model_path=f"../model_interpretable.pt").to(device)
# Set the checkpoint path and layer info
sae_checkpoint_path = "../checkpoints/sae_checkpoint_step_4500000.pt"
layer_number = 8  # conv4a
layer_name = ordered_layer_names[layer_number]


# Load the SAE model
assert os.path.exists(sae_checkpoint_path), "SAE checkpoint path not found"
print(f"Loading ConvSAE model from {sae_checkpoint_path}")
sae = load_sae_from_checkpoint(sae_checkpoint_path).to(device)
print(f"Successfully loaded ConvSAE model")

# Load your main model and ensure it's on device


# Generate custom maze observations
maze_patterns = create_example_maze_sequence()
observations, venv = maze_patterns

# # Run the custom maze sequence and save as GIF
# gif_observations = run_custom_maze_sequence(
#     maze_patterns=maze_patterns[0], save_path="example_maze_sequence", render_as_gif=True, fps=3
# )

# Prepare storage
all_frames = []
all_sae_activations = []

# Identify the target module dynamically
def get_module(model, layer_name):
    module = model
    for element in layer_name.split("."):
        if "[" in element:
            base, idx = element.rstrip("]").split("[")
            module = getattr(module, base)[int(idx)]
        else:
            module = getattr(module, element)
    return module

module = get_module(model, layer_name)

# Define activation hook
def hook_sae_activations(module, input, output):
    with torch.no_grad():
        output = output.to(device)  # Explicitly move the output to GPU
        sae.to(device)  # Explicitly ensure SAE is moved to GPU right here
        _, _, acts, _ = sae(output)
        sae_activations.append(acts.squeeze().cpu())
    return output

# Process each observation
for obs in observations:
    sae_activations = []  # Clear previous activations

    # Convert observation to RGB
    converted_obs = helpers.observation_to_rgb(obs)
    frame_image = helpers.tensor_to_image(converted_obs)
    all_frames.append(frame_image)

    # Register hook, run model inference, remove hook
    handle = module.register_forward_hook(hook_sae_activations)
    with torch.no_grad():
        converted_obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(device)
        model(converted_obs_tensor)
    handle.remove()

    # Store SAE activations
    all_sae_activations.append(sae_activations[0])

# Visualization
viz_frames = []
for i, (frame, sae_act) in enumerate(zip(all_frames, all_sae_activations)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(frame)
    ax1.set_title("Original Observation")
    ax1.axis("off")

    helpers.plot_layer_channels(
        {f"{layer_name}_sae": sae_act}, f"{layer_name}_sae", layout=(4, 4), ax=ax2
    )

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    viz_frames.append(image)
    plt.close(fig)

# Save final GIF
imageio.mimsave("maze_sequence_with_activations.gif", viz_frames, fps=2)
print("Saved maze sequence with activations as GIF.")