# %%
import utils.helpers as helpers
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sae_cnn import load_sae_from_checkpoint, ordered_layer_names
import imageio
from src.utils.environment_modification_experiments import (
    create_example_maze_sequence,
    run_custom_maze_sequence,
)


# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = helpers.load_interpretable_model(model_path=f"../model_interpretable.pt").to(device)

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


# Define activation hook
def hook_sae_activations(module, input, output):
    with torch.no_grad():
        output = output.to(device)  # Explicitly move the output to GPU
        sae.to(device)  # Explicitly ensure SAE is moved to GPU right here
        _, _, acts, _ = sae(output)
        sae_activations.append(acts.squeeze().cpu())
    return output



# Set the checkpoint path and layer info
sae_checkpoint_path = "checkpoints/layer_4_conv2b/sae_checkpoint_step_250000.pt"
layer_number = 8  # conv4a
layer_name = ordered_layer_names[layer_number]

module = get_module(model, layer_name)
sae_activations = []  # Clear previous activations
# Load the SAE model
assert os.path.exists(sae_checkpoint_path), "SAE checkpoint path not found"
print(f"Loading ConvSAE model from {sae_checkpoint_path}")
sae = load_sae_from_checkpoint(sae_checkpoint_path).to(device)
print(f"Successfully loaded ConvSAE model")





# Prepare storage
all_frames = []
all_sae_activations = []



# Load your main model and ensure it's on device

# Generate custom maze observations




# # Run through environment and collect frames
# total_reward, frames, observations = helpers.run_episode_and_save_as_gif(
#     venv, 
#     model,
#     filepath="../environment_run.gif",
#     save_gif=True,
#     episode_timeout=200,
#     is_procgen_env=True
# )
# venv.close()

# handle.remove()
# helpers.plot_single_observation(torch.tensor(observations[0]).squeeze())



entity_pairs = [
    (3, 4),  # gem, blue key
    (4, 5),  # blue key, green key
    (5, 6),   # green key, red key
    (4, 3),  # blue key, gem
    (5, 4),  # green key, blue key
    (6, 5)   # red key, green key
]



# maze_patterns = create_example_maze_sequence()
# observations, venv = maze_patterns


handle = module.register_forward_hook(hook_sae_activations)

# Process each entity pair
for entity1_value, entity2_value in entity_pairs:

    entity_names = {
        3: "gem", 4: "blue_key", 5: "green_key", 6: "red_key"
    }
    entity1_name = entity_names[entity1_value]
    entity2_name = entity_names[entity2_value]
    print(f"\nProcessing entity pair: {entity1_name} and {entity2_name}")
    
    # Create maze sequence with the current entity pair values
    maze_patterns = create_example_maze_sequence(entity1_value, entity2_value)
    observations, venv = maze_patterns
    
    # Clear previous data
    all_frames = []
    all_sae_activations = []
    
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
    for i, (frame, sae_act) in enumerate(zip(observations[:15], all_sae_activations)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.imshow(frame.squeeze().transpose(1,2,0))
        ax1.set_title(f"Original Observation: {entity1_name} and {entity2_name}")
        ax1.axis("off")

        # Get the activation plot as a figure
        act_fig = helpers.plot_layer_channels(
            {f"{layer_name}_sae": sae_act}, f"{layer_name}_sae", return_image=True
        )
        
        # Convert activation figure to image array
        act_fig.canvas.draw()
        act_image = np.frombuffer(act_fig.canvas.tostring_rgb(), dtype=np.uint8)
        act_image = act_image.reshape(act_fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(act_fig)
        
        # Display activation image in second subplot
        ax2.imshow(act_image)
        ax2.axis("off")

        # Convert full figure to image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        viz_frames.append(image)
        plt.close(fig)

    # Save final GIF with entity pair in filename
    output_filename = f"maze_activation_visualisations/conv2b_maze_sequence_{entity1_name}_{entity2_name}_activations.gif"
    imageio.mimsave(output_filename, viz_frames, fps=2)
    print(f"Saved maze sequence with {entity1_name} and {entity2_name} activations as {output_filename}")
    # Close the environment
    venv.close()



# %%

import importlib
import src.utils.environment_modification_experiments as env_mod

# After making changes to the module
importlib.reload(env_mod)

# Add this code at the end of your file

# Import the specific maze creation function
from src.utils.environment_modification_experiments import create_specific_l_shaped_maze_env

# Define maze variants with their directions based on the comments in the code
maze_variants = [
    {"variant": 0, "direction": "right", "description": "Original L-shaped maze (player bottom-left, target top-right)"},
    {"variant": 1, "direction": "right", "description": "Flipped horizontally (player bottom-right, target top-left)"},
    {"variant": 2, "direction": "up", "description": "Flipped vertically (player top-left, target bottom-right)"},
    {"variant": 3, "direction": "up", "description": "Rotated 180 degrees (player top-right, target bottom-left)"},
    {"variant": 4, "direction": "left", "description": "S-shaped path (player bottom-left, target top-right)"},
    {"variant": 5, "direction": "left", "description": "Inverted L (player bottom-left, target mid-right)"},
    {"variant": 6, "direction": "down", "description": "T-junction (player bottom, target top-right)"},
    {"variant": 7, "direction": "down", "description": "U-shaped path (player bottom-left, target bottom-right)"}
]

# Create a figure with all maze variants
def visualize_all_maze_variants():
    """Creates a figure with all maze variants and their directions in the title"""
    # Create a grid of subplots - 2 rows, 4 columns for 8 variants
    n_rows, n_cols = 2, 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    
    maze_observations = {}
    
    for i, variant_info in enumerate(maze_variants):
        # Create the specific maze environment
        venv = create_specific_l_shaped_maze_env(maze_variant=variant_info['variant'])
        
        # Get observation
        obs = venv.reset()
        maze_observations[f"variant_{variant_info['variant']}"] = obs[0]
        
        # Plot in the corresponding subplot
        ax = axes[i]
        print(obs.shape)
        ax.imshow(obs[0].squeeze().transpose(1,2,0))
        
        # Add title with variant number and direction
        ax.set_title(f"Variant {variant_info['variant']}: {variant_info['direction'].upper()}\n{variant_info['description']}", 
                    fontsize=10)
        
        ax.axis('off')
        venv.close()
    
    plt.tight_layout()
    plt.savefig("all_maze_variants.png", dpi=150)
    plt.show()
    
    return maze_observations

# Visualize all maze variants in a single figure
maze_observations = visualize_all_maze_variants()

# Now let's analyze SAE activations for each maze variant
def analyze_maze_variant_activations(channel_to_analyze=0):
    """Analyzes SAE activations for all maze variants"""
    # Create a grid of subplots - 2 rows, 4 columns for 8 variants
    n_rows, n_cols = 2, 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, variant_info in enumerate(maze_variants):
        # Create the maze environment
        venv = create_specific_l_shaped_maze_env(maze_variant=variant_info['variant'])
        obs = venv.reset()
        
        # Get SAE activations
        sae_activations = []
        
        # Define activation hook
        def hook_sae_activations(module, input, output):
            with torch.no_grad():
                output = output.to(device)
                sae.to(device)
                _, _, acts, _ = sae(output)
                sae_activations.append(acts.squeeze().cpu())
            return output
        
        # Register hook
        module = get_module(model, layer_name)
        handle = module.register_forward_hook(hook_sae_activations)
        
        # Run model inference
        with torch.no_grad():
            converted_obs = helpers.observation_to_rgb(obs[0])
            obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(device)
            model(obs_tensor.unsqueeze(0))
        
        handle.remove()
        
        # Plot the observation and channel activation
        ax = axes[i]
        
        # If we have activations, plot the specified channel
        if len(sae_activations) > 0:
            channel_act = sae_activations[0][channel_to_analyze].numpy()
            
            # Create a small subplot within the main subplot for the observation
            ax_obs = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
            ax_obs.imshow(obs[0].transpose(1,2,0))
            ax_obs.axis('off')
            
            # Plot the channel activation in the main subplot
            im = ax.imshow(channel_act, cmap='viridis')
            ax.set_title(f"Variant {variant_info['variant']}: {variant_info['direction'].upper()}", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No activations", ha='center', va='center')
        
        ax.axis('off')
        venv.close()
    
    plt.suptitle(f"Channel {channel_to_analyze} Activations Across Maze Variants", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"channel_{channel_to_analyze}_across_maze_variants.png", dpi=150)
    plt.show()

# Analyze activations for the first few channels
for channel in range(5):
    analyze_maze_variant_activations(channel)

# %%
# Let's also find the top activating channels for each maze variant

def find_top_channels_for_maze_variant(variant_idx, top_k=8):
    """Finds the top activating channels for a specific maze variant"""
    variant_info = maze_variants[variant_idx]
    
    # Create the maze environment
    venv = create_specific_l_shaped_maze_env(maze_variant=variant_info['variant'])
    obs = venv.reset()
    
    # Get SAE activations
    sae_activations = []
    
    # Define activation hook
    def hook_sae_activations(module, input, output):
        with torch.no_grad():
            output = output.to(device)
            sae.to(device)
            _, _, acts, _ = sae(output)
            sae_activations.append(acts.squeeze().cpu())
        return output
    
    # Register hook
    module = get_module(model, layer_name)
    handle = module.register_forward_hook(hook_sae_activations)
    
    # Run model inference
    with torch.no_grad():
        converted_obs = helpers.observation_to_rgb(obs[0])
        obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(device)
        model(obs_tensor.unsqueeze(0))
    
    handle.remove()
    venv.close()
    
    # If we have activations, find the top channels
    if len(sae_activations) > 0:
        # Calculate sum of activations for each channel
        channel_sums = torch.sum(sae_activations[0], dim=(1, 2))
        
        # Get top-k channels
        top_channels = torch.argsort(channel_sums, descending=True)[:top_k]
        
        print(f"Top {top_k} channels for Variant {variant_info['variant']} ({variant_info['direction'].upper()}):")
        for i, channel in enumerate(top_channels):
            print(f"  {i+1}. Channel {channel.item()}: {channel_sums[channel].item():.4f}")
        
        return top_channels.tolist(), channel_sums[top_channels].tolist()
    else:
        print(f"No activations for Variant {variant_info['variant']}")
        return [], []

# Find top channels for each maze variant
top_channels_by_variant = {}
for i in range(len(maze_variants)):
    top_channels, top_sums = find_top_channels_for_maze_variant(i)
    top_channels_by_variant[i] = {"channels": top_channels, "means": top_sums}

# %%
# Let's visualize the top channel for each direction type

# Group variants by direction
direction_groups = {}
for variant_info in maze_variants:
    direction = variant_info["direction"]
    if direction not in direction_groups:
        direction_groups[direction] = []
    direction_groups[direction].append(variant_info["variant"])

# For each direction, visualize the top channel
for direction, variants in direction_groups.items():
    # Create a figure for this direction
    n_variants = len(variants)
    fig, axes = plt.subplots(1, n_variants, figsize=(5 * n_variants, 5))
    if n_variants == 1:
        axes = [axes]
    
    for i, variant_idx in enumerate(variants):
        variant_info = maze_variants[variant_idx]
        
        # Get the top channel for this variant
        top_channel = top_channels_by_variant[variant_idx]["channels"][0] if top_channels_by_variant[variant_idx]["channels"] else 0
        
        # Create the maze environment
        venv = create_specific_l_shaped_maze_env(maze_variant=variant_info['variant'])
        obs = venv.reset()
        
        # Get SAE activations for the top channel
        sae_activations = []
        
        # Define activation hook
        def hook_sae_activations(module, input, output):
            with torch.no_grad():
                output = output.to(device)
                sae.to(device)
                _, _, acts, _ = sae(output)
                sae_activations.append(acts.squeeze().cpu())
            return output
        
        # Register hook
        module = get_module(model, layer_name)
        handle = module.register_forward_hook(hook_sae_activations)
        
        # Run model inference
        with torch.no_grad():
            converted_obs = helpers.observation_to_rgb(obs[0])
            obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(device)
            model(obs_tensor)
        
        handle.remove()
        venv.close()
        
        # Plot the observation and channel activation
        ax = axes[i]
        
        # If we have activations, plot the top channel
        if len(sae_activations) > 0:
            channel_act = sae_activations[0][top_channel].numpy()
            
            # Create a small subplot within the main subplot for the observation
            ax_obs = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
            ax_obs.imshow(obs[0])
            ax_obs.axis('off')
            
            # Plot the channel activation in the main subplot
            im = ax.imshow(channel_act, cmap='viridis')
            ax.set_title(f"Variant {variant_info['variant']}\nTop Channel: {top_channel}", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No activations", ha='center', va='center')
        
        ax.axis('off')
    
    plt.suptitle(f"Top Channel Activations for {direction.upper()} Direction Mazes", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"top_channel_{direction}_direction.png", dpi=150)
    plt.show()

# %%
# Let's modify the strongest activating part of channel 0 in each maze variant and see how it affects behavior

def modify_channel_in_maze_variant(variant_idx, channel_to_modify=0, modification_factor=2.0):
    """Modifies the strongest activating part of a channel in a maze variant"""
    variant_info = maze_variants[variant_idx]
    
    # Create the maze environment
    venv = create_specific_l_shaped_maze_env(maze_variant=variant_info['variant'])
    obs = venv.reset()
    
    # Find strongest activating region for this channel
    top_regions, _ = find_strongest_activating_regions(
        model=model,
        sae=sae,
        obs=obs[0],
        layer_name=layer_name,
        channel_to_modify=channel_to_modify,
        top_k=1
    )
    
    # Modify this region and run episode
    print(f"Running agent in Variant {variant_info['variant']} ({variant_info['direction'].upper()}) with modified channel {channel_to_modify}")
    frames = modify_sae_spatial_activations(
        model=model,
        sae=sae,
        venv=venv,
        layer_name=layer_name,
        channel_to_modify=channel_to_modify,
        modification_factor=modification_factor,
        threshold=0.3,  # Lower threshold to ensure we catch activations
        target_region=top_regions[0] if top_regions else None,
        region_size=3
    )
    
    venv.close()
    return frames

# Test modifying channel 0 in each maze variant
for i in range(len(maze_variants)):
    modify_channel_in_maze_variant(i, channel_to_modify=0, modification_factor=2.0)


# %%

def find_top_channels_for_maze_variant_base_model(variant_idx, top_k=8):
    """Finds the top activating channels in conv4a for a specific maze variant using the base model"""
    variant_info = maze_variants[variant_idx]
    
    # Create the maze environment
    venv = create_specific_l_shaped_maze_env(maze_variant=variant_info['variant'])
    obs = venv.reset()
    
    # Get conv4a activations
    conv3a_activations = []
    
    # Define activation hook
    def hook_conv3a_activations(module, input, output):
        with torch.no_grad():
            # Store the raw activations from conv4a
            conv3a_activations.append(output.squeeze().cpu())
        return output
    
    # Register hook for conv4a layer
    layer_name = "conv3a"  # This is the layer we want to analyze
    module = get_module(model, layer_name)
    handle = module.register_forward_hook(hook_conv3a_activations)
    
    # Run model inference
    with torch.no_grad():
        converted_obs = helpers.observation_to_rgb(obs[0])
        obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(device)
        model(obs_tensor.unsqueeze(0))
    
    handle.remove()
    venv.close()
    
    # If we have activations, find the top channels
    if len(conv3a_activations) > 0:
        # Calculate mean activation for each channel
        channel_means = torch.mean(conv3a_activations[0], dim=(1, 2))
        
        # Get top-k channels
        top_channels = torch.argsort(channel_means, descending=True)[:top_k]
        
        print(f"Top {top_k} channels in conv4a for Variant {variant_info['variant']} ({variant_info['direction'].upper()}):")
        for i, channel in enumerate(top_channels):
            print(f"  {i+1}. Channel {channel.item()}: {channel_means[channel].item():.4f}")
        
        return top_channels.tolist(), channel_means[top_channels].tolist()
    else:
        print(f"No activations for Variant {variant_info['variant']}")
        return [], []

# Find top channels for each maze variant using the base model
top_channels_by_variant_base_model = {}
for i in range(len(maze_variants)):
    top_channels, top_means = find_top_channels_for_maze_variant_base_model(i)
    top_channels_by_variant_base_model[i] = {"channels": top_channels, "means": top_means}

# %%
# Let's visualize all permutations of entity types in different maze patterns and SAE layers
def visualize_all_permutations(save_dir="visualisations"):
    """
    Visualizes all permutations of different entity types in maze patterns
    across different SAE layers and saves the results.
    
    Args:
        save_dir (str): Root directory to save visualizations
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Import required functions and modules 
    from src.utils.environment_modification_experiments import create_two_entity_maze
    import utils.helpers as helpers
    import torch
    import time
    
    # Load the interpretable model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = helpers.load_interpretable_model(model_path=f"../model_interpretable.pt").to(device)
    
    # Entity types from the entity_type_map
    entity_types = ["blue_key", "red_key", "green_key", "gem"]
    
    # Target SAE layers - these are the ones you specifically mentioned
    target_layers = {
        1: "conv1a",  # layer_1_conv1a
        3: "conv2a",  # layer_3_conv2a
        4: "conv2b",  # layer_4_conv2b
        6: "conv3a",  # layer_6_conv3a
        8: "conv4a",  # layer_8_conv4a
    }
    
    # Get module by layer name
    def get_module(model, layer_name):
        module = model
        for element in layer_name.split("."):
            if "[" in element:
                base, idx = element.rstrip("]").split("[")
                module = getattr(module, base)[int(idx)]
            else:
                module = getattr(module, element)
        return module
    
    # Use the patterns defined in create_two_entity_maze
    # Extract patterns from the function source
    import inspect
    import re
    source = inspect.getsource(create_two_entity_maze)
    
    # Find the patterns
    patterns = []
    pattern_matches = re.findall(r'pattern\d+\s*=\s*np\.array\(\[(.*?)\]\)', source, re.DOTALL)
    for i, pattern_match in enumerate(pattern_matches[:3]):  # Limit to 3 patterns for simplicity
        pattern_str = f"np.array([{pattern_match}])"
        pattern = eval(pattern_str)
        patterns.append((f"pattern_{i+1}", pattern))
    
    # Start permutation visualization
    total_permutations = len(entity_types) * len(entity_types) * len(patterns) * len(target_layers)
    print(f"Starting visualization of {total_permutations} permutations...")
    start_time = time.time()
    
    # Process each layer
    for layer_number, layer_short_name in target_layers.items():
        layer_name = ordered_layer_names[layer_number]
        print(f"\nProcessing layer {layer_number}: {layer_name}")
        
        # Create layer output directory
        layer_dir = os.path.join(save_dir, f"layer_{layer_number}_{layer_short_name}")
        os.makedirs(layer_dir, exist_ok=True)
        
        # Load the SAE for this layer
        sae_checkpoint_path = f"checkpoints/layer_{layer_number}_{layer_short_name}/sae_checkpoint_step_300000.pt"
        
        # Skip if checkpoint doesn't exist
        if not os.path.exists(sae_checkpoint_path):
            print(f"Skipping layer {layer_number}: checkpoint not found at {sae_checkpoint_path}")
            continue
            
        print(f"Loading SAE from {sae_checkpoint_path}")
        sae = load_sae_from_checkpoint(sae_checkpoint_path).to(device)
        
        # Get the module for this layer
        module = get_module(model, layer_name)
        
        # Process each entity permutation
        for entity1_type in entity_types:
            for entity2_type in entity_types:
                # Create entity permutation directory
                entity_dir = os.path.join(layer_dir, f"{entity1_type}_{entity2_type}")
                os.makedirs(entity_dir, exist_ok=True)
                
                for pattern_name, pattern in patterns:
                    print(f"  Creating {entity1_type}/{entity2_type} in {pattern_name}")
                    
                    # Create the maze with these entities
                    obs, venv = create_two_entity_maze(pattern, entity1_type=entity1_type, entity2_type=entity2_type)
                    
                    # Get SAE activations
                    sae_activations = []
                    
                    # Define activation hook
                    def hook_sae_activations(module, input, output):
                        with torch.no_grad():
                            output = output.to(device)
                            sae.to(device)
                            _, _, acts, _ = sae(output)
                            sae_activations.append(acts.squeeze().cpu())
                        return output
                    
                    # Register hook, run model inference, remove hook
                    handle = module.register_forward_hook(hook_sae_activations)
                    with torch.no_grad():
                        converted_obs = helpers.observation_to_rgb(obs)
                        obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(device)
                        model(obs_tensor)
                    handle.remove()
                    
                    # Create visualization
                    if len(sae_activations) > 0:
                        # Create a figure with the observation and SAE activations
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                        
                        # Original observation
                        ax1.imshow(obs.squeeze().transpose(1, 2, 0))
                        ax1.set_title(f"Maze with {entity1_type} and {entity2_type}")
                        ax1.axis("off")
                        
                        # SAE activations
                        act_fig = helpers.plot_layer_channels(
                            {f"{layer_name}_sae": sae_activations[0]}, 
                            f"{layer_name}_sae", 
                            return_image=True
                        )
                        
                        # Convert activation figure to image array
                        act_fig.canvas.draw()
                        act_image = np.frombuffer(act_fig.canvas.tostring_rgb(), dtype=np.uint8)
                        act_image = act_image.reshape(act_fig.canvas.get_width_height()[::-1] + (3,))
                        plt.close(act_fig)
                        
                        # Display activation image in second subplot
                        ax2.imshow(act_image)
                        ax2.set_title(f"SAE Activations for layer {layer_short_name}")
                        ax2.axis("off")
                        
                        # Save the figure
                        save_path = os.path.join(entity_dir, f"{pattern_name}.png")
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    
                    # Close environment to free resources
                    venv.close()
    
    # Report completion
    elapsed_time = time.time() - start_time
    print(f"\nCompleted all {total_permutations} permutations in {elapsed_time:.2f} seconds")
    print(f"Results saved to {save_dir}")

# Run the visualization function
# Uncomment the line below to execute:
visualize_all_permutations()