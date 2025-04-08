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
    create_trident_maze
)
import copy # Import copy for deep copying activations if needed


# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Add list for features to zero out ---
# Define which SAE feature indices to zero out (e.g., [10, 25, 100])
# Set to [] to disable zeroing.
FEATURES_TO_ZERO = [i for i in range(128) if i not in [44]] 
# ---

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
sae_activations = [] # Global list to store activations from the hook
def hook_sae_activations(module, input, output):
    global sae_activations # Ensure we modify the global list
    with torch.no_grad():
        output = output.to(device)
        sae.to(device)

        # Encode to get latent activations
        latent_acts = sae.encode(output) # Shape (batch, features, H', W')

        # --- Zero out specified features ---
        modified_acts = latent_acts
        if FEATURES_TO_ZERO:
            # Clone to avoid modifying the original tensor if needed elsewhere,
            # though here we modify before decode anyway.
            modified_acts = latent_acts.clone()
            num_features = modified_acts.shape[1]
            valid_indices = [idx for idx in FEATURES_TO_ZERO if 0 <= idx < num_features]
            if valid_indices:
                 # Zero out features across batch, height, width dims
                 modified_acts[:, valid_indices, :, :] = 0.0
            if len(valid_indices) != len(FEATURES_TO_ZERO):
                 print(f"Warning: Invalid feature indices in {FEATURES_TO_ZERO}. Used: {valid_indices}")
        # ---

        # Store the potentially modified activations for visualization
        sae_activations.append(modified_acts.squeeze().cpu()) # Store modified acts

        # Decode using the potentially modified activations
        reconstructed_output = sae.decode(modified_acts)

        # Reshape reconstruction (if necessary, depends on SAE architecture)
        # This reshape might need adjustment based on your specific SAE's output shape vs the layer's output shape
        try:
            reconstructed_output = reconstructed_output.reshape_as(output)
        except RuntimeError as e:
             print(f"Warning: Could not reshape reconstructed output. Shapes: Recon={reconstructed_output.shape}, Original={output.shape}. Error: {e}")
             # Fallback or error handling needed here? For now, return original output on error.
             return output

    # Return the reconstruction based on (potentially modified) activations
    return reconstructed_output



# Set the checkpoint path and layer info
sae_checkpoint_path = "checkpoints/layer_8_conv4a/sae_checkpoint_step_15000000_vast.pt"
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




entity_pairs = [
    (3, 4),  # gem, blue key
    # (4, 5),  # blue key, green key
    # (5, 6),   # green key, red key
    # (4, 3),  # blue key, gem
    # (5, 4),  # green key, blue key
    # (6, 5)   # red key, green key
]



# maze_patterns = create_example_maze_sequence()
# observations, venv = maze_patterns


handle = module.register_forward_hook(hook_sae_activations)

maze_patterns = create_trident_maze()
observations, venv = maze_patterns


# Run through environment and collect frames
total_reward, frames, observations = helpers.run_episode_and_save_as_gif(
    venv, 
    model,
    filepath="../environment_run.gif",
    save_gif=True,
    episode_timeout=300,
    is_procgen_env=True
)
venv.close()


