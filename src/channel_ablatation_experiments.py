# %%
import utils.helpers as helpers
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sae_cnn import load_sae_from_checkpoint, ordered_layer_names
import imageio
from src.utils.environment_modification_experiments import create_trident_maze
from utils.helpers import run_episode_and_get_final_state
from utils.heist import (
    EnvState,
    ENTITY_TYPES,
    KEY_COLORS
)
import copy

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GEM = 'gem'
KEY_BLUE = 'blue_key'
KEY_GREEN = 'green_key'
KEY_RED = 'red_key'
COLOR_IDX_TO_ENTITY_NAME = {
    0: KEY_BLUE,
    1: KEY_GREEN,
    2: KEY_RED,
}


FEATURES_TO_ZERO = [i for i in range(128) if i not in [75]]


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
            modified_acts = latent_acts.clone()
            num_features = modified_acts.shape[1]
            valid_indices = [idx for idx in FEATURES_TO_ZERO if 0 <= idx < num_features]
            if valid_indices:
                 modified_acts[:, valid_indices, :, :] = 0.0
            if len(valid_indices) != len(FEATURES_TO_ZERO):
                 print(f"Warning: Invalid feature indices in {FEATURES_TO_ZERO}. Used: {valid_indices}")

        # Store the potentially modified activations for visualization
        sae_activations.append(modified_acts.squeeze().cpu()) # Store modified acts

        # Decode using the potentially modified activations
        reconstructed_output = sae.decode(modified_acts)

        try:
            reconstructed_output = reconstructed_output.reshape_as(output)
        except RuntimeError as e:
             print(f"Warning: Could not reshape reconstructed output. Shapes: Recon={reconstructed_output.shape}, Original={output.shape}. Error: {e}")
             return output # Return original if reshape fails

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




handle = module.register_forward_hook(hook_sae_activations)

maze_patterns = create_trident_maze()
observations, venv = maze_patterns

# --- Get Initial Entities ---
initial_entities = set()

initial_state_bytes = venv.env.callmethod("get_state")[0]
initial_env_state = EnvState(initial_state_bytes)
if initial_env_state.entity_exists(ENTITY_TYPES["gem"]):
    initial_entities.add(GEM)
for color_idx, entity_name in COLOR_IDX_TO_ENTITY_NAME.items():
    if initial_env_state.entity_exists(ENTITY_TYPES["key"], color_idx):
            initial_entities.add(entity_name)
print(f"Initial entities present: {initial_entities}")
initial_entities = set() 



total_reward, frames, last_state_bytes, ended_by_gem, ended_by_timeout = run_episode_and_get_final_state(
    venv,
    model,
    filepath="../environment_run_ablated.gif", 
    save_gif=True, 
    episode_timeout=300,
    is_procgen_env=True
)

handle.remove()


final_entities = set()
if last_state_bytes:
    final_env_state = EnvState(last_state_bytes)
    final_state_vals = final_env_state.state_vals

    # Check for gem
    if final_env_state.count_entities(ENTITY_TYPES["gem"]) > 0:
        final_entities.add(GEM)
    
    # Check for keys based on count
    for color_idx, entity_name in COLOR_IDX_TO_ENTITY_NAME.items():
        if final_env_state.count_entities(ENTITY_TYPES["key"], color_idx) == 1:
            pass
        else:
            final_entities.add(entity_name)
    
    print(f"final_entities : {final_entities}")
    print(f"reward {total_reward}")
  

venv.close() # Ensure environment is closed after analysis



# %%

final_env_state.count_entities(2,0)
