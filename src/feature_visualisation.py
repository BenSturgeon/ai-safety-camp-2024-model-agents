# %% [markdown]
# # Full SAE Feature Visualization with Lucent
#
# 1. Load your RL CNN (via `load_interpretable_model()`).
# 2. Load a trained SAE checkpoint.
# 3. Replace the chosen layer in the RL CNN with the SAE.
# 4. Run Lucent's `render_vis` to visualize a chosen SAE channel in pixel space.

# %%
import sys, os
import torch as t
import torch.nn as nn
import numpy as np
import glob
import matplotlib.pyplot as plt

sys.path.insert(
    0, 
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.sae_cnn import ConvSAE, device  # your definitions
from src.utils.helpers import load_interpretable_model
from src.extract_sae_features import replace_layer_with_sae  # Or wherever you keep that
import lucent
from lucent.optvis import render, param, objectives


# %% [markdown]
# ## 1. Load the RL model & the SAE

# %%
model = load_interpretable_model()  # your RL CNN
model.to(device)
model.eval()

checkpoint_dir = "checkpoints"
layer_checkpoints = []
for layer_dir in os.listdir(checkpoint_dir):
    layer_path = os.path.join(checkpoint_dir, layer_dir)
    if os.path.isdir(layer_path):
        checkpoints = glob.glob(os.path.join(layer_path, "sae_checkpoint_step_*.pt"))
        if checkpoints:
            # pick the newest
            latest = max(checkpoints, key=lambda x: int(x.split("_step_")[-1].split(".")[0]))
            layer_checkpoints.append((layer_dir, latest))

if not layer_checkpoints:
    raise FileNotFoundError("No SAE checkpoints found in 'checkpoints/'")

# pick the newest or a specific layer
layer_id, checkpoint_path = max(layer_checkpoints, key=lambda x: os.path.getmtime(x[1]))
print(f"Loading SAE checkpoint from {checkpoint_path} for layer: {layer_id}")

checkpoint = t.load(checkpoint_path, map_location=device)
state_dict = checkpoint["model_state_dict"]

conv_enc_weight = state_dict["conv_enc.weight"]
in_channels = conv_enc_weight.shape[1]
hidden_channels = conv_enc_weight.shape[0]

sae = ConvSAE(
    in_channels=in_channels, 
    hidden_channels=hidden_channels, 
    l1_coeff=1e-5  # or whatever
)
sae.load_state_dict(state_dict)
sae.eval().to(device)

print(f"SAE in_channels={in_channels}, hidden_channels={hidden_channels}")

# %% [markdown]
# ## 2. Replace the target layer with the SAE
# 
# Suppose your `layer_id` is something like `layer_3_conv2a`. 
# You probably have a function or logic to map `layer_id` -> layer_number
# or a submodule name. We'll just assume we can do something like:

# %%
# If your replace_layer_with_sae expects an integer, parse from the layer_id
# or adapt to your usage. For example, your 'ordered_layer_names' might be used.
# For demonstration, let's assume `replace_layer_with_sae(model, 3, sae)`:
# or if `layer_id` is "layer_3_conv2a", parse out the "3" or so.

# Fictitious parse:
import re

layer_num_match = re.search(r"layer_(\d+)_", layer_id)
if not layer_num_match:
    raise ValueError(f"Could not parse layer number from: {layer_id}")
layer_number = int(layer_num_match.group(1))

# Actually do the replacement
revert_fn = replace_layer_with_sae(model, sae, layer_number)
print(f"Replaced layer #{layer_number} in the RL model with the SAE.")


# %% [markdown]
# ## 3. Visualize a specific SAE channel with Lucent
#
# After replacement, when we do `model.forward(img)`, the final output
# is `(batch_size, hidden_channels, H, W)` from the SAE.
# 
# So we just need to tell Lucent to "maximize channel X" in the **final** module.
# By default, Lucent looks for named layers in `model` with e.g. `model.features[...].conv`.
# 
# But we've effectively turned the entire model's output *into* the SAE's hidden activations.
#
# So the simplest approach: **Lucent objective** = "channel X in the final output".
#
# We'll define a small helper for that objective:

# %%
def final_channel_objective(channel_idx: int):
    # channel objective on the final output
    # shape is (N, hidden_channels, H, W)
    return objectives.channel("output", channel_idx)



# %% [markdown]
# ### But we need a named layer "output"
# 
# By default, Lucent is used with known "layer names" in typical torchvision models.
# If your `model` *directly* returns the SAE hidden acts, there's no submodule name to hook.
# 
# One simple trick: wrap your model in a small module so that the final
# activation appears as a named submodule. That way, Lucent can do its hooking.

# %%
class FinalWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}  # Dictionary to store features
        
        # Register hook for conv4a
        def hook_fn(module, input, output):
            self.features['model_conv4a'] = output
            
        self.model.conv4a.register_forward_hook(hook_fn)
    
    def forward(self, x):
        _ = self.model(x)
        return self.features['model_conv4a']

# Create and prepare the model
wrapped_model = FinalWrapper(model)
wrapped_model = wrapped_model.to(device)
wrapped_model.eval()  # Make sure it's in eval mode


# %%

from lucent.modelzoo.util import get_model_layers  # Add this import

class FinalWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}
        
        def hook_fn(module, input, output):
            self.features["model_conv4a"] = output  # Changed to underscore
            
        self.model.conv4a.register_forward_hook(hook_fn)
    
    def forward(self, x):
        # Run the model but ignore its output
        _ = self.model(x)
        # Return only the feature maps
        return self.features["model_conv4a"]

# Create and prepare model
wrapped_model = FinalWrapper(model)
wrapped_model = wrapped_model.to(device)
wrapped_model.eval()  # Important!

# Define transforms
def clamp_0_1_torch(x):
    return (x.clamp(0, 1),)

transform_f = [clamp_0_1_torch]

# Visualization
channel_idx = 0
obj = objectives.channel("model_conv4a", channel_idx)  # Using underscore, not dot

_ = render.render_vis(
    wrapped_model,
    obj,
    param_f=lambda: param.image(64),
    show_inline=True,
    transforms=transform_f
)
# %%
# Create the wrapper
wrapped_model = FinalWrapper(model)
wrapped_model = wrapped_model.to(device)
wrapped_model.eval()

# Create a test input
test_input = t.rand(1, 3, 64, 64).to(device)

# Run a forward pass
output = wrapped_model(test_input)

# The hook should have printed the features dictionary
# But we can also examine manually:
print(output)
# When finished, Lucent will show an inline animation and the final image.


# %% [markdown]
# ### Interpreting the Result
#
# - The final displayed image is an RGB pattern that, when fed into your RL CNN
#   up to `layer_number`, then passed to the SAE, strongly activates channel #0 of the SAE.
# - You can repeat for each channel index you care about, or store these images to disk.
# - If you want to do transformations or Fourier-based parameterizations, you can do:
#     ```
#     param_f=param.image(128, fft=True, decorrelate=True)
#     ```
#   for often nicer results.
#
# If you want to revert the model to its original state (removing the SAE),
# call `revert_fn()` that was returned by `replace_layer_with_sae`.


