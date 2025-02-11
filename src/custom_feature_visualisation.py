# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# # ================================
# # Your CustomCNN definition here
# # (As in your provided code)
# # ================================
# class CustomCNN(nn.Module):
#     def __init__(self, obs_space, num_outputs, conv_dropout_prob=0.01, fc_dropout_prob=0):
#         super(CustomCNN, self).__init__()

#         h, w, c = obs_space.shape
#         self.num_outputs = num_outputs

#         self.conv1a = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=7, padding=3)
#         self.pool1 = nn.LPPool2d(2, kernel_size=2, stride=2)

#         self.conv2a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
#         self.conv2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
#         self.pool2 = nn.LPPool2d(2, kernel_size=2, stride=2)

#         self.conv3a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
#         self.pool3 = nn.LPPool2d(2, kernel_size=2, stride=2)

#         self.conv4a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
#         self.pool4 = nn.LPPool2d(2, kernel_size=2, stride=2)

#         # Compute the flattened dimension after convolutions and pooling
#         self.flattened_dim = self._get_flattened_dim(h, w)

#         self.fc1 = nn.Linear(in_features=self.flattened_dim, out_features=256)
#         self.fc2 = nn.Linear(in_features=256, out_features=512)
#         self.fc3 = nn.Linear(in_features=512, out_features=num_outputs)

#         self.value_fc = nn.Linear(in_features=512, out_features=1)

#         self.dropout_conv = nn.Dropout2d(p=conv_dropout_prob)
#         self.dropout_fc = nn.Dropout(p=fc_dropout_prob)

#     def _get_flattened_dim(self, h, w):
#         x = torch.zeros(1, 3, h, w)  # Dummy input to compute the shape
#         x = self.pool1(self.conv1a(x))
#         x = self.pool2(self.conv2b(self.conv2a(x)))
#         x = self.pool3(self.conv3a(x))
#         x = self.pool4(self.conv4a(x))
#         return x.numel()

#     def forward(self, obs):
#         assert obs.ndim == 4, f"Expected 4D input, got {obs.ndim}D"
#         if not isinstance(obs, torch.Tensor):
#             obs = torch.tensor(obs)

#         # Check for different input formats and convert if necessary
#         if obs.shape[1:] != (3, 64, 64):
#             if obs.shape[1:] == (64, 3, 64):  # NHWC format
#                 obs = obs.permute(0, 2, 1, 3)
#             elif obs.shape[1:] == (64, 64, 3):  # NHWC format
#                 obs = obs.permute(0, 3, 1, 2)

#         # Input range check and normalization
#         obs_min, obs_max = obs.min(), obs.max()
#         if obs_min < 0 or obs_max > 255:
#             raise ValueError(
#                 f"Input values out of expected range. Min: {obs_min}, Max: {obs_max}"
#             )
#         elif obs_max <= 1:
#             x = obs  # Already in [0, 1] range
#         else:
#             x = obs.float() / 255.0  # scale to 0-1

#         x = x.to(self.conv1a.weight.device)

#         x = torch.relu(self.conv1a(x))
#         x = self.pool1(x)
#         x = self.dropout_conv(x)

#         x = torch.relu(self.conv2a(x))
#         x = torch.relu(self.conv2b(x))
#         x = self.pool2(x)
#         x = self.dropout_conv(x)

#         x = torch.relu(self.conv3a(x))
#         x = self.pool3(x)
#         x = self.dropout_conv(x)

#         # Here is the fourth conv layer we want to visualize.
#         # We apply ReLU here (note: if you want the pre-activation, you can register the hook before relu)
#         x = torch.relu(self.conv4a(x))
#         x = self.pool4(x)
#         x = self.dropout_conv(x)

#         x = torch.flatten(x, start_dim=1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.dropout_fc(x)

#         logits = self.fc3(x)
#         dist = torch.distributions.Categorical(logits=logits)
#         value = self.value_fc(x)

#         return dist, value

#     def save_to_file(self, model_path):
#         torch.save(self.state_dict(), model_path)

#     def load_from_file(self, model_path, device):
#         self.load_state_dict(torch.load(model_path, map_location=device))

#     def get_state_dict(self):
#         return self.state_dict()

# # ====================================
# # Feature Visualization Implementation
# # ====================================

# def total_variation(x):
#     """
#     Total variation regularization helps enforce spatial smoothness.
#     """
#     tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
#     tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
#     return tv_h + tv_w

# def visualize_channel_activation(model, target_channel, num_steps=100, lr=0.1, reg_weight=1e-4, device='cpu'):
#     """
#     Optimizes an input image so that the activation of a specific channel (target_channel)
#     in the conv4a layer is maximized.

#     Parameters:
#       - model: an instance of CustomCNN
#       - target_channel: integer index of the channel to maximize in conv4a
#       - num_steps: number of optimization steps
#       - lr: learning rate for gradient ascent
#       - reg_weight: weight for the total variation regularization (helps smooth the image)
#       - device: 'cpu' or 'cuda'

#     Returns:
#       - A numpy array of shape (64, 64, 3) representing the optimized image.
#     """
#     model.eval()
#     model.to(device)

#     # Start with a random image in the range [0, 1] and ensure gradients are enabled.
#     input_img = torch.rand((1, 3, 64, 64), device=device, requires_grad=True)

#     optimizer = optim.Adam([input_img], lr=lr)

#     # Dictionary to store the activation of conv4a via a forward hook.
#     activations = {}

#     def hook_fn(module, input, output):
#         # The hook captures the output of conv4a (before pooling, but after we apply relu in forward)
#         activations['conv4'] = output

#     # Register the hook on conv4a.
#     hook_handle = model.conv4a.register_forward_hook(hook_fn)

#     for step in range(num_steps):
#         optimizer.zero_grad()

#         # Forward pass. (Our model's forward applies a relu on conv4a output.)
#         _ = model(input_img)

#         # Retrieve the activation from our hook. Note that our hook was on conv4a (before the relu in forward)
#         # so we apply relu manually to match the forward behavior.
#         act = torch.relu(activations['conv4'])
#         # Select the activation map for the target channel; shape: (H, W)
#         channel_activation = act[0, target_channel, :, :]

#         # Our loss is the negative mean activation. (Minimizing the negative is equivalent to maximizing the activation.)
#         loss = -channel_activation.mean()

#         # Add total variation regularization to enforce smoothness.
#         loss = loss + reg_weight * total_variation(input_img)

#         loss.backward()
#         optimizer.step()

#         # Optionally clamp the image to keep values between 0 and 1.
#         with torch.no_grad():
#             input_img.clamp_(0, 1)

#         if step % 10 == 0:
#             print(f"Step {step:03d} - Loss: {loss.item():.4f}")

#     # Remove the hook so it doesn't interfere with further computations.
#     hook_handle.remove()

#     # Convert the image to a numpy array for visualization.
#     result = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
#     return result

# # ====================================
# # Example Usage
# # ====================================

# # Create a dummy observation space object with the expected shape.
# class DummyObsSpace:
#     def __init__(self, shape):
#         self.shape = shape

# obs_space = DummyObsSpace((64, 64, 3))
# num_outputs = 10  # Example number of outputs for your network.
# model = CustomCNN(obs_space, num_outputs)

# # Set device (use 'cuda' if available)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # For example, visualize channel 0 of conv4a.
# target_channel = 0
# optimized_img = visualize_channel_activation(model,
#                                                target_channel=target_channel,
#                                                num_steps=100,   # Increase this number for better results
#                                                lr=0.1,
#                                                reg_weight=1e-4,
#                                                device=device)

# # Plot the resulting image.
# plt.figure(figsize=(4, 4))
# plt.imshow(optimized_img)
# plt.title(f"Maximized Activation for conv4a Channel {target_channel}")
# plt.axis('off')
# plt.show()



# %%

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models

# ------------------------------
# Load pretrained GoogLeNet (Inception V1)
# ------------------------------
model = models.googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# ------------------------------
# Total Variation Regularization
# ------------------------------
def total_variation(x):
    """
    Total variation regularization to encourage spatial smoothness.
    """
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ------------------------------
# Feature Visualization Function for Inception V1
# ------------------------------
def visualize_inception_channel(model, target_module_name, target_channel, num_steps=100, lr=0.1, reg_weight=1e-4, device='cpu'):
    """
    Optimize an input image to maximize the activation of a specific channel
    in a chosen module of a model (here, Inception V1).

    Parameters:
      - model: the pretrained model (GoogLeNet/Inception V1)
      - target_module_name: string name of the module (e.g., 'inception4a')
      - target_channel: integer index of the channel to maximize
      - num_steps: number of gradient ascent steps
      - lr: learning rate for the optimizer
      - reg_weight: weight for total variation regularization
      - device: 'cpu' or 'cuda'
      
    Returns:
      - result: a NumPy array (H x W x C) of the optimized image
    """
    model.eval()
    model.to(device)

    # GoogLeNet expects an image of size (1, 3, 224, 224)
    input_img = torch.rand((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    # Dictionary to store the activation from the target module.
    activations = {}
    
    def hook_fn(module, input, output):
        activations[target_module_name] = output

    # Register the forward hook on the target module.
    target_module = getattr(model, target_module_name)
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass: this will trigger the hook
        _ = model(input_img)
        
        # Get the activation from our hook.
        act = activations[target_module_name]
        # In case the module returns a tuple, pick the first element.
        if isinstance(act, tuple):
            act = act[0]
        # act is assumed to be of shape [1, C, H, W]. Extract the target channel.
        channel_activation = act[0, target_channel, :, :]
        
        # Define the loss: negative mean activation (to maximize activation)
        loss = -channel_activation.mean()
        # Add total variation regularization to promote smoothness.
        loss = loss + reg_weight * total_variation(input_img)
        
        loss.backward()
        optimizer.step()
        
        # Clamp the input image to keep values between 0 and 1.
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if step % 10 == 0:
            print(f"Step {step:03d} - Loss: {loss.item():.4f}")
    
    # Remove the hook after optimization.
    hook_handle.remove()
    
    # Convert the optimized image to a NumPy array (H x W x C)
    result = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return result

# ------------------------------
# Run Visualization on Inception V1
# ------------------------------
# For example, visualize channel 476 in the inception4a module.
target_module_name = 'inception4a'
target_channel = 476  # Ensure that this channel index exists in inception4a
optimized_img = visualize_inception_channel(model,
                                              target_module_name=target_module_name,
                                              target_channel=target_channel,
                                              num_steps=512,  # Increase steps for a more refined result
                                              lr=0.1,
                                              reg_weight=1e-4,
                                              device=device)

# Plot the resulting image.
plt.figure(figsize=(8, 8))
plt.imshow(optimized_img)
plt.title(f"Maximized Activation for {target_module_name} Channel {target_channel}")
plt.axis('off')
plt.show()

# %%

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim

# ------------------------------
# Load Pretrained GoogLeNet (Inception V1)
# ------------------------------
model = models.googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# ------------------------------
# Regularization Functions
# ------------------------------
def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

def jitter(img, ox, oy):
    """
    Randomly shift the image by (ox, oy) pixels.
    """
    return torch.roll(img, shifts=(ox, oy), dims=(2, 3))

# ------------------------------
# Feature Visualization Function with Improvements
# ------------------------------
def visualize_inception_channel(model, target_module_name, target_channel, num_steps=300, lr=0.05, 
                                tv_weight=1e-3, l2_weight=1e-3, jitter_amount=8, device='cpu'):
    """
    Optimize an input image to maximize the activation of a specific channel
    in a target module of the model, using additional regularizers and jitter.
    """
    model.eval()
    model.to(device)

    # GoogLeNet expects (1, 3, 224, 224)
    input_img = torch.rand((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    activations = {}
    
    def hook_fn(module, input, output):
        activations[target_module_name] = output

    target_module = getattr(model, target_module_name)
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply random jitter each step
        ox, oy = np.random.randint(-jitter_amount, jitter_amount+1, 2)
        jittered_img = jitter(input_img, ox, oy)
        
        _ = model(jittered_img)
        act = activations[target_module_name]
        if isinstance(act, tuple):
            act = act[0]
        channel_activation = act[0, target_channel, :, :]
        
        # Loss: negative activation (to maximize) plus regularizers
        loss = -channel_activation.mean()
        loss = loss + tv_weight * total_variation(input_img)
        loss = loss + l2_weight * torch.norm(input_img)
        
        loss.backward()
        optimizer.step()
        
        # Clamp image values to [0, 1]
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if step % 20 == 0:
            print(f"Step {step:03d} - Loss: {loss.item():.4f}")
    
    hook_handle.remove()
    
    result = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return result

# ------------------------------
# Run the Improved Visualization
# ------------------------------
target_module_name = 'inception4a'
target_channel = 475  # Ensure this index exists; adjust as needed
optimized_img = visualize_inception_channel(model,
                                              target_module_name=target_module_name,
                                              target_channel=target_channel,
                                              num_steps=300,  # More steps for a refined result
                                              lr=0.05,
                                              tv_weight=1e-2,
                                              l2_weight=5e-2,
                                              jitter_amount=8,
                                              device=device)

plt.figure(figsize=(8, 8))
plt.imshow(optimized_img)
plt.title(f"Optimized Activation for {target_module_name} Channel {target_channel}")
plt.axis('off')
plt.show()

# %%
import torch
from torchvision.models import googlenet
from lucent.optvis import render, objectives

# Load pretrained GoogLeNet/InceptionV1
model = googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# Example feature visualization for channel 476 in inception4a layer
objective = objectives.channel('inception4a', 476)  # Visualize channel 476 in inception4a layer
list_of_images = render.render_vis(model, objective)  # Generate visualization using Lucent's render_vis

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.imshow(list_of_images[0][0])  # Show first image from the list
plt.axis('off')
plt.title('Channel 476 in inception4a layer')
plt.show()

# %%

from lucent.optvis import render, param, transform
from lucent.modelzoo import inceptionv1

# Get Lucent's default transforms and their parameters
default_transforms = transform.standard_transforms
print("Default transforms:", default_transforms)

# Get default optimization parameters
default_params = param
print("\nDefault optimization parameters:", default_params)

# Run with verbose logging
objective = "mixed4a:476"
imgs = render.render_vis(model, objective, verbose=True)

# %%
from lucent.optvis import render, param, transform
from lucent.modelzoo.util import get_model_layers

# Print available layers
print("Available layers:")
print(get_model_layers(model))

# Get Lucent's default transforms and their parameters
default_transforms = transform.standard_transforms
print("\nDefault transforms:", default_transforms)

# Get default optimization parameters
default_params = param
print("\nDefault optimization parameters:", default_params)

# Run with verbose logging
objective = "inception4a:476"  # Using the correct layer name for GoogLeNet
imgs = render.render_vis(model, objective, verbose=True)

# %%
from lucent.optvis import render, param, transform
from lucent.modelzoo.util import get_model_layers

# Print available layers
print("Available layers:")
print(get_model_layers(model))

# Get Lucent's default transforms and their parameters
default_transforms = transform.standard_transforms
print("\nDefault transforms:", default_transforms)

# Let's look at the specific parameter attributes
print("\nOptimization parameters:")
print("Color correlation:", param.color_correlation_svd_sqrt)
print("FFT params:", param.rfft2d_freqs)
print("Spatial params:", param.spatial)

# Try to access any training-specific parameters
if hasattr(param, 'training'):
    print("\nTraining parameters:", param.training)

# Run with verbose logging to see parameters during execution
objective = "inception4a:476"
imgs = render.render_vis(model, objective, verbose=True)
# %%
from lucent.optvis import render, param, transform
from lucent.modelzoo.util import get_model_layers

# Print available layers
print("Available layers:")
print(get_model_layers(model))

# Let's look at the specific parameter attributes
print("\nOptimization parameters:")
print("Color correlation:", param.color_correlation_svd_sqrt)
print("FFT params:", param.rfft2d_freqs)
print("Spatial params:", param.spatial)

# Let's inspect the render.render_vis function defaults
print("\nRender function defaults:")
print("Render function:", render.render_vis.__defaults__)

# Run with modified parameters to expose more information
objective = "inception4a:476"
imgs = render.render_vis(
    model, 
    objective,
    thresholds=(1, 128, 256, 512),  # Add intermediate thresholds to see progress
    verbose=True,
    transforms=transform.standard_transforms,  # Explicitly use standard transforms
)

# Print transform information
print("\nStandard transforms:")
for t in transform.standard_transforms:
    print(t)
# %%
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

# ------------------------------
# Total Variation Regularization Function
# ------------------------------
def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ------------------------------
# Transform Function (Lucent's Sequence)
# ------------------------------
def apply_transforms(x, device):
    # 1. Pad with 12px (constant value 0.5)
    padded = F.pad(x, (12, 12, 12, 12), mode='constant', value=0.5)
    
    # 2. Jitter by up to 8px in each direction
    ox, oy = torch.randint(-8, 9, (2,), device=device)
    jittered = torch.roll(padded, shifts=(int(ox), int(oy)), dims=(2, 3))
    
    # 3. Random scale: choose a factor between 0.9 and 1.1
    scale = 1 + (torch.randint(0, 11, (1,), device=device).item() - 5) / 50.0
    scaled = F.interpolate(jittered, scale_factor=scale, mode='bilinear', align_corners=False)
    
    # 4. Random rotate between -10 and +10 degrees using an affine transformation
    angle = torch.randint(-10, 11, (1,), device=device).item()
    theta = torch.tensor([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
        [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0]
    ], device=device, dtype=torch.float).unsqueeze(0)
    grid = F.affine_grid(theta, scaled.size(), align_corners=False)
    rotated = F.grid_sample(scaled, grid, align_corners=False)
    
    # 5. Fine jitter by up to 4px in each direction
    ox, oy = torch.randint(-4, 5, (2,), device=device)
    final = torch.roll(rotated, shifts=(int(ox), int(oy)), dims=(2, 3))
    
    return final

# ------------------------------
# Visualization Function for Inception (using inception4a)
# ------------------------------
def visualize_inception_channel(model, target_channel, num_steps=512, lr=0.05, 
                                tv_weight=1e-2, l2_weight=5e-2, device='cpu'):
    """
    Optimizes a random input image to maximize the activation of a given channel
    in the inception4a module of the model, using a series of transforms plus
    total variation and L2 regularization.
    """
    model.eval()
    model.to(device)
    
    # Start with a random image of shape (1, 3, 64, 64)
    input_img = torch.rand((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    # Dictionary to store activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations['inception4a'] = output
    
    # Register a hook on the inception4a module
    hook_handle = model.inception4a.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply the transform sequence
        transformed_img = apply_transforms(input_img, device)
        
        # Forward pass (hook captures activations from inception4a)
        _ = model(transformed_img)
        
        # Select the target channel from the activations (apply ReLU)
        act = torch.relu(activations['inception4a'])
        channel_activation = act[0, target_channel, :, :]
        
        # Loss: negative mean activation plus regularization terms
        loss = -channel_activation.mean()
        loss += tv_weight * total_variation(input_img)
        loss += l2_weight * torch.norm(input_img)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if step % 20 == 0:
            print(f"Step {step:03d} - Loss: {loss.item():.4f}")
    
    hook_handle.remove()
    result = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return result

# ------------------------------
# Example Usage: Visualize a Channel in Inception's inception4a Layer
# ------------------------------
if __name__ == '__main__':
    # Load the pretrained GoogLeNet (Inception V1)
    model = models.googlenet(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Choose a target channel (ensure this index is valid for inception4a)
    target_channel = 475
    
    optimized_img = visualize_inception_channel(model, target_channel, 
                                                 num_steps=512, lr=0.05, 
                                                 tv_weight=1e-1, l2_weight=5e-2, 
                                                 device=device)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(optimized_img)
    plt.title(f"Optimized Activation for Inception4a Channel {target_channel}")
    plt.axis('off')
    plt.show()


# %%

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

# ------------------------------
# Total Variation Regularization
# ------------------------------
def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ------------------------------
# Apply Lucent-Inspired Transforms to a Batch
# ------------------------------
def apply_transforms_batch(x, device):
    """
    Apply a series of random transforms to each image in the batch.
    x: Tensor of shape [B, 3, H, W]
    Returns: transformed tensor of shape [B, 3, 64, 64]
    """
    B = x.shape[0]
    transformed_list = []
    for i in range(B):
        xi = x[i:i+1]  # [1, 3, H, W]
        # 1. Pad with 12px (constant value 0.5)
        padded = F.pad(xi, (12, 12, 12, 12), mode='constant', value=0.5)
        # 2. Jitter by up to ±8 pixels
        ox, oy = torch.randint(-8, 9, (2,), device=device)
        jittered = torch.roll(padded, shifts=(int(ox.item()), int(oy.item())), dims=(2, 3))
        # 3. Random scale: choose factor between 0.9 and 1.1
        scale = 1 + (torch.randint(0, 11, (1,), device=device).item() - 5) / 50.0
        scaled = F.interpolate(jittered, scale_factor=scale, mode='bilinear', align_corners=False)
        # 4. Random rotate: angle between -10 and +10 degrees
        angle = torch.randint(-10, 11, (1,), device=device).item()
        theta = torch.tensor([
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0]
        ], device=device, dtype=torch.float).unsqueeze(0)
        grid = F.affine_grid(theta, scaled.size(), align_corners=False)
        rotated = F.grid_sample(scaled, grid, align_corners=False)
        # 5. Fine jitter by up to ±4 pixels
        ox, oy = torch.randint(-4, 5, (2,), device=device)
        final = torch.roll(rotated, shifts=(int(ox.item()), int(oy.item())), dims=(2, 3))
        # Resize to fixed 64x64
        final = F.interpolate(final, size=(224, 224), mode='bilinear', align_corners=False)
        transformed_list.append(final)
    return torch.cat(transformed_list, dim=0)

# ------------------------------
# Diversity Loss (Based on Gram Matrix Cosine Similarity)
# ------------------------------
def diversity_loss(activations):
    """
    Given activations from the target layer for a batch of images,
    compute the average pairwise cosine similarity between the flattened Gram matrices.
    Lower similarity (i.e. more diversity) is better.
    
    activations: Tensor of shape [B, C, H, W]
    """
    B, C, H, W = activations.shape
    A = activations.view(B, C, -1)  # [B, C, H*W]
    G = torch.bmm(A, A.transpose(1, 2))  # [B, C, C]
    G_flat = G.view(B, -1)  # [B, C*C]
    
    loss_div = 0.0
    count = 0
    for i in range(B):
        for j in range(i+1, B):
            sim = torch.dot(G_flat[i], G_flat[j]) / (torch.norm(G_flat[i]) * torch.norm(G_flat[j]) + 1e-8)
            loss_div += sim
            count += 1
    if count > 0:
        loss_div = loss_div / count
    return loss_div

# ------------------------------
# Advanced Visualization Function (with Diversity and Warm Color Transform)
# ------------------------------
def visualize_inception_channel_diverse(model, target_channel, num_steps=1024, batch_size=4,
                                         lr=0.05, tv_weight=1e-2, l2_weight=5e-2,
                                         diversity_weight=1e-2, device='cpu'):
    """
    Optimize a batch of images to maximize the activation of a given channel in
    Inception's inception4a layer. In addition to the main objective, we add TV and L2
    regularization and a diversity term to encourage the images to show different facets.
    After optimization, we apply a warm color transform to bias the outputs toward yellows,
    browns, and reds.
    """
    model.eval()
    model.to(device)
    
    # Initialize a batch of random images (64x64)
    input_img = torch.rand((batch_size, 3, 64, 64), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    activations = {}
    def hook_fn(module, inp, out):
        activations['inception4a'] = out
    hook_handle = model.inception4a.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        transformed_img = apply_transforms_batch(input_img, device)
        _ = model(transformed_img)
        
        act = torch.relu(activations['inception4a'])
        main_loss = - torch.mean(act[:, target_channel, :, :])
        tv_loss = total_variation(input_img)
        l2_loss = torch.norm(input_img)
        div_loss = diversity_loss(act)
        
        total_loss = main_loss + tv_weight * tv_loss + l2_weight * l2_loss + diversity_weight * div_loss
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if step % 50 == 0:
            print(f"Step {step:04d} - Total Loss: {total_loss.item():.4f}, "
                  f"Main: {main_loss.item():.4f}, TV: {tv_loss.item():.4f}, "
                  f"L2: {l2_loss.item():.4f}, Div: {div_loss.item():.4f}")
    
    hook_handle.remove()
    
    # Post-process: Apply a warm color transform.
    # Here we boost the red channel and reduce the blue channel.
    with torch.no_grad():
        # scaling factors for R, G, B channels respectively
        warm_scale = torch.tensor([1.2, 1.05, 0.8], device=device).view(1, 3, 1, 1)
        input_img = input_img * warm_scale
        # Normalize each image individually to [0, 1]
        B = input_img.shape[0]
        for i in range(B):
            img = input_img[i]
            mi = img.min()
            ma = img.max()
            input_img[i] = (img - mi) / (ma - mi + 1e-8)
    
    return input_img.detach().cpu()  # shape: [batch_size, 3, 64, 64]

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == '__main__':
    model = models.googlenet(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    target_channel = 476
    optimized_batch = visualize_inception_channel_diverse(model, target_channel, 
                                                           num_steps=512, batch_size=4,
                                                           lr=0.05, tv_weight=1e-2,
                                                           l2_weight=5e-2, diversity_weight=1e-2,
                                                           device=device)
    
    batch_size = optimized_batch.shape[0]
    fig, axs = plt.subplots(1, batch_size, figsize=(4 * batch_size, 4))
    for i in range(batch_size):
        img = optimized_batch[i].permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.suptitle(f"Inception4a Channel {target_channel} - Diverse, Warm Optimizations", fontsize=16)
    plt.show()


# %%

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

# ------------------------------
# Total Variation Regularization
# ------------------------------
def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ------------------------------
# Apply Lucent-Inspired Transforms to a Batch
# ------------------------------
def apply_transforms_batch(x, device):
    """
    Apply a series of random transforms to each image in the batch.
    x: Tensor of shape [B, 3, H, W]
    Returns: transformed tensor of shape [B, 3, 224, 224]
    """
    B = x.shape[0]
    transformed_list = []
    for i in range(B):
        xi = x[i:i+1]  # [1, 3, H, W]
        # 1. Pad with 12px (constant value 0.5)
        padded = F.pad(xi, (12, 12, 12, 12), mode='constant', value=0.5)
        # 2. Jitter by up to ±8 pixels
        ox, oy = torch.randint(-8, 9, (2,), device=device)
        jittered = torch.roll(padded, shifts=(int(ox.item()), int(oy.item())), dims=(2, 3))
        # 3. Random scale: choose factor between 0.9 and 1.1
        scale = 1 + (torch.randint(0, 11, (1,), device=device).item() - 5) / 50.0
        scaled = F.interpolate(jittered, scale_factor=scale, mode='bilinear', align_corners=False)
        # 4. Random rotate: angle between -10 and +10 degrees
        angle = torch.randint(-10, 11, (1,), device=device).item()
        theta = torch.tensor([
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0]
        ], device=device, dtype=torch.float).unsqueeze(0)
        grid = F.affine_grid(theta, scaled.size(), align_corners=False)
        rotated = F.grid_sample(scaled, grid, align_corners=False)
        # 5. Fine jitter by up to ±4 pixels
        ox, oy = torch.randint(-4, 5, (2,), device=device)
        final = torch.roll(rotated, shifts=(int(ox.item()), int(oy.item())), dims=(2, 3))
        # Resize to fixed 224x224
        final = F.interpolate(final, size=(224, 224), mode='bilinear', align_corners=False)
        transformed_list.append(final)
    return torch.cat(transformed_list, dim=0)

# ------------------------------
# Diversity Loss (Based on Gram Matrix Cosine Similarity)
# ------------------------------
def diversity_loss(activations):
    """
    Given activations from the target layer for a batch of images,
    compute the average pairwise cosine similarity between the flattened Gram matrices.
    Lower similarity (i.e. more diversity) is better.
    
    activations: Tensor of shape [B, C, H, W]
    """
    B, C, H, W = activations.shape
    # Reshape activations to [B, C, H*W]
    A = activations.view(B, C, -1)
    # Compute Gram matrices for each image: [B, C, C]
    G = torch.bmm(A, A.transpose(1, 2))
    # Flatten each Gram matrix: [B, C*C]
    G_flat = G.view(B, -1)
    
    loss_div = 0.0
    count = 0
    for i in range(B):
        for j in range(i+1, B):
            sim = torch.dot(G_flat[i], G_flat[j]) / (torch.norm(G_flat[i]) * torch.norm(G_flat[j]) + 1e-8)
            loss_div += sim
            count += 1
    if count > 0:
        loss_div = loss_div / count
    return loss_div

# ------------------------------
# Advanced Visualization Function (with Diversity and Color Correlation)
# ------------------------------
def visualize_inception_channel_diverse(model, target_channel, num_steps=1024, batch_size=4,
                                         lr=0.05, tv_weight=1e-2, l2_weight=5e-2,
                                         diversity_weight=1e-2, device='cpu'):
    """
    Optimize a batch of images to maximize the activation of a given channel in
    Inception's inception4a layer. In addition to the main objective, we add TV and L2
    regularization and a diversity term. We also apply Lucent's color correlation matrix
    in each iteration. The output images will be 224×224.
    """
    model.eval()
    model.to(device)
    
    # Initialize a batch of random images (224x224)
    input_img = torch.rand((batch_size, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    # Add Lucent's color correlation matrix
    color_correlation = torch.tensor([
        [ 0.26,  0.09,  0.02],
        [ 0.27,  0.00, -0.05],
        [ 0.27, -0.09,  0.03]
    ], device=device, dtype=torch.float)
    
    # Dictionary to store activations from inception4a
    activations = {}
    def hook_fn(module, inp, out):
        activations['inception4a'] = out
    hook_handle = model.inception4a.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply transforms to the batch
        transformed_img = apply_transforms_batch(input_img, device)
        _ = model(transformed_img)
        
        # Get activations and apply ReLU
        act = torch.relu(activations['inception4a'])
        main_loss = - torch.mean(act[:, target_channel, :, :])
        tv_loss = total_variation(input_img)
        l2_loss = torch.norm(input_img)
        div_loss = diversity_loss(act)
        
        total_loss = main_loss + tv_weight * tv_loss + l2_weight * l2_loss + diversity_weight * div_loss
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            # Apply Lucent's color correlation transform per iteration:
            input_img = torch.matmul(input_img.permute(0, 2, 3, 1), color_correlation)
            input_img = input_img.permute(0, 3, 1, 2)
            input_img.clamp_(0, 1)
            
            # Normalize each image individually to ensure full [0,1] range
            for i in range(batch_size):
                img = input_img[i]
                mi = img.min()
                ma = img.max()
                input_img[i] = (img - mi) / (ma - mi + 1e-8)
        
        if step % 50 == 0:
            print(f"Step {step:04d} - Total Loss: {total_loss.item():.4f}, "
                  f"Main: {main_loss.item():.4f}, TV: {tv_loss.item():.4f}, "
                  f"L2: {l2_loss.item():.4f}, Div: {div_loss.item():.4f}")
    
    hook_handle.remove()
    return input_img.detach().cpu()  # shape: [batch_size, 3, 224, 224]

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == '__main__':
    # Load the pretrained GoogLeNet (Inception V1)
    model = models.googlenet(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Choose a target channel (ensure this index is valid for inception4a)
    target_channel = 476
    
    # Optimize a batch of images with the diverse visualization method
    optimized_batch = visualize_inception_channel_diverse(model, target_channel, 
                                                           num_steps=1024, batch_size=4,
                                                           lr=0.05, tv_weight=1e-2,
                                                           l2_weight=5e-2, diversity_weight=1e-2,
                                                           device=device)
    
    # Plot the resulting images side-by-side
    batch_size = optimized_batch.shape[0]
    fig, axs = plt.subplots(1, batch_size, figsize=(6 * batch_size, 6))
    for i in range(batch_size):
        img = optimized_batch[i].permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.suptitle(f"Inception4a Channel {target_channel} - Diverse, Color-Correlated Optimizations", fontsize=18)
    plt.show()


# %%

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

# ------------------------------
# Total Variation Regularization
# ------------------------------
def total_variation(x):
    # Encourages spatial smoothness
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ------------------------------
# Apply Lucent-Inspired Transforms to a Batch
# ------------------------------
def apply_transforms_batch(x, device):
    """
    Applies a sequence of transforms (padding, jitter, random scaling,
    rotation, fine jitter) to each image in the batch. Returns images resized to 224×224.
    x: Tensor of shape [B, 3, H, W]
    """
    B = x.shape[0]
    transformed_list = []
    for i in range(B):
        xi = x[i:i+1]  # [1, 3, H, W]
        # 1. Pad with 12px (constant value 0.5)
        padded = F.pad(xi, (12, 12, 12, 12), mode='constant', value=0.5)
        # 2. Jitter by up to ±8 pixels
        ox, oy = torch.randint(-8, 9, (2,), device=device)
        jittered = torch.roll(padded, shifts=(int(ox.item()), int(oy.item())), dims=(2, 3))
        # 3. Random scale: factor between 0.9 and 1.1
        scale = 1 + (torch.randint(0, 11, (1,), device=device).item() - 5) / 50.0
        scaled = F.interpolate(jittered, scale_factor=scale, mode='bilinear', align_corners=False)
        # 4. Random rotate: angle between -10 and +10 degrees
        angle = torch.randint(-10, 11, (1,), device=device).item()
        theta = torch.tensor([
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0]
        ], device=device, dtype=torch.float).unsqueeze(0)
        grid = F.affine_grid(theta, scaled.size(), align_corners=False)
        rotated = F.grid_sample(scaled, grid, align_corners=False)
        # 5. Fine jitter by up to ±4 pixels
        ox, oy = torch.randint(-4, 5, (2,), device=device)
        final = torch.roll(rotated, shifts=(int(ox.item()), int(oy.item())), dims=(2, 3))
        # Resize to fixed 224×224
        final = F.interpolate(final, size=(224, 224), mode='bilinear', align_corners=False)
        transformed_list.append(final)
    return torch.cat(transformed_list, dim=0)

# ------------------------------
# Diversity Loss (Based on Gram Matrix Cosine Similarity)
# ------------------------------
def diversity_loss(activations):
    """
    Computes the average pairwise cosine similarity between flattened Gram matrices
    of the activations. Lower similarity means more diversity.
    activations: Tensor of shape [B, C, H, W]
    """
    B, C, H, W = activations.shape
    A = activations.view(B, C, -1)  # [B, C, H*W]
    G = torch.bmm(A, A.transpose(1, 2))  # [B, C, C]
    G_flat = G.view(B, -1)  # [B, C*C]
    
    loss_div = 0.0
    count = 0
    for i in range(B):
        for j in range(i+1, B):
            sim = torch.dot(G_flat[i], G_flat[j]) / (torch.norm(G_flat[i]) * torch.norm(G_flat[j]) + 1e-8)
            loss_div += sim
            count += 1
    if count > 0:
        loss_div = loss_div / count
    return loss_div

# ------------------------------
# Advanced Visualization Function (with Diversity and Post-Processing Color Correlation)
# ------------------------------
def visualize_inception_channel_diverse(model, target_channel, num_steps=1024, batch_size=4,
                                         lr=0.05, tv_weight=1e-2, l2_weight=5e-2,
                                         diversity_weight=1e-2, device='cpu'):
    """
    Optimizes a batch of images to maximize the activation of the target channel in
    the inception4a layer of GoogLeNet (Inception V1). Uses total variation, L2, and diversity
    losses to regularize the optimization. After optimization, applies Lucent's color correlation
    matrix (derived from natural image statistics) as a post-processing step.
    
    Output images are 224×224.
    """
    model.eval()
    model.to(device)
    
    # Initialize a batch of random images (224×224)
    input_img = torch.rand((batch_size, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    # Hook to capture activations from the inception4a layer
    activations = {}
    def hook_fn(module, inp, out):
        activations['inception4a'] = out
    hook_handle = model.inception4a.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply the transformation pipeline
        transformed_img = apply_transforms_batch(input_img, device)
        _ = model(transformed_img)  # Hook captures activations
        
        act = torch.relu(activations['inception4a'])
        main_loss = -torch.mean(act[:, target_channel, :, :])
        tv_loss = total_variation(input_img)
        l2_loss = torch.norm(input_img)
        div_loss = diversity_loss(act)
        
        total_loss = main_loss + tv_weight * tv_loss + l2_weight * l2_loss + diversity_weight * div_loss
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if step % 50 == 0:
            print(f"Step {step:04d} - Total Loss: {total_loss.item():.4f}, "
                  f"Main: {main_loss.item():.4f}, TV: {tv_loss.item():.4f}, "
                  f"L2: {l2_loss.item():.4f}, Div: {div_loss.item():.4f}")
    
    hook_handle.remove()
    
    # ------------------------------
    # Post-Processing: Apply Lucent's Color Correlation Matrix
    # ------------------------------
    # This matrix is the SVD square-root of the color correlation matrix derived from natural images.
    color_correlation = torch.tensor([
        [ 0.26,  0.09,  0.02],
        [ 0.27,  0.00, -0.05],
        [ 0.27, -0.09,  0.03]
    ], device=device, dtype=torch.float)
    
    with torch.no_grad():
        # Permute so that colors are in the last dimension, apply the matrix, then permute back.
        output_img = torch.matmul(input_img.permute(0, 2, 3, 1), color_correlation)
        output_img = output_img.permute(0, 3, 1, 2)
        output_img.clamp_(0, 1)
        # Normalize each image individually to [0,1]
        for i in range(batch_size):
            mi = output_img[i].min()
            ma = output_img[i].max()
            output_img[i] = (output_img[i] - mi) / (ma - mi + 1e-8)
    
    return output_img.detach().cpu()  # [B, 3, 224, 224]

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == '__main__':
    # Load pretrained GoogLeNet (Inception V1)
    model = models.googlenet(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Choose the target channel index (ensure this index is valid for inception4a)
    target_channel = 476
    
    # Optimize a batch of images using the diverse, color-constrained method
    optimized_batch = visualize_inception_channel_diverse(
        model, target_channel, num_steps=1024, batch_size=4,
        lr=0.05, tv_weight=1e-2, l2_weight=5e-2, diversity_weight=1e-2, device=device
    )
    
    # Plot the resulting images side-by-side
    batch_size = optimized_batch.shape[0]
    fig, axs = plt.subplots(1, batch_size, figsize=(6 * batch_size, 6))
    for i in range(batch_size):
        img = optimized_batch[i].permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.suptitle(f"Inception4a Channel {target_channel}\nDiverse, Color-Correlated Optimizations", fontsize=18)
    plt.show()


# %%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np

# ------------------------------
# Load pretrained GoogLeNet (Inception V1)
# ------------------------------
model = models.googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# ------------------------------
# Total Variation Regularization
# ------------------------------
def total_variation(x):
    """
    Total variation regularization to encourage spatial smoothness.
    """
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ------------------------------
# Transformation Pipeline
# ------------------------------
def apply_transforms(x, device):
    """
    Applies a series of transforms to the image:
      1. Pad with 12px (constant value 0.5)
      2. Coarse jitter: shift randomly by up to ±8 pixels
      3. Random scale: choose a factor in [0.9, 1.1]
      4. Random rotation: rotate by an angle between -10 and +10 degrees
      5. Fine jitter: shift randomly by up to ±4 pixels
    Finally, resize to 224×224.
    """
    # 1. Pad: (left, right, top, bottom)
    x = F.pad(x, (12, 12, 12, 12), mode='constant', value=0.5)
    
    # 2. Coarse jitter:
    ox, oy = torch.randint(-8, 9, (2,), device=device)
    x = torch.roll(x, shifts=(int(ox.item()), int(oy.item())), dims=(2,3))
    
    # 3. Random scale:
    scale = 1 + (torch.randint(0, 11, (1,), device=device).item() - 5) / 50.0
    x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
    
    # 4. Random rotation:
    angle = torch.randint(-10, 11, (1,), device=device).item()
    theta = torch.tensor([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
        [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0]
    ], device=device, dtype=torch.float).unsqueeze(0)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    
    # 5. Fine jitter:
    ox, oy = torch.randint(-4, 5, (2,), device=device)
    x = torch.roll(x, shifts=(int(ox.item()), int(oy.item())), dims=(2,3))
    
    # Resize to 224x224 (if not already)
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    return x

# ------------------------------
# Feature Visualization Function for Inception V1 with Transforms
# ------------------------------
def visualize_inception_channel(model, target_module_name, target_channel, num_steps=512, lr=0.1, reg_weight=1e-4, device='cpu'):
    """
    Optimizes an input image to maximize the activation of a specific channel in a given module.
    This version integrates a transform pipeline to enhance robustness.
    
    Parameters:
      - model: the pretrained model (GoogLeNet/Inception V1)
      - target_module_name: string name of the module (e.g., 'inception4a')
      - target_channel: integer index of the channel to maximize
      - num_steps: number of gradient ascent steps
      - lr: learning rate for the optimizer
      - reg_weight: weight for total variation regularization
      - device: 'cpu' or 'cuda'
      
    Returns:
      - result: a NumPy array (H x W x C) of the optimized image.
    """
    model.eval()
    model.to(device)
    
    # Initialize a random image of size (1, 3, 224, 224)
    input_img = torch.rand((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    # Dictionary to store activations from the target module
    activations = {}
    def hook_fn(module, input, output):
        activations[target_module_name] = output
    target_module = getattr(model, target_module_name)
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply transformation pipeline before the forward pass.
        transformed_img = apply_transforms(input_img, device)
        
        # Forward pass: the hook captures activations from the target module.
        _ = model(transformed_img)
        act = activations[target_module_name]
        if isinstance(act, tuple):
            act = act[0]
        
        # act is [1, C, H, W]; select the target channel.
        channel_activation = act[0, target_channel, :, :]
        
        # Loss: negative mean activation (to maximize activation) + TV regularization.
        loss = -channel_activation.mean() + reg_weight * total_variation(input_img)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if step % 10 == 0:
            print(f"Step {step:03d} - Loss: {loss.item():.4f}")
    
    hook_handle.remove()
    
    # (Optional) Post-process: you could add additional normalization here if needed.
    result = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return result

# ------------------------------
# Run Visualization on Inception V1
# ------------------------------
target_module_name = 'inception4a'
target_channel = 476  # Ensure this channel index exists in inception4a
optimized_img = visualize_inception_channel(model,
                                              target_module_name=target_module_name,
                                              target_channel=target_channel,
                                              num_steps=512,
                                              lr=0.1,
                                              reg_weight=1e-4,
                                              device=device)

# Plot the resulting image.
plt.figure(figsize=(8, 8))
plt.imshow(optimized_img)
plt.title(f"Maximized Activation for {target_module_name} Channel {target_channel}")
plt.axis('off')
plt.show()





# %%


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim

# ------------------------------
# Load Pretrained GoogLeNet (Inception V1)
# ------------------------------
model = models.googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# ------------------------------
# Regularization Functions
# ------------------------------
def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

def jitter(img, ox, oy):
    """
    Randomly shift the image by (ox, oy) pixels.
    """
    return torch.roll(img, shifts=(ox, oy), dims=(2, 3))

# ------------------------------
# Helper: Apply Color Transformation
# ------------------------------
def apply_color_correlation(x, color_matrix):
    """
    Apply a 3x3 color correlation matrix to an image tensor.
    x: tensor of shape (batch, 3, H, W)
    color_matrix: tensor of shape (3, 3)
    Returns: transformed image (batch, 3, H, W)
    """
    # Here we use einsum to apply the matrix to the channel dimension.
    # For each pixel: new_pixel[i] = sum_j color_matrix[i,j] * old_pixel[j]
    return torch.einsum('ij,bjhw->bihw', color_matrix, x)

# ------------------------------
# Feature Visualization Function with Color Matrix Integration
# ------------------------------
def visualize_inception_channel(model, target_module_name, target_channel, 
                                num_steps=300, lr=0.05, tv_weight=1e-3, l2_weight=1e-3, 
                                jitter_amount=8, device='cpu', color_matrix=None):
    """
    Optimize an input image to maximize the activation of a specific channel
    in a target module of the model. After each update step, if a color_matrix is
    provided, the image is transformed by it.
    """
    model.eval()
    model.to(device)

    # GoogLeNet expects an image of shape (1, 3, 224, 224)
    input_img = torch.rand((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    activations = {}
    
    def hook_fn(module, input, output):
        activations[target_module_name] = output

    # Register a forward hook on the target module
    target_module = getattr(model, target_module_name)
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply random jitter each step
        ox, oy = np.random.randint(-jitter_amount, jitter_amount+1, 2)
        jittered_img = jitter(input_img, ox, oy)
        
        _ = model(jittered_img)
        act = activations[target_module_name]
        if isinstance(act, tuple):
            act = act[0]
        channel_activation = act[0, target_channel, :, :]
        
        # Loss: negative activation (to maximize activation) plus regularizers
        loss = -channel_activation.mean()
        loss = loss + tv_weight * total_variation(input_img)
        loss = loss + l2_weight * torch.norm(input_img)
        
        loss.backward()
        optimizer.step()
        
        # After the optimizer step, apply the color decorrelation if provided.
        with torch.no_grad():
            if color_matrix is not None:
                # Apply the matrix to each pixel's RGB values.
                input_img.data = apply_color_correlation(input_img.data, color_matrix)
            input_img.data.clamp_(0, 1)
        
        if step % 20 == 0:
            print(f"Step {step:03d} - Loss: {loss.item():.4f}")
    
    hook_handle.remove()
    
    result = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return result

# ------------------------------
# Define Hyperparameter and Color Matrix Sweeps
# ------------------------------

# Hyperparameters for regularization
tv_weights = [1e-2]  # Fixed TV weight for this sweep
l2_weights = [1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]  # Sweep over L2 weights
# We'll keep jitter constant here (or you can add an extra loop if desired)
jitter_amount = 4

# Define color matrices to sweep over.
# Each tuple is (name, matrix). If the matrix is None, no color transform is applied.
color_matrices = [
    ("No Color Transform", None),
    ("Lucent", torch.tensor([
            [ 0.26,  0.09,  0.02],
            [ 0.27,  0.00, -0.05],
            [ 0.27, -0.09,  0.03]
        ], device=device)),
    ("Identity", torch.eye(3, device=device))
]

# The target module and channel we want to visualize
target_module_name = 'inception4a'
target_channel = 475  # Ensure this index exists; adjust if needed

# For clarity, we will create one subplot grid per color matrix.
num_color = len(color_matrices)
num_l2 = len(l2_weights)
fig, axs = plt.subplots(num_color, num_l2, figsize=(15, 5 * num_color))
fig.suptitle(
    f"Hyperparameter Sweep for {target_module_name} Channel {target_channel}\n"
    f"TV weight fixed at: {tv_weights[0]:.0e}\nL2 weights: {l2_weights}\nJitter amount: {jitter_amount}",
    fontsize=14
)

print("TV weight:", tv_weights[0])
print("L2 weights:", l2_weights)
print("Color matrices:", [name for name, _ in color_matrices])

# Sweep over the color matrices and L2 weight values.
for i, (color_name, color_matrix) in enumerate(color_matrices):
    for j, l2_w in enumerate(l2_weights):
        print(f"\nOptimizing with Color: {color_name}, TV weight={tv_weights[0]:.0e}, L2 weight={l2_w:.0e}")
        optimized_img = visualize_inception_channel(
            model,
            target_module_name=target_module_name,
            target_channel=target_channel,
            num_steps=500,
            lr=0.05,
            tv_weight=tv_weights[0],
            l2_weight=l2_w,
            jitter_amount=jitter_amount,
            device=device,
            color_matrix=color_matrix
        )
        axs[i, j].imshow(optimized_img)
        axs[i, j].set_title(f'{color_name}\nL2 = {l2_w:.0e}')
        axs[i, j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



# %%

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim

# ------------------------------
# Load Pretrained GoogLeNet (Inception V1)
# ------------------------------
model = models.googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# ------------------------------
# Regularization Functions
# ------------------------------
def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

def jitter(img, ox, oy):
    """
    Randomly shift the image by (ox, oy) pixels.
    """
    return torch.roll(img, shifts=(ox, oy), dims=(2, 3))

# ------------------------------
# Feature Visualization Function with Improvements
# ------------------------------
def visualize_inception_channel(model, target_module_name, target_channel, num_steps=300, lr=0.05, 
                                tv_weight=1e-3, l2_weight=1e-3, jitter_amount=8, device='cpu'):
    """
    Optimize an input image to maximize the activation of a specific channel
    in a target module of the model, using additional regularizers and jitter.
    """
    model.eval()
    model.to(device)

    # GoogLeNet expects (1, 3, 224, 224)
    input_img = torch.rand((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    activations = {}
    
    def hook_fn(module, input, output):
        activations[target_module_name] = output

    target_module = getattr(model, target_module_name)
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply random jitter each step
        ox, oy = np.random.randint(-jitter_amount, jitter_amount+1, 2)
        jittered_img = jitter(input_img, ox, oy)
        
        _ = model(jittered_img)
        act = activations[target_module_name]
        if isinstance(act, tuple):
            act = act[0]
        channel_activation = act[0, target_channel, :, :]
        
        # Loss: negative activation (to maximize) plus regularizers
        loss = -channel_activation.mean()
        loss = loss + tv_weight * total_variation(input_img)
        loss = loss + l2_weight * torch.norm(input_img)
        
        loss.backward()
        optimizer.step()
        
        # Clamp image values to [0, 1]
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if step % 20 == 0:
            print(f"Step {step:03d} - Loss: {loss.item():.4f}")
    
    hook_handle.remove()
    
    result = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return result

# ------------------------------
# Run the Improved Visualization
# ------------------------------
target_module_name = 'inception4a'
target_channel = 475  # Ensure this index exists; adjust as needed

# Define hyperparameter ranges to sweep
tv_weights = [ 1e-2]  # More TV weight values
l2_weights = [ 1e-3, 1e-2, 2e-2, 3e-2, 4-2,5e-2 ]  # More L2 weight values
jitter_amounts = [2, 4, 8]  # Add jitter amount sweep


# Create subplot grid
fig, axs = plt.subplots(len(tv_weights), len(l2_weights), figsize=(15, 15))
fig.suptitle(f"Hyperparameter Sweep for {target_module_name} Channel {target_channel}\nTV weights: {tv_weights}\nL2 weights: {l2_weights}\nJitter amounts: {jitter_amounts}")
# Print hyperparameter values
print("TV weights:", tv_weights)
print("L2 weights:", l2_weights) 
print("Jitter amounts:", jitter_amounts)

# Sweep over hyperparameters
for i, tv_w in enumerate(tv_weights):
    for j, l2_w in enumerate(l2_weights):
        print(f"\nOptimizing with TV weight={tv_w:.0e}, L2 weight={l2_w:.0e}")
        optimized_img = visualize_inception_channel(model,
                                                  target_module_name=target_module_name,
                                                  target_channel=target_channel,
                                                  num_steps=500,
                                                  lr=0.05,
                                                  tv_weight=tv_w,
                                                  l2_weight=l2_w,
                                                  jitter_amount=4,
                                                  device=device)
        
        axs[i,j].imshow(optimized_img)
        axs[i,j].set_title(f'TV={tv_w:.0e}, L2={l2_w:.0e}')
        axs[i,j].axis('off')

plt.tight_layout()
plt.show()



