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
