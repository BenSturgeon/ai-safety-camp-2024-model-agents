# Common Gotchas in Debugging Procgen Environments

# 1. Creating the environment
# Ensure that create_venv is called with num=1
venv = create_venv(num=1, start_level=0, num_levels=1)  # Adjust other parameters as needed

# 2. Action generation
# After calculating the action, convert it to a numpy array
action = generate_action(model, observation)
action = np.array([action])  # Convert to numpy array with shape (1,)

# 3. Observation handling
# Do not unsqueeze the observation after generation
observation = venv.reset()  # This already returns the correct shape

# 4. Model input preparation
# Ensure the observation is in the correct format for the model
observation = observation.transpose(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
observation = torch.tensor(observation, dtype=torch.float32)


# 9. Device management for PyTorch
# Ensure model and tensors are on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
observation = observation.to(device)

# 10. Batch dimension
# Remember that even for a single environment, inputs often need a batch dimension
if observation.dim() == 3:
    observation = observation.unsqueeze(0)  # Add batch dimension if not present


# image plotting
Make sure to not change images to type int when plotting
plt.imshow(obs.astype(np.uint8)) # not this!
plt.imshow(obs) # this!

# wrapping obs with numpy when getting dsi
sometimes you might get this error when trying to run the environment

RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
if this happens the fix may be to wrap the observation with a numpy array
obs_tensor = t.tensor(np.array(obs), dtype=t.float32)
print(obs_tensor.shape)
obs_tensor = einops.rearrange(obs_tensor, " b c h w -> b h  w c").to(device)

# For discussion of training objectives in SAEs look here
https://www.alignmentforum.org/posts/CkFBMG6A9ytkiXBDM/sparse-autoencoders-future-work