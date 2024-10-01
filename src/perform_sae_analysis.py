

def measure_logit_difference(model, sae, layer_number, num_samples=100):
    # Generate observations
    observations = []
    env = heist.create_venv(num=1, num_levels=0, start_level=random.randint(1, 100000))
    obs = env.reset()
    for _ in range(num_samples):
        observations.append(obs[0].copy())
        action = env.action_space.sample()
        obs, reward, done, info = env.step(np.array([action]))
        if done[0]:
            obs = env.reset()

    # Convert observations to tensor
    obs_tensor = t.tensor(np.array(observations), dtype=t.float32)
    print(obs_tensor.shape)
    obs_tensor = einops.rearrange(obs_tensor, " b c h w -> b h  w c").to(device)
    print(obs_tensor.shape)

    # Get logits without SAE
    with t.no_grad():
        outputs = model(obs_tensor)
        logits_without_sae = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        logits_without_sae = logits_without_sae.cpu().numpy()

    # Register the hook to replace the layer with SAE outputs
    handle = replace_layer_with_sae(model, sae, layer_number)

    # Get logits with SAE
    with t.no_grad():
        outputs = model(obs_tensor)
        logits_with_sae = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
        logits_with_sae = logits_with_sae.cpu().numpy()

    # Remove the hook
    handle.remove()

    # Compute differences
    logit_differences = logits_with_sae - logits_without_sae

    # Return the differences
    return logits_without_sae, logits_with_sae, logit_differences

# %%


# Function to collect strongly activating features and corresponding observations
def collect_strong_activations(sae, model, layer_number, threshold=1.0, num_episodes=10, max_steps_per_episode=1000, feature_indices=None, device=device):
    # Ensure model and sae are in eval mode
    model.eval()
    sae.eval()

    # Initialize ModelActivations for the layer
    layer_name = ordered_layer_names[layer_number]
    layer_paths = [layer_name]
    model_activations = ModelActivations(model, layer_paths=layer_paths)

    # Initialize data structure to store observations
    from collections import defaultdict
    feature_observations = defaultdict(list)

    # If feature_indices is None, monitor all features
    if feature_indices is None:
        feature_indices = range(sae.cfg.d_hidden)

    # Create the environment
    env = heist.create_venv(
        num=1,
        num_levels=0,
        start_level=random.randint(1, 100000),
        distribution_mode="easy"
    )

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            obs_tensor = t.tensor(obs, dtype=t.float32)  # Add batch dimension

            # Run the model and get activations
            with t.no_grad():
                outputs, activations = model_activations.run_with_cache(obs_tensor)
                # Get the layer activation
                layer_activation = activations[layer_name.replace('.', '_')]
                # Flatten layer activation
                h = layer_activation.view(1, -1).to(device)
                # Pass through SAE to get feature activations
                _, _, _, acts = sae(h)
                acts = acts.cpu().numpy()[0]  # Get numpy array, remove batch dimension

            # Check for strongly activated features
            for idx in feature_indices:
                activation_value = acts[idx]
                if activation_value >= threshold:
                    # Record the observation and activation value
                    feature_observations[idx].append((obs.copy(), activation_value))

            # Get action from the model outputs
            logits = outputs[0].logits if isinstance(outputs, tuple) else outputs.logits
            probabilities = t.softmax(logits, dim=-1)
            action = t.multinomial(probabilities, num_samples=1).item()

            # Step the environment
            obs, reward, done, info = env.step(np.array([action]))
            steps += 1

    env.close()
    return feature_observations

# Main script to load SAE, collect features, and visualize them
# %%


# Load the main model
model = helpers.load_interpretable_model()
model.to(device)
model.eval()

# Define the layer paths
layer_name = ordered_layer_names[layer_number]
layer_paths = [layer_name]

# Initialize ModelActivations
model_activations = ModelActivations(model, layer_paths=layer_paths)

# Get a sample observation
env = heist.create_venv(num=1, num_levels=0, start_level=random.randint(1, 100000))
obs = env.reset()
obs = t.tensor(obs, dtype=t.float32)  # Add batch dimension

# Run model_activations to get activation shape
with t.no_grad():
    outputs, activations = model_activations.run_with_cache(obs)
layer_activation = activations[layer_name.replace('.', '_')]
activation_shape = layer_activation.shape  # [batch_size, channels, height, width]
_, channels, height, width = activation_shape
d_in = channels * height * width





# %%
# Collect and visualize strongly activating features
feature_activations = collect_strong_activations(
    sae,
    model=model,
    layer_number=layer_number,
    threshold=0.5,          # Adjust threshold as needed
    num_episodes=50,        # Number of episodes to run
    max_steps_per_episode=100,
    feature_indices=None,    # Monitor all features
    device=device
)
# %%
# Shuffle the features
shuffled_features = list(feature_activations.items())
random.shuffle(shuffled_features)
feature_activations = dict(shuffled_features)

print(f"Features have been shuffled. New order: {list(feature_activations.keys())}")

save_obs = None
# For each feature, visualize some of the observations
for idx, obs_list in feature_activations.items():
    print(f"Feature {idx} activated strongly {len(obs_list)} times.")
    # Visualize first 4 observations in a grid
    num_visualize = min(4, len(obs_list))
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f'Feature {idx} Activations')
    for i in range(num_visualize):
        obs, activation_value = obs_list[i]
        save_obs = obs
        obs = einops.rearrange(obs[0], "c h w -> w h c")
        row, col = divmod(i, 2)
        axs[row, col].imshow(obs)
        axs[row, col].set_title(f'Activation: {activation_value:.2f}')
        axs[row, col].axis('off')
    
    # If less than 4 observations, remove empty subplots
    for i in range(num_visualize, 4):
        row, col = divmod(i, 2)
        fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.show()

# %%
# Plot activation frequencies as bars
plt.figure(figsize=(12, 6))
activation_counts = [len(obs_list) for obs_list in feature_activations.values()]
feature_indices = list(feature_activations.keys())

plt.bar(feature_indices, activation_counts)
plt.xlabel('Feature Index')
plt.ylabel('Activation Frequency')
plt.title('Feature Activation Frequencies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calculate and print some statistics
total_activations = sum(activation_counts)
mean_activations = total_activations / len(feature_indices)
max_activations = max(activation_counts)
min_activations = min(activation_counts)

print(f"Total activations: {total_activations}")
print(f"Mean activations per feature: {mean_activations:.2f}")
print(f"Max activations for a single feature: {max_activations}")
print(f"Min activations for a single feature: {min_activations}")

# %%
print(feature_activations)
