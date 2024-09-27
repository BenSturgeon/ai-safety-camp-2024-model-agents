from src.utils import helpers
from src.utils import heist
import random
import numpy as np
import imageio
import torch

ENTITY_COLORS = {"blue": 0, "green": 1, "red": 2}

ENTITY_TYPES = {"key": 2, "lock": 1, "gem": 9, "player": 0}

# Constants
KEY_COLORS = {
    0: "blue",
    1: "green",
    2: "red",
}
MOUSE = 0
KEY = 2
LOCKED_DOOR = 1
WORLD_DIM = 25
EMPTY = 100
BLOCKED = 51
GEM = 9

ordered_layer_names = {
    1: "conv1a",
    2: "pool1",
    3: "conv2a",
    4: "conv2b",
    5: "pool2",
    6: "conv3a",
    7: "pool3",
    8: "conv4a",
    9: "pool4",
    10: "fc1",
    11: "fc2",
    12: "fc3",
    13: "value_fc",
    14: "dropout_conv",
    15: "dropout_fc",
}


def create_corridor_environment(entity_one, entity_two, corridor_length=9):
    # Keep recreating the environment until we find one with a red key
    while True:
        venv = heist.create_venv(
            num=1, start_level=random.randint(1000, 10000), num_levels=0
        )
        state = heist.state_from_venv(venv, 0)
        if state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["red"]):
            break

    # Get the full grid and its dimensions
    full_grid = state.full_grid(with_mouse=False)
    height, width = full_grid.shape

    # Calculate the middle of the grid
    middle_y = height // 2

    # Define the corridor
    corridor_start_x = 0
    corridor_end_x = 8
    corridor_y = middle_y

    # Create a new grid with walls everywhere except for the corridor
    new_grid = np.full_like(full_grid, BLOCKED)
    new_grid[corridor_y] = EMPTY

    # Set the new grid
    state.set_grid(new_grid)

    # Place the player in the middle of the corridor
    # Store original positions of entities before removing them
    original_positions = {}
    for ent in state.state_vals["ents"]:
        if ent["image_type"].val in [
            ENTITY_TYPES["key"],
            ENTITY_TYPES["lock"],
            ENTITY_TYPES["gem"],
        ]:
            key = (ent["image_type"].val, ent["image_theme"].val)
            original_positions[key] = (ent["x"].val, ent["y"].val)

    # Remove all entities (which moves them off-screen)
    state.remove_all_entities()

    player_x = corridor_y
    player_y = corridor_length // 2
    state.set_mouse_pos(player_x, player_y)

    # Get entity types and colors
    entity_one_type, entity_one_color = entity_one
    entity_two_type, entity_two_color = entity_two

    # Function to restore entity to the corridor
    def restore_entity_to_corridor(entity_type, entity_color, x_pos):
        key = (ENTITY_TYPES[entity_type], ENTITY_COLORS[entity_color])
        if key in original_positions:
            state.set_entity_position(
                ENTITY_TYPES[entity_type],
                ENTITY_COLORS[entity_color],
                corridor_y,
                x_pos,
            )

    # Restore entity one to the start of the corridor
    restore_entity_to_corridor(entity_one_type, entity_one_color, corridor_start_x)

    # Restore entity two to the end of the corridor
    restore_entity_to_corridor(entity_two_type, entity_two_color, corridor_end_x)

    # Update the environment with the new state
    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        venv.reset()

    return venv


def create_corridor_with_corner_environment(entity_one, entity_two, corridor_length=9):
    # Keep recreating the environment until we find one with a red key
    while True:
        venv = heist.create_venv(
            num=1, start_level=random.randint(1000, 10000), num_levels=0
        )
        state = heist.state_from_venv(venv, 0)
        if state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["red"]):
            break

    # Get the full grid and its dimensions
    full_grid = state.full_grid(with_mouse=False)
    height, width = full_grid.shape

    # Calculate the middle of the grid
    middle_y = height // 2

    # Define the corridor
    corridor_start_x = 0
    corridor_end_x = 8
    corridor_y = middle_y

    # Create a new grid with walls everywhere except for the corridor
    new_grid = np.full_like(full_grid, BLOCKED)
    new_grid[corridor_y] = EMPTY
    new_grid[0, 8] = EMPTY
    new_grid[1, 8] = EMPTY
    new_grid[2, 8] = EMPTY
    new_grid[3, 8] = EMPTY
    new_grid[4, 8] = EMPTY

    # Set the new grid
    state.set_grid(new_grid)

    # Place the player in the middle of the corridor
    # Store original positions of entities before removing them
    original_positions = {}
    for ent in state.state_vals["ents"]:
        if ent["image_type"].val in [
            ENTITY_TYPES["key"],
            ENTITY_TYPES["lock"],
            ENTITY_TYPES["gem"],
        ]:
            key = (ent["image_type"].val, ent["image_theme"].val)
            original_positions[key] = (ent["x"].val, ent["y"].val)

    # Remove all entities (which moves them off-screen)
    state.remove_all_entities()

    player_x = corridor_y
    player_y = corridor_length // 2
    state.set_mouse_pos(player_x, player_y)

    # Get entity types and colors
    entity_one_type, entity_one_color = entity_one
    entity_two_type, entity_two_color = entity_two

    # Function to restore entity to the corridor
    def restore_entity_to_corridor(entity_type, entity_color, x_pos, y_pos):
        key = (ENTITY_TYPES[entity_type], ENTITY_COLORS[entity_color])
        if key in original_positions:
            state.set_entity_position(
                ENTITY_TYPES[entity_type], ENTITY_COLORS[entity_color], y_pos, x_pos
            )

    # Restore entity one to the start of the corridor
    restore_entity_to_corridor(
        entity_one_type, entity_one_color, corridor_start_x, corridor_y
    )

    # Restore entity two to the end of the corridor
    restore_entity_to_corridor(entity_two_type, entity_two_color, corridor_end_x, 0)

    # Update the environment with the new state
    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        venv.reset()

    return venv


def compare_first_key_collected(
    model_path,
    layer_number,
    channel,
    modification_value,
    num_episodes=100,
    corridor_length=9,
    max_steps=100,
):
    model = helpers.load_interpretable_model(model_path)
    blue_key_collected_first = 0
    green_key_collected_first = 0
    no_key_collected = 0

    steering_layer = helpers.rename_path(ordered_layer_names[layer_number])

    for episode in range(num_episodes):
        venv = create_corridor_environment(
            entity_one=("key", "blue"),
            entity_two=("key", "green"),
            corridor_length=corridor_length,
        )

        obs = venv.reset()
        done = False
        steps = 0

        # Generate steering vector
        state = heist.state_from_venv(venv)
        original_blue_pos = state.get_entity_position(
            ENTITY_TYPES["key"], ENTITY_COLORS["blue"]
        )

        # Temporarily remove blue key to generate steering vector
        state.remove_entity(ENTITY_TYPES["key"], ENTITY_COLORS["blue"])
        state_bytes = state.state_bytes
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

        # Restore blue key
        state.restore_entity_position(
            ENTITY_TYPES["key"], ENTITY_COLORS["blue"], original_blue_pos
        )
        state_bytes = state.state_bytes
        venv.env.callmethod("set_state", [state_bytes])

        # Calculate steering vector
        model_activations = helpers.ModelActivations(model)
        _, unmodified_activations = model_activations.run_with_cache(
            helpers.observation_to_rgb(obs), [steering_layer]
        )
        _, modified_activations = model_activations.run_with_cache(
            helpers.observation_to_rgb(modified_obs), [steering_layer]
        )

        steering_vector = (
            unmodified_activations[steering_layer][0][channel]
            - modified_activations[steering_layer][0][channel]
        )

        while not done and steps < max_steps:
            action = heist.generate_steering_action(
                model, obs, steering_vector, steering_layer, channel, modification_value
            )

            obs, reward, done, info = venv.step(action)
            steps += 1

            if reward > 0:  # A key was collected
                state = heist.state_from_venv(venv)
                remaining_keys = [
                    ent
                    for ent in state.state_vals["ents"]
                    if ent["image_type"].val == ENTITY_TYPES["key"]
                ]

                if len(remaining_keys) == 1:
                    if remaining_keys[0]["image_theme"].val == ENTITY_COLORS["blue"]:
                        green_key_collected_first += 1
                    else:
                        blue_key_collected_first += 1
                break

        if steps == max_steps:
            no_key_collected += 1

        venv.close()

    results = {
        "blue_key_collected_first": blue_key_collected_first,
        "green_key_collected_first": green_key_collected_first,
        "no_key_collected": no_key_collected,
        "total_episodes": num_episodes,
    }

    return results


def run_entity_steering_experiment_by_channel(
    model_path,
    layer_number,
    modification_value,
    episode,
    channel,
    entity_name,
    entity_color=None,
    num_levels=1,
    start_level=None,
    episode_timeout=200,
    save_gif=False,
):
    entity_type = ENTITY_TYPES.get(entity_name)
    entity_theme = ENTITY_COLORS.get(entity_color) if entity_color else None

    if start_level == None:
        start_level = random.randint(1, 100000)

    venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
    state = heist.state_from_venv(venv, 0)
    while not state.entity_exists(entity_type, entity_theme):
        start_level = random.randint(1, 100000)
        venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
        state = heist.state_from_venv(venv, 0)
    unchanged_obs = venv.reset()

    # Save the current position of the target entity
    original_position = state.get_entity_position(entity_type, entity_theme)

    # Move the target entity off-screen
    state.remove_entity(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

    state = heist.state_from_venv(venv, 0)

    # Restore the entity to its original position
    state.restore_entity_position(entity_type, entity_theme, original_position)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])

    model = helpers.load_interpretable_model(model_path=model_path)
    steering_layer_unchanged = ordered_layer_names[layer_number]
    steering_layer = helpers.rename_path(steering_layer_unchanged)

    model_activations = helpers.ModelActivations(model)
    model_activations.clear_hooks()
    output1, unmodified_activations = model_activations.run_with_cache(
        helpers.observation_to_rgb(unchanged_obs), [ordered_layer_names[layer_number]]
    )
    model_activations.clear_hooks()
    output2, modified_obs_activations = model_activations.run_with_cache(
        helpers.observation_to_rgb(modified_obs), [ordered_layer_names[layer_number]]
    )

    steering_vector = (
        unmodified_activations[steering_layer][0][channel]
        - modified_obs_activations[steering_layer][0][channel]
    )

    observation = venv.reset()
    done = False
    total_reward = 0
    frames = []
    observations = []
    count = 0
    entity_picked_up = False
    count_pickups = 0
    steps_until_pickup = 0

    # Count initial number of target entities
    initial_state = heist.state_from_venv(venv, 0)
    initial_entity_count = initial_state.count_entities(entity_type, entity_theme)
    current_entity_count = state.count_entities(entity_type, entity_theme)

    while not done:
        if save_gif:
            frames.append(venv.render(mode="rgb_array"))

        observation = np.squeeze(observation)
        observation = np.transpose(observation, (1, 2, 0))
        converted_obs = helpers.observation_to_rgb(observation)
        action = helpers.generate_action_with_steering(
            model,
            converted_obs,
            steering_vector,
            steering_layer,
            modification_value,
            is_procgen_env=True,
        )

        observation, reward, done, info = venv.step(action)
        total_reward += reward
        observations.append(converted_obs)
        steps_until_pickup += 1
        if steps_until_pickup > 120:
            done = True

        state = heist.state_from_venv(venv, 0)
        current_entity_count = state.count_entities(entity_type, entity_theme)

        if current_entity_count < initial_entity_count:
            entity_picked_up = True
            done = True
            count_pickups += 1

        elif entity_type == ENTITY_TYPES["gem"] and reward > 0:
            entity_picked_up = True
            done = True
            count_pickups += 1

    if save_gif:
        file_name = f"gifs/episode_steering_{episode}_{entity_color}_{entity_name}_layer_{layer_number}_channel_{channel}_value_{modification_value}.gif"
        imageio.mimsave(file_name, frames, fps=30)
        print(f"Saved gif ad {file_name}")

    state = heist.state_from_venv(venv, 0)
    state_vals = state.state_vals

    lock_positions_after = heist.get_lock_statuses(state_vals)

    return total_reward, steps_until_pickup, count_pickups


def run_entity_steering_experiment(
    model_path,
    layer_number,
    modification_value,
    episode,
    entity_name,
    entity_color=None,
    num_levels=1,
    start_level=random.randint(1, 100000),
    episode_timeout=200,
    save_gif=False,
):
    entity_type = ENTITY_TYPES.get(entity_name)
    entity_theme = ENTITY_COLORS.get(entity_color) if entity_color else None
    if entity_type is None:
        print(f"Invalid entity name: {entity_name}")
        return None

    venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
    state = heist.state_from_venv(venv, 0)
    while not state.entity_exists(entity_type, entity_theme):
        start_level = random.randint(1, 100000)
        venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
        state = heist.state_from_venv(venv, 0)
    unchanged_obs = venv.reset()

    # Save the current position of the target entity
    original_position = state.get_entity_position(entity_type, entity_theme)

    # Move the target entity off-screen
    state.remove_entity(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

    state = heist.state_from_venv(venv, 0)

    # Restore the entity to its original position
    state.restore_entity_position(entity_type, entity_theme, original_position)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])

    model = helpers.load_interpretable_model(model_path=model_path)
    steering_layer_unchanged = ordered_layer_names[layer_number]
    steering_layer = helpers.rename_path(steering_layer_unchanged)

    model_activations = helpers.ModelActivations(model)
    model_activations.clear_hooks()
    output1, unmodified_activations = model_activations.run_with_cache(
        helpers.observation_to_rgb(unchanged_obs), [ordered_layer_names[layer_number]]
    )
    model_activations.clear_hooks()
    output2, modified_obs_activations = model_activations.run_with_cache(
        helpers.observation_to_rgb(modified_obs), [ordered_layer_names[layer_number]]
    )

    steering_vector = (
        unmodified_activations[steering_layer][0]
        - modified_obs_activations[steering_layer][0]
    )

    observation = venv.reset()
    done = False
    total_reward = 0
    frames = []
    observations = []
    count = 0
    entity_picked_up = False
    count_pickups = 0
    steps_until_pickup = 0

    # Count initial number of target entities
    initial_state = heist.state_from_venv(venv, 0)
    initial_entity_count = initial_state.count_entities(entity_type, entity_theme)

    while not done:
        if save_gif:
            frames.append(venv.render(mode="rgb_array"))

        observation = np.squeeze(observation)
        observation = np.transpose(observation, (1, 2, 0))
        converted_obs = helpers.observation_to_rgb(observation)
        action = helpers.generate_action_with_steering(
            model,
            converted_obs,
            steering_vector,
            steering_layer,
            modification_value,
            is_procgen_env=True,
        )

        observation, reward, done, info = venv.step(action)
        total_reward += reward
        observations.append(converted_obs)
        steps_until_pickup += 1
        if steps_until_pickup > 300:
            done = True

        state = heist.state_from_venv(venv, 0)
        current_entity_count = state.count_entities(entity_type, entity_theme)

        if current_entity_count < initial_entity_count:
            entity_picked_up = True
            done = True
            count_pickups += 1
            print(
                f"{entity_name.capitalize()} picked up after {steps_until_pickup} steps"
            )
        elif entity_type == ENTITY_TYPES["gem"] and reward > 0:
            entity_picked_up = True
            done = True
            count_pickups += 1
            print(
                f"{entity_name.capitalize()} picked up after {steps_until_pickup} steps"
            )

    if save_gif:
        imageio.mimsave(f"episode_steering_{episode}.gif", frames, fps=30)
        print("Saved gif!")

    if not entity_picked_up:
        print(f"{entity_name.capitalize()} was not picked up during the episode")
    else:
        print(f"{entity_name.capitalize()} picked up during the episode")

    state = heist.state_from_venv(venv, 0)
    state_vals = state.state_vals

    lock_positions_after = heist.get_lock_statuses(state_vals)

    return total_reward, steps_until_pickup, count_pickups


def steering_vector(
    venv,
    model_path,
    layer_number,
    steering_channel,
    modification_value,
    episode,
    entity_name,
    entity_color=None,
    num_levels=0,
    start_level=5,
    episode_timeout=200,
    save_gif=False,
    filepath="../gifs/unsteered_run.gif",
):
    model = helpers.load_interpretable_model(model_path=model_path)
    original_env = venv
    state = heist.state_from_venv(venv, 0)
    env_state = heist.EnvState(state.state_bytes)
    entity_type = ENTITY_TYPES.get(entity_name)
    entity_theme = ENTITY_COLORS.get(entity_color) if entity_color else None
    print(entity_theme)
    original_position = env_state.get_key_position(entity_theme)
    unchanged_obs = venv.reset()
    unsteered_actions = helpers.run_episode_and_save_as_gif(
        env=original_env,
        model=model,
        save_gif=save_gif,
        filepath=filepath + "unsteered.gif",
    )

    if entity_type is None:
        print(f"Invalid entity name: {entity_name}")
        return None

    # Move the target entity off-screen
    state.move_entity_offscreen(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

    state = heist.state_from_venv(venv, 0)

    # Restore the entity to its original position
    state.restore_entity_position(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])

    steering_layer_unchanged = ordered_layer_names[layer_number]
    steering_layer = helpers.rename_path(steering_layer_unchanged)

    model_activations = helpers.ModelActivations(model)
    model_activations.clear_hooks()
    output1, unmodified_activations = model_activations.run_with_cache(
        helpers.observation_to_rgb(unchanged_obs), [ordered_layer_names[layer_number]]
    )
    model_activations.clear_hooks()
    output2, modified_obs_activations = model_activations.run_with_cache(
        helpers.observation_to_rgb(modified_obs), [ordered_layer_names[layer_number]]
    )

    steering_vector = (
        unmodified_activations[steering_layer][0]
        - modified_obs_activations[steering_layer][0]
    )

    actions = helpers.run_episode_with_steering_channels_and_save_as_gif(
        venv,
        model,
        steering_vector,
        steering_layer=ordered_layer_names[layer_number],
        steering_channel=steering_channel,
        modification_value=modification_value,
        filepath=filepath + "steered.gif",
        save_gif=save_gif,
        episode_timeout=episode_timeout,
    )

    return steering_vector, original_position


def create_patched_steering_vector(
    original_vector, patch_position, patch_value, grid_size=(8, 8)
):
    # Create a copy of the original vector
    patched_vector = original_vector.clone()

    # Set the value at the specified position
    patched_vector[patch_position[0], patch_position[1]] = patch_value

    # Reshape the vector to ensure it matches the grid size
    reshaped_vector = patched_vector.reshape(grid_size)

    return reshaped_vector


def run_patching_experiment(
    model_path,
    layer_number,
    episode,
    channel,
    patch_position,
    patch_value,
    entity_name,
    entity_color=None,
    num_levels=0,
    start_level=random.randint(1, 100000),
    episode_timeout=200,
    save_gif=False,
):
    entity_type = ENTITY_TYPES.get(entity_name)
    entity_theme = ENTITY_COLORS.get(entity_color) if entity_color else None
    if entity_type is None:
        print(f"Invalid entity name: {entity_name}")
        return None

    venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
    state = heist.state_from_venv(venv, 0)
    while not state.entity_exists(entity_type, entity_theme):
        start_level = random.randint(1, 100000)
        venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
        state = heist.state_from_venv(venv, 0)
    print(state.entity_exists(entity_type, entity_theme))
    unchanged_obs = venv.reset()

    # Save the current position of the target entity
    original_position = state.get_entity_position(entity_type, entity_theme)

    # Move the target entity off-screen
    state.remove_entity(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

    state = heist.state_from_venv(venv, 0)

    # Restore the entity to its original position
    state.restore_entity_position(entity_type, entity_theme, original_position)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])

    model = helpers.load_interpretable_model(model_path=model_path)
    steering_layer_unchanged = ordered_layer_names[layer_number]
    steering_layer = helpers.rename_path(steering_layer_unchanged)

    model_activations = helpers.ModelActivations(model)
    model_activations.clear_hooks()
    output1, unmodified_activations = model_activations.run_with_cache(
        helpers.observation_to_rgb(unchanged_obs), [ordered_layer_names[layer_number]]
    )
    model_activations.clear_hooks()
    output2, modified_obs_activations = model_activations.run_with_cache(
        helpers.observation_to_rgb(modified_obs), [ordered_layer_names[layer_number]]
    )

    steering_vector = (
        unmodified_activations[steering_layer][0][channel]
        - modified_obs_activations[steering_layer][0][channel]
    )

    # Create the patched steering vector
    patched_steering_vector = create_patched_steering_vector(
        steering_vector, patch_position, patch_value
    )

    observation = venv.reset()
    done = False
    total_reward = 0
    frames = []
    observations = []
    count = 0
    entity_picked_up = False
    count_pickups = 0
    steps_until_pickup = 0

    # Count initial number of target entities
    initial_state = heist.state_from_venv(venv, 0)
    initial_entity_count = initial_state.count_entities(entity_type, entity_theme)

    while not done:
        if save_gif:
            frames.append(venv.render(mode="rgb_array"))

        observation = np.squeeze(observation)
        observation = np.transpose(observation, (1, 2, 0))
        converted_obs = helpers.observation_to_rgb(observation)
        action = helpers.generate_action_with_patching(
            model,
            converted_obs,
            patched_steering_vector,
            steering_layer,
            is_procgen_env=True,
        )

        observation, reward, done, info = venv.step(action)
        total_reward += reward
        observations.append(converted_obs)
        steps_until_pickup += 1
        if steps_until_pickup > 150:
            done = True

        state = heist.state_from_venv(venv, 0)
        current_entity_count = state.count_entities(entity_type, entity_theme)

        if current_entity_count < initial_entity_count:
            entity_picked_up = True
            done = True
            count_pickups += 1
            print(
                f"{entity_name.capitalize()} picked up after {steps_until_pickup} steps"
            )
        elif entity_type == ENTITY_TYPES["gem"] and reward > 0:
            entity_picked_up = True
            done = True
            count_pickups += 1
            print(
                f"{entity_name.capitalize()} picked up after {steps_until_pickup} steps"
            )

    if save_gif:
        imageio.mimsave(
            f"gifs/episode_patching_{episode}_green_key_layer_{layer_number}_channel_{channel}_grid_{patch_position[0]}_{patch_position[1]}_{patch_value}.gif",
            frames,
            fps=30,
        )
        print("Saved gif!")

    if not entity_picked_up:
        print(f"{entity_name.capitalize()} was not picked up during the episode")
    else:
        print(f"{entity_name.capitalize()} picked up during the episode")

    state = heist.state_from_venv(venv, 0)
    state_vals = state.state_vals

    lock_positions_after = heist.get_lock_statuses(state_vals)

    return total_reward, steps_until_pickup, count_pickups
