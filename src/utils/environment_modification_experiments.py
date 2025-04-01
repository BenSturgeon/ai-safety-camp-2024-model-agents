from utils import helpers
from utils import heist
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


def flip_maze_pattern(pattern):
    """
    Flips a maze pattern vertically to make the coordinates more intuitive.
    This allows for designing mazes where (0,0) is in the bottom-left corner.
    
    Args:
        pattern (np.array): The original maze pattern
        
    Returns:
        np.array: The vertically flipped maze pattern
    """
    return np.flip(pattern, axis=0)

def create_specific_l_shaped_maze_env(maze_variant=0, entities=None):
    """
    Creates one of 8 permutations of similar L-shaped mazes with a white mouse (player)
    and multiple target entities (gem/key) at specified positions.
    
    Args:
        maze_variant (int): Which maze variant to create (0-7)
        entities (list): List of entity dictionaries with the following structure:
            [
                {
                    "type": "key" or "gem",
                    "color": "red", "green", or "blue",
                    "position": (y, x)  # Using intuitive coordinates where (0,0) is bottom-left
                },
                ...
            ]
            If None, a default entity (blue key) will be placed based on the maze variant.
    
    ✅ Step 1: Create environment and ensure entity existence
    ✅ Step 2: Initialize new maze grid with specific pattern
    ✅ Step 3: Remove entities and explicitly place them
    
    Returns:
        venv: Updated virtual environment
    """
    # ✅ Step 1: Ensure entity exists in the environment
    while True:
        venv = heist.create_venv(
            num=1, start_level=random.randint(1000, 10000), num_levels=0
        )
        state = heist.state_from_venv(venv, 0)
        if state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["red"]):
            break

    # ✅ Step 2: Get the full grid and its dimensions
    full_grid = state.full_grid(with_mouse=False)
    height, width = full_grid.shape

    # Define maze size and coordinates
    maze_size = 7
    middle_y = height // 2
    middle_x = width // 2

    start_y = middle_y - maze_size // 2
    start_x = middle_x - maze_size // 2

    # Initialize grid with walls (51=BLOCKED)
    new_grid = np.full_like(full_grid, 51)

    # Define 8 different maze patterns (1=path, 0=wall, 2=target entity position, 3=player position)
    # Each pattern represents a different configuration with L-shaped paths
    # These patterns are already flipped vertically for more intuitive design where (0,0) is at the bottom left

    maze_patterns = [
        # Variant 0: Original L-maze from example (player bottom-left, target top-right) 
        # right
        flip_maze_pattern(np.array([
            [0, 1, 1, 2, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 0], 
            [1, 3, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])),
        # Variant 1: Flipped horizontally (player bottom-right, target top-left)
        # right
        flip_maze_pattern(np.array([
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 3, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 2, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])),
        # Variant 2: Flipped vertically (player top-left, target bottom-right)
        # up
        flip_maze_pattern(np.array([
            [0, 0, 1, 0, 0, 0, 0],
            [1, 2, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 3, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0]
        ])),
        # Variant 3: Rotated 180 degrees (player top-right, target bottom-left)
        # up
        flip_maze_pattern(np.array([
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 2, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 3, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0]
        ])),
        # Variant 4: S-shaped path (player bottom-left, target top-right)
        # left
        flip_maze_pattern(np.array([
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 3, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 2, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])),
        # Variant 5: Inverted L (player bottom-left, target mid-right)
        # left
        flip_maze_pattern(np.array([
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 3, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 2, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])),
        # Variant 6: T-junction (player bottom, target top-right)
        # down
        flip_maze_pattern(np.array([
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 3, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 2, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])),
        # Variant 7: U-shaped path (player bottom-left, target bottom-right)
        # down
        flip_maze_pattern(np.array([
            [1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 3, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 2, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]))
    ]
    
    # Helper function to convert from intuitive y-coordinate to actual grid y-coordinate
    def convert_y_coord(y_pos, maze_size=maze_size):
        """
        Convert from intuitive y-coordinate (where 0 is bottom) to the actual grid 
        y-coordinate (where 0 is top)
        """
        return (maze_size - 1) - y_pos
    # Get the selected pattern
    if maze_variant < 0 or maze_variant >= len(maze_patterns):
        maze_variant = 0  # Default to original maze if out of range
    pattern = maze_patterns[maze_variant]
    
    # Find player position (3) and target (2) in pattern
    player_pos = target_pos = None
    for i in range(maze_size):
        for j in range(maze_size):
            val = pattern[i,j]
            if val == 3:
                player_pos = (i,j) 
            elif val == 2:
                target_pos = (i,j)
            if val in [1,2,3]:
                new_grid[i,j] = 100 # Empty space
    
    if not player_pos:
        raise ValueError("No player position (3) found in maze pattern")
    
    player_y, player_x = player_pos

    state.set_grid(new_grid)
    state.remove_all_entities()
    state.set_mouse_pos(player_y, player_x)

    if not entities and target_pos is not None:
        target_y, target_x = target_pos
        entities = [{
            "type": "key",
            "color": "blue", 
            "position": (convert_y_coord(target_y-1), target_x-1)
        }]
    
    # Place all entities at their specified locations
    if entities:  # Add this check to ensure entities is not None
        for entity in entities:
            entity_type = entity["type"]
            entity_color = entity["color"]
            intuitive_y, x = entity["position"]
            
            # Convert the intuitive y-coordinate to actual grid y-coordinate
            y = start_y + convert_y_coord(intuitive_y)
            
            # Place entity at desired location
            state.set_entity_position(
                ENTITY_TYPES[entity_type], 
                ENTITY_COLORS[entity_color],
                y, 
                x + start_x
            )
            
            # Update the environment state after each entity placement to ensure it takes effect
            state_bytes = state.state_bytes
            if state_bytes is not None:
                venv.env.callmethod("set_state", [state_bytes])

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


def create_custom_maze_sequence(maze_patterns, maze_size=7):
    """
    Creates a sequence of maze environments based on a list of patterns with numerical encoding
    where different numbers represent different elements:
    - 0 = wall/empty space
    - 1 = open corridor
    - 2 = player (mouse)
    - 3 = gem
    - 4 = blue key
    - 5 = green key
    - 6 = red key
    - 7 = blue lock
    - 8 = green lock
    - 9 = red lock
    
    Args:
        maze_patterns (list): List of 2D numpy arrays representing maze patterns with the 
                              numerical encoding specified above
        maze_size (int): Size of the maze (default: 7x7)
    
    Returns:
        tuple: (observations, venv) where observations is a list of observations from each
               maze environment and venv is the final environment
    """
    # Map of encoding values to entity types and colors
    entity_mapping = {
        2: {"type": "player", "color": None},
        3: {"type": "gem", "color": None},
        4: {"type": "key", "color": "blue"},
        5: {"type": "key", "color": "green"},
        6: {"type": "key", "color": "red"},
        7: {"type": "lock", "color": "blue"},
        8: {"type": "lock", "color": "green"},
        9: {"type": "lock", "color": "red"}
    }
    
    observations = []
    
    # Create initial environment
    while True:
        venv = heist.create_venv(
            num=1, start_level=random.randint(1000, 10000), num_levels=0
        )
        state = heist.state_from_venv(venv, 0)
        # Make sure the environment has the key entities we need
        if (state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["red"]) and
            state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["blue"]) and
            state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["green"])):
            break
    
    # Get grid dimensions
    full_grid = state.full_grid(with_mouse=False)
    height, width = full_grid.shape
    
    # Calculate center position to place the maze
    middle_y = height // 2
    middle_x = width // 2
    start_y = middle_y - maze_size // 2
    start_x = middle_x - maze_size // 2
    
    # Helper function to convert from intuitive y-coordinate to actual grid y-coordinate
    def convert_y_coord(y_pos, maze_size=maze_size):
        """
        Convert from intuitive y-coordinate (where 0 is bottom) to the actual grid 
        y-coordinate (where 0 is top)
        """
        return (maze_size - 1) - y_pos
    
    # Process each maze pattern in the sequence
    for pattern_idx, pattern in enumerate(maze_patterns):
        print(f"Processing maze pattern {pattern_idx+1}/{len(maze_patterns)}")
        
        # Make sure pattern is a numpy array with correct dimensions
        if not isinstance(pattern, np.ndarray):
            pattern = np.array(pattern)
        
        if pattern.shape != (maze_size, maze_size):
            raise ValueError(f"Maze pattern must be {maze_size}x{maze_size}, got {pattern.shape}")
        
        # Create walls everywhere by default
        new_grid = np.full_like(full_grid, BLOCKED)
        
        # Collect entity positions and player position
        player_pos = None
        entities = []
        
        # First pass: Process the maze pattern to identify corridors and entities
        for i in range(maze_size):
            for j in range(maze_size):
                value = pattern[i, j]
                
                # Convert intuitive coordinates (where 0,0 is bottom-left)
                actual_i = convert_y_coord(i)
                
                if value == 1:  # Open corridor
                    new_grid[start_y + actual_i, start_x + j] = EMPTY
                elif value >= 2:  # Entity or player
                    # Mark this position as empty (walkable)
                    new_grid[start_y + actual_i, start_x + j] = EMPTY
                    
                    if value == 2:  # Player
                        player_pos = (actual_i, j)
                    elif value in entity_mapping:  # Entity
                        entity_data = entity_mapping[value]
                        entities.append({
                            "type": entity_data["type"],
                            "color": entity_data["color"],
                            "position": (i, j)  # Store intuitive coordinates
                        })
        
        # Set the new grid layout
        state.set_grid(new_grid)
        
        # Update the environment state after setting the grid
        state_bytes = state.state_bytes
        if state_bytes is not None:
            venv.env.callmethod("set_state", [state_bytes])
        
        # Remove all existing entities
        state.remove_all_entities()
        
        # Update the environment state after removing entities
        state_bytes = state.state_bytes
        if state_bytes is not None:
            venv.env.callmethod("set_state", [state_bytes])
        
        # Set player position if found in the pattern
        if player_pos:
            mouse_y, mouse_x = player_pos
            state.set_mouse_pos(start_y + mouse_y, start_x + mouse_x)
            
            # Update the environment state after setting player position
            state_bytes = state.state_bytes
            if state_bytes is not None:
                venv.env.callmethod("set_state", [state_bytes])
        else:
            # If no player position specified in the pattern, place player in a valid empty cell
            valid_positions = []
            for i in range(maze_size):
                for j in range(maze_size):
                    actual_i = convert_y_coord(i)
                    if new_grid[start_y + actual_i, start_x + j] == EMPTY:
                        valid_positions.append((actual_i, j))
            
            if valid_positions:
                mouse_y, mouse_x = random.choice(valid_positions)
                state.set_mouse_pos(start_y + mouse_y, start_x + mouse_x)
                
                # Update the environment state after setting player position
                state_bytes = state.state_bytes
                if state_bytes is not None:
                    venv.env.callmethod("set_state", [state_bytes])
        
        # Place all entities
        for entity in entities:
            entity_type = entity["type"]
            entity_color = entity["color"]
            intuitive_y, x = entity["position"]
            
            # Convert the intuitive y-coordinate to actual grid y-coordinate
            y = start_y + convert_y_coord(intuitive_y)
            
            # Place entity at desired location
            state.set_entity_position(
                ENTITY_TYPES[entity_type], 
                ENTITY_COLORS[entity_color] if entity_color else None,
                y, 
                x + start_x
            )
            
            # Update the environment state after each entity placement
            state_bytes = state.state_bytes
            if state_bytes is not None:
                venv.env.callmethod("set_state", [state_bytes])
        
        # Final reset to ensure all changes are applied
        obs = venv.reset()
        observations.append(obs)
    
    return observations, venv


def create_example_maze_sequence(entity1=4, entity2=None):
    """
    Creates an example sequence of maze environments to demonstrate the custom maze functionality.
    
    This example creates a sequence showing an entity moving around the maze in a predefined path.
    
    Args:
        entity1 (int): Code for the main entity to track (default: 4, blue key)
        entity2 (int): Code for a secondary static entity, or None to not include (default: None)
    
    Returns:
        tuple: (observations, venv) - List of observations and final environment
    """
    # Define maze patterns (7x7 grids)
    # 0 = wall, 1 = corridor, 2 = player, 3 = stationary entity, 4 = moving entity
    # Values: 3 = gem, 4 = blue key, 5 = green key, 6 = red key, 7 = blue lock, 8 = green lock, 9 = red lock
    patterns = [
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 4, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),
        
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 4, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),
        
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 4, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),

        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 4, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),
        
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 4, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),
        
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 4, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),

        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 4, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),

        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 4, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),

        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 4, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),
        
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 4, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ]),
        
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 4, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0]
        ])
    ]

    maze_patterns = []

    for pattern in patterns:
        modified_pattern = pattern.copy()

        loc_moving_entity = (pattern == 4)
        loc_stationary_entity = (pattern == 3)

        modified_pattern[loc_moving_entity] = entity1

        if entity2 is not None:
            modified_pattern[loc_stationary_entity] = entity2
        else:
            modified_pattern[loc_stationary_entity] = 1

        maze_patterns.append(modified_pattern)

    return create_custom_maze_sequence(maze_patterns)


def create_box_maze(entity1=4, entity2=None):
    """
    Creates an example sequence of maze environments to demonstrate the custom maze functionality.
    
    This example creates a sequence showing an entity moving around the maze in a predefined path.
    
    Args:
        entity1 (int): Code for the main entity to track (default: 4, blue key)
        entity2 (int): Code for a secondary static entity, or None to not include (default: None)
    
    Returns:
        tuple: (observations, venv) - List of observations and final environment
    """
    # Define maze patterns (7x7 grids)
    # 0 = wall, 1 = corridor, 2 = player, 3 = first entity, 4 = second entity
    # Values: 3 = gem, 4 = blue key, 5 = green key, 6 = red key, 7 = blue lock, 8 = green lock, 9 = red lock
    pattern = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 3, 0, 0, 0, 4, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 2]])
        
    # Create a copy to modify, ensuring original placeholders are preserved for lookup
    final_pattern = pattern.copy()

    # Find original locations of placeholders
    loc_entity1 = (pattern == 3)
    loc_entity2 = (pattern == 4)

    # Replace placeholder 3 with entity1
    final_pattern[loc_entity1] = entity1

    # Replace placeholder 4 with entity2 or corridor (1)
    if entity2 is not None:
        final_pattern[loc_entity2] = entity2
    else:
        final_pattern[loc_entity2] = 1  # Replace with corridor if entity2 is None


    return create_custom_maze_sequence([final_pattern])

def create_two_entity_maze(pattern, entity1_type="blue_key", entity2_type="red_key"):
    """
    Creates a maze environment with a player and 2 entities.
    The pattern should use:
    - 0 for walls
    - 1 for corridors 
    - 2 for player position
    - 4 for first entity position (was 3)
    - 5 for second entity position (was 4)

    Args:
        pattern (np.ndarray): 2D numpy array representing maze layout
        entity1_type (str): Type of first entity ("blue_key", "red_key", "green_key", "gem")
        entity2_type (str): Type of second entity ("blue_key", "red_key", "green_key", "gem")

    Returns:
        tuple: (observation, venv) - Initial observation and environment
    """
    # Map entity types to their properties
    entity_type_map = {
        "blue_key": ("key", "blue"),
        "red_key": ("key", "red"), 
        "green_key": ("key", "green"),
        "gem": ("gem", None)
    }

    pattern1 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 5, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])
    
    pattern2 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 5, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])
    
    pattern3 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 5, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])

    pattern4 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 5, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])
    
    pattern5 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 5, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])

    pattern6 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 5, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])

    pattern7 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 5, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])

    pattern8 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 5, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])

    pattern9 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 5, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])
    
    pattern10 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 5, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])
    
    pattern11 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 5, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0]
    ])
    
 
    # Create sequence of mazes
    maze_patterns = [
        pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, 
        pattern9, pattern10, pattern11
    ]

    # Find positions of entities
    entity1_pos = np.where(pattern == 4)  # Changed from 3 to 4
    entity2_pos = np.where(pattern == 5)  # Changed from 4 to 5
    player_pos = np.where(pattern == 2)

    # Convert pattern to basic maze (walls and corridors only)
    maze_pattern = pattern.copy()
    maze_pattern[pattern > 1] = 1  # Convert all special tiles to corridors

    # Create entities list
    entities = []
    if len(entity1_pos[0]) > 0:
        entity_type, entity_color = entity_type_map[entity1_type]
        entities.append({
            "type": entity_type,
            "color": entity_color,
            "position": (entity1_pos[0][0], entity1_pos[1][0])
        })
    
    if len(entity2_pos[0]) > 0:
        entity_type, entity_color = entity_type_map[entity2_type]
        entities.append({
            "type": entity_type,
            "color": entity_color,
            "position": (entity2_pos[0][0], entity2_pos[1][0])
        })

    # Create the maze with the pattern and entities
    player_position = (player_pos[0][0], player_pos[1][0]) if len(player_pos[0]) > 0 else None
    observations, venv = create_custom_maze_sequence([maze_pattern], entities=[entities], player_positions=[player_position])
    
    return observations[0], venv


def run_custom_maze_sequence(maze_patterns, save_path=None, render_as_gif=False, fps=5):
    """
    Runs a sequence of custom mazes and optionally saves the observations as images or a GIF.
    
    Args:
        maze_patterns (list): List of 2D numpy arrays representing maze patterns
        save_path (str, optional): Path to save the images or GIF. If None, nothing is saved.
        render_as_gif (bool): Whether to render as a GIF or individual images
        fps (int): Frames per second for the GIF
    
    Returns:
        list: List of observations from the maze sequence
    """
    # Create the maze sequence
    observations, venv = create_custom_maze_sequence(maze_patterns)
    
    # Convert observations to renderable format
    rendered_frames = []
    for i, obs in enumerate(observations):
        # Render the observation as an RGB array
        frame = venv.render(mode="rgb_array")
        rendered_frames.append(frame)
        
        # Save individual frames if requested
        if save_path and not render_as_gif:
            frame_path = f"{save_path}_frame_{i}.png"
            imageio.imwrite(frame_path, frame)
            print(f"Saved frame {i} to {frame_path}")
    
    # Save as GIF if requested
    if save_path and render_as_gif:
        gif_path = f"{save_path}.gif"
        imageio.mimsave(gif_path, rendered_frames, fps=fps)
        print(f"Saved GIF to {gif_path}")
    
    # Clean up
    venv.close()
    
    return observations


def run_patched_steering_experiment(
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


def create_entity_movement_sequence(base_maze, entity_positions, num_frames=5):
    """
    Creates a sequence of maze environments where entities move along specified paths.
    
    Note: This function only generates the maze patterns with proper numerical encoding.
    The actual state byte updates happen when these patterns are processed by 
    create_custom_maze_sequence().
    
    Args:
        base_maze (np.ndarray): Base maze layout with corridors (1) and walls (0)
        entity_positions (dict): Dictionary mapping entity codes (2-9) to lists of positions
                                 Each position is a tuple (y, x) in intuitive coordinates
                                 where (0,0) is the bottom-left corner
        num_frames (int): Number of intermediate frames between positions
        
    Returns:
        list: List of maze patterns that can be passed to create_custom_maze_sequence
    """
    # Make sure base_maze is a numpy array
    if not isinstance(base_maze, np.ndarray):
        base_maze = np.array(base_maze)
    
    # Validate base maze - should only contain 0s and 1s
    if not np.all(np.isin(base_maze, [0, 1])):
        raise ValueError("Base maze should only contain 0s (walls) and 1s (corridors)")
    
    maze_size = base_maze.shape[0]
    if base_maze.shape != (maze_size, maze_size):
        raise ValueError(f"Base maze must be square, got shape {base_maze.shape}")
    
    # Function to interpolate positions
    def interpolate_positions(start_pos, end_pos, num_steps):
        y1, x1 = start_pos
        y2, x2 = end_pos
        # Create a straight-line path between positions
        y_steps = np.linspace(y1, y2, num_steps + 2)[1:-1]  # Exclude start and end
        x_steps = np.linspace(x1, x2, num_steps + 2)[1:-1]  # Exclude start and end
        return [(int(round(y)), int(round(x))) for y, x in zip(y_steps, x_steps)]
    
    # Create first pattern with all entities at their starting positions
    first_pattern = base_maze.copy()
    for entity_code, positions in entity_positions.items():
        if len(positions) > 0:
            y, x = positions[0]
            first_pattern[y, x] = entity_code
    
    maze_patterns = [first_pattern]
    
    # Track the current position of each entity
    current_positions = {entity_code: positions[0] for entity_code, positions in entity_positions.items() if len(positions) > 0}
    
    # For each entity, generate the movement sequence
    max_steps = max([len(positions) for positions in entity_positions.values() if len(positions) > 0])
    
    for step in range(1, max_steps):
        # Create a new pattern based on the previous one
        new_pattern = base_maze.copy()
        
        # Update each entity that has a position for this step
        for entity_code, positions in entity_positions.items():
            if step < len(positions):
                # Entity has a position for this step
                current_pos = positions[step-1]
                next_pos = positions[step]
                
                # Generate intermediate frames
                if num_frames > 0:
                    intermediates = interpolate_positions(current_pos, next_pos, num_frames)
                    
                    # Create a pattern for each intermediate position
                    for intermediate_pos in intermediates:
                        intermediate_pattern = base_maze.copy()
                        
                        # Add all entities at their current positions
                        for ec, pos in current_positions.items():
                            if ec != entity_code:  # Other entities stay in place
                                y, x = pos
                                intermediate_pattern[y, x] = ec
                        
                        # Update the moving entity
                        y, x = intermediate_pos
                        intermediate_pattern[y, x] = entity_code
                        
                        maze_patterns.append(intermediate_pattern)
                
                # Update the current position of this entity
                current_positions[entity_code] = next_pos
            
            # Add the entity to the new pattern
            if entity_code in current_positions:
                y, x = current_positions[entity_code]
                new_pattern[y, x] = entity_code
        
        # Add the new pattern to the sequence
        maze_patterns.append(new_pattern)
    
    return maze_patterns


def create_example_entity_movement():
    """
    Creates an example of entity movement through a maze.
    
    This example shows:
    1. A player (2) moving through an L-shaped corridor
    2. A blue key (4) that moves towards the player
    3. A red lock (9) that blocks part of the path
    
    Returns:
        list: List of maze patterns showing entity movement
    """
    # Define a simple L-shaped maze with corridors (1) and walls (0)
    base_maze = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Define paths for different entities
    entity_positions = {
        # Player (2) moves from bottom-left up and right
        2: [(5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
        
        # Blue key (4) moves from top-right down to meet player
        4: [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1)],
        
        # Red lock (9) appears to block the path and then disappears
        9: [(1, 3), (1, 3), (1, 3), (1, 3), (1, 3), (0, 0)]  # Last position is off-grid (disappears)
    }
    
    # Generate the sequence with 1 intermediate frame between positions
    maze_patterns = create_entity_movement_sequence(base_maze, entity_positions, num_frames=1)
    
    return maze_patterns


def run_and_save_example_movement(save_path="maze_movement", render_as_gif=True):
    """
    Runs the example entity movement and saves it as a GIF or images.
    
    Args:
        save_path (str): Path to save the output (without extension)
        render_as_gif (bool): Whether to render as a GIF or individual images
        
    Returns:
        list: List of observations from the maze sequence
    """
    maze_patterns = create_example_entity_movement()
    return run_custom_maze_sequence(maze_patterns, save_path, render_as_gif, fps=3)


def create_custom_maze_with_movements(base_pattern, entity_movements, num_frames=1, save_path=None, render_as_gif=True, fps=3):
    """
    Creates and runs a custom maze with specified entity movements in one convenient function.
    
    This is a high-level wrapper that combines several functions to make it easy to:
    1. Define a base maze layout
    2. Specify how entities should move through that maze
    3. Generate the sequence of observations
    4. Optionally save as a GIF or image sequence
    
    Args:
        base_pattern (np.ndarray): 2D array representing the base maze with:
            - 0 = wall/empty space
            - 1 = open corridor
            - 2 = player initial position (optional)
            - 3-9 = initial positions of other entities (optional)
        entity_movements (dict): Dictionary mapping entity codes (2-9) to lists of positions
                                Each position is a tuple (y, x) in intuitive coordinates
        num_frames (int): Number of intermediate frames between positions
        save_path (str, optional): Path to save the images or GIF. If None, nothing is saved.
        render_as_gif (bool): Whether to render as a GIF or individual images
        fps (int): Frames per second for the GIF
        
    Returns:
        tuple: (observations, venv) - List of observations and final environment
    """
    # Extract the base maze layout (corridors and walls only)
    base_maze = np.zeros_like(base_pattern)
    # Copy just the corridors
    base_maze[base_pattern == 1] = 1
    
    # Extract initial entity positions from the base pattern
    initial_positions = {}
    for code in range(2, 10):  # Entity codes 2-9
        positions = np.where(base_pattern == code)
        if len(positions[0]) > 0:
            # Get the first position found for this entity
            y, x = positions[0][0], positions[1][0]
            
            # If this entity has movement data, add its initial position
            if code in entity_movements:
                movement_path = entity_movements[code]
                # If the first position in the movement path doesn't match the one in the pattern,
                # insert the initial position from the pattern
                if len(movement_path) == 0 or movement_path[0] != (y, x):
                    entity_movements[code] = [(y, x)] + movement_path
            else:
                # Entity exists in pattern but has no movement data, add as static
                entity_movements[code] = [(y, x)]
    
    # Generate the maze pattern sequence
    maze_patterns = create_entity_movement_sequence(base_maze, entity_movements, num_frames)
    
    # Run the sequence and return results
    return run_custom_maze_sequence(maze_patterns, save_path, render_as_gif, fps)


def run_example_complete_maze_sequence(save_path="complete_maze_example", render_as_gif=True):
    """
    Demonstrates how to use create_custom_maze_with_movements with a complete example.
    
    This example shows:
    1. A player navigating a maze to collect a key and unlock a door
    2. The maze has a more complex layout with multiple paths
    3. A key that appears and can be collected
    4. A lock that blocks a path until the key is collected
    
    Returns:
        tuple: (observations, venv) - List of observations and final environment
    """
    # Define a maze with:
    # - 0 = wall
    # - 1 = corridor
    # - 2 = player starting position
    # - 4 = blue key starting position
    # - 7 = blue lock starting position
    base_pattern = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 7, 0, 1, 0],
        [0, 1, 0, 1, 0, 4, 0],
        [0, 2, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Define movement paths (each position is y, x)
    entity_movements = {
        # Player (2) movement path - going to get the key and then the lock
        2: [
            (5, 1),  # Start
            (5, 2), (5, 3),  # Move right
            (4, 3), (3, 3),  # Move up to the lock
            (3, 3),  # Wait at the lock (can't pass yet)
            (4, 3), (5, 3),  # Move back down
            (5, 4), (5, 5),  # Move right
            (4, 5),  # Move up to get the key
            (4, 5),  # Collect the key (key disappears)
            (5, 5), (5, 4), (5, 3),  # Move back left
            (4, 3), (3, 3),  # Move up to the lock again
            (3, 3),  # Unlock the lock (lock disappears)
            (2, 3), (1, 3), (1, 4), (1, 5)  # Complete the path to top-right
        ],
        
        # Blue key (4) - stays in place until collected
        4: [(4, 5), (4, 5), (4, 5), (4, 5), (4, 5), (4, 5), (4, 5), (4, 5), (4, 5), (0, 0)],
        
        # Blue lock (7) - stays in place until unlocked 
        7: [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (0, 0)]
    }
    
    # Run the sequence with 0 intermediate frames (direct movement)
    return create_custom_maze_with_movements(
        base_pattern, 
        entity_movements, 
        num_frames=0,
        save_path=save_path,
        render_as_gif=render_as_gif,
        fps=2
    )
