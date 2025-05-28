from utils import helpers
from utils import heist
import random
import numpy as np
import imageio
import torch
from procgen.gym_registration import make_env
import os

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

ENTITY_CODE_DESCRIPTION = {
    3: "gem",
    4: "blue_key",
    5: "green_key",
    6: "red_key",
    7: "blue_lock",
    8: "green_lock",
    9: "red_lock"
}

# Path to the viable seeds file
VIABLE_SEEDS_FILE = os.path.join(os.path.dirname(__file__), 'viable_seeds.txt')
LOADED_SEEDS = []

def load_viable_seeds():
    global LOADED_SEEDS
    if not LOADED_SEEDS: # Load only once
        try:
            with open(VIABLE_SEEDS_FILE, 'r') as f:
                LOADED_SEEDS = [int(line.strip()) for line in f if line.strip()]
            if not LOADED_SEEDS:
                print(f"Warning: {VIABLE_SEEDS_FILE} is empty or contains no valid seeds. Using default seed 0.")
                LOADED_SEEDS = [0] # Fallback to a default seed
            else:
                print(f"Successfully loaded {len(LOADED_SEEDS)} seeds from {VIABLE_SEEDS_FILE}")
        except FileNotFoundError:
            print(f"Warning: {VIABLE_SEEDS_FILE} not found. Using default seed 0.")
            LOADED_SEEDS = [0] # Fallback
        except ValueError:
            print(f"Warning: {VIABLE_SEEDS_FILE} contains non-integer values. Check the file. Using default seed 0.")
            LOADED_SEEDS = [0] # Fallback
load_viable_seeds() # Load seeds when the module is imported

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
    
    # Create initial environment using a randomly selected pre-verified seed
    if not LOADED_SEEDS: # Should have been loaded, but as a safeguard
        print("Error: Viable seeds list is empty. Cannot create environment.")
        # Potentially raise an error or return a dummy environment
        # For now, let's try to load again or use a hardcoded default if critical
        load_viable_seeds() # Attempt to reload
        if not LOADED_SEEDS: # Still no seeds
             selected_seed = 0 # Last resort default
             print(f"Critical fallback: using seed {selected_seed} as no viable seeds could be loaded.")
        else:
            selected_seed = random.choice(LOADED_SEEDS)
    else:
        selected_seed = random.choice(LOADED_SEEDS)

    # print(f"  Using seed: {selected_seed} for this maze instance.") # Optional: for debugging

    venv = heist.create_venv(
        num=1, 
        start_level=selected_seed, 
        num_levels=1, # Ensures start_level acts as a unique seed
        distribution_mode="easy" 
    )
    state = heist.state_from_venv(venv, 0)
    
    # ---- START DEBUG BLOCK ----
    # Check for key presence immediately after venv creation with the selected seed
    key_types_to_check = {
        "red": ENTITY_COLORS["red"],
        "green": ENTITY_COLORS["green"],
        "blue": ENTITY_COLORS["blue"]
    }
    missing_keys_at_init = []
    for color_name, color_code in key_types_to_check.items():
        if not state.entity_exists(ENTITY_TYPES["key"], color_code):
            missing_keys_at_init.append(color_name)
    
    if missing_keys_at_init:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"WARNING from create_custom_maze_sequence: Seed {selected_seed} DID NOT produce all required keys at initial venv creation!")
        print(f"  Missing keys: {', '.join(missing_keys_at_init)}")
        print(f"  This is BEFORE any custom grid modifications or entity removals.")
        print(f"  Listing all entities found in state for seed {selected_seed}:")
        
        # Helper to get readable entity names
        type_val_to_name = {v: k for k, v in ENTITY_TYPES.items()}
        theme_val_to_name = {v: k for k, v in ENTITY_COLORS.items()}
        
        if not state.state_vals["ents"]:
            print("    No entities found in state.state_vals[\"ents\"] list.")
        else:
            for i, ent_data in enumerate(state.state_vals["ents"]):
                ent_type_val = ent_data["image_type"].val
                ent_theme_val = ent_data["image_theme"].val
                ent_x_pos = ent_data["x"].val
                ent_y_pos = ent_data["y"].val
                
                type_name = type_val_to_name.get(ent_type_val, f"UnknownType({ent_type_val})")
                readable_name = type_name
                if type_name in ["key", "lock"]:
                    color_name = theme_val_to_name.get(ent_theme_val, f"UnknownTheme({ent_theme_val})")
                    readable_name = f"{color_name.capitalize()} {type_name}"
                
                print(f"    Ent {i}: {readable_name} at (x={ent_x_pos:.2f}, y={ent_y_pos:.2f}), raw_type={ent_type_val}, raw_theme={ent_theme_val}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    # ---- END DEBUG BLOCK ----
    
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
        # print(f"Processing maze pattern {pattern_idx+1}/{len(maze_patterns)}") # Commented out for tqdm
        
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


def create_trident_maze():
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
            [4, 1, 1, 1, 1, 1, 6],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 2, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [5, 1, 1, 1, 1, 1, 3]])
    
    return create_custom_maze_sequence([pattern])

def create_cross_maze():
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
            [0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [4, 1, 1, 2, 1, 1, 3],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0]])


    # Randomize the positions of the entities (values 3, 4, 5, 6)
    entity_values = [3, 4, 5, 6]  # gem, blue key, green key, red key
    random.shuffle(entity_values)
    
    # Create a mapping from original values to shuffled values
    entity_mapping = {
        3: entity_values[0],
        4: entity_values[1],
        5: entity_values[2],
        6: entity_values[3]
    }
    
    # Apply the mapping to the pattern
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            if pattern[i, j] in entity_mapping:
                pattern[i, j] = entity_mapping[pattern[i, j]]
    
    return create_custom_maze_sequence([pattern])


def create_fork_maze():
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
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [5, 0, 4, 0, 3, 0, 6]])


    # Randomize the positions of the entities (values 3, 4, 5, 6)
    entity_values = [3, 4, 5, 6]  # gem, blue key, green key, red key
    random.shuffle(entity_values)
    
    # Create a mapping from original values to shuffled values
    entity_mapping = {
        3: entity_values[0],
        4: entity_values[1],
        5: entity_values[2],
        6: entity_values[3]
    }
    
    # Apply the mapping to the pattern
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            if pattern[i, j] in entity_mapping:
                pattern[i, j] = entity_mapping[pattern[i, j]]
    
    return create_custom_maze_sequence([pattern])


def create_corners_maze():
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
            [3, 0, 0, 2, 0, 0, 4],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [5, 0, 0, 0, 0, 0, 6]])


    # Randomize the positions of the entities (values 3, 4, 5, 6)
    entity_values = [3, 4, 5, 6]  # gem, blue key, green key, red key
    random.shuffle(entity_values)
    
    # Create a mapping from original values to shuffled values
    entity_mapping = {
        3: entity_values[0],
        4: entity_values[1],
        5: entity_values[2],
        6: entity_values[3]
    }
    
    # Apply the mapping to the pattern
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            if pattern[i, j] in entity_mapping:
                pattern[i, j] = entity_mapping[pattern[i, j]]
    
    return create_custom_maze_sequence([pattern])


def create_sequential_maze():
    """
    Creates a maze with distinct paths to each entity type.
    The maze is designed to test sequential entity collection behavior.
    
    Returns:
        tuple: (observations, venv) - List of observations and final environment
    """
    # 0 = wall, 1 = corridor, 2 = player, 3 = first entity, 4 = second entity
    # Values: 3 = gem, 4 = blue key, 5 = green key, 6 = red key, 7 = blue lock, 8 = green lock, 9 = red lock
    pattern = np.array([
        [2, 0, 0, 0, 0, 0, 0],  # Row 0: Player start in middle
        [1, 1, 4, 0, 0, 0, 0],  # Row 1: Single path down
        [1, 0, 0, 0, 3, 1, 1],  # Row 2: Three paths branch out
        [1, 1, 5, 0, 0, 0, 1],  # Row 3: Paths continue
        [1, 0, 0, 0, 6, 0, 1],  # Row 4: Paths continue
        [1, 0, 0, 0, 1, 0, 1],  # Row 5: Paths continue
        [1, 1, 1, 1, 1, 1, 1]   # Row 6: ATZs at the end of each path
    ], dtype=np.int32)

    # Randomize the positions of the entities (values 3, 4, 5, 6)
    entity_values = [3, 4, 5, 6]  # gem, blue key, green key, red key
    random.shuffle(entity_values)
    
    # Create a mapping from original values to shuffled values
    entity_mapping = {
        3: entity_values[0],
        4: entity_values[1],
        5: entity_values[2],
        6: entity_values[3]
    }
    
    # Apply the mapping to the pattern
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            if pattern[i, j] in entity_mapping:
                pattern[i, j] = entity_mapping[pattern[i, j]]
    
    return create_custom_maze_sequence([pattern])