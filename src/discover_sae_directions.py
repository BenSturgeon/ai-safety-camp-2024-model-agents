# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.helpers import load_interpretable_model
import os
from tqdm import tqdm
from utils import heist
import random
import imageio
from utils.steering_experiments import ENTITY_TYPES, ENTITY_COLORS, ordered_layer_names

# %%
def create_straight_corridor(direction="horizontal", length=9, entity_positions=None):
    """
    Creates a straight corridor environment in the specified direction.
    
    Args:
        direction (str): Direction of the corridor - "horizontal", "vertical"
        length (int): Length of the corridor
        entity_positions (list): List of tuples (entity_type, entity_color, x, y) for entity placement
    
    Returns:
        venv: The created environment
    """
    # Create a default environment
    venv = heist.create_venv(
        num=1, start_level=random.randint(1000, 10000), num_levels=0
    )
    state = heist.state_from_venv(venv, 0)
    
    # Get the full grid and its dimensions
    full_grid = state.full_grid(with_mouse=False)
    height, width = full_grid.shape
    
    # Calculate the middle positions
    middle_y = height // 2
    middle_x = width // 2
    
    # Create a new grid with walls everywhere
    new_grid = np.full_like(full_grid, 51)  # 51 is BLOCKED
    
    # Define the corridor based on direction
    if direction == "horizontal":
        # Create a horizontal corridor
        new_grid[middle_y, middle_x-length//2:middle_x+length//2+1] = 100  # 100 is EMPTY
        player_x = middle_x - length // 4  # Place player on the left side
        player_y = middle_y
    elif direction == "vertical":
        # Create a vertical corridor
        new_grid[middle_y-length//2:middle_y+length//2+1, middle_x] = 100  # 100 is EMPTY
        player_x = middle_x
        player_y = middle_y - length // 4  # Place player on the top side
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")
    
    # Set the new grid
    state.set_grid(new_grid)
    
    # Remove all entities
    state.remove_all_entities()
    
    # Set the player position
    state.set_mouse_pos(player_y, player_x)
    
    # Place entities if specified
    if entity_positions:
        for entity_type, entity_color, x, y in entity_positions:
            entity_type_id = ENTITY_TYPES[entity_type]
            entity_color_id = ENTITY_COLORS[entity_color]
            state.set_entity_position(entity_type_id, entity_color_id, y, x)
    
    # Update the environment with the new state
    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        venv.reset()
    
    return venv

# %%
def create_T_junction(orientation="up", length=9, entity_positions=None):
    """
    Creates a T-junction corridor environment.
    
    Args:
        orientation (str): Direction of the T - "up", "down", "left", "right"
        length (int): Length of the corridors
        entity_positions (list): List of tuples (entity_type, entity_color, x, y) for entity placement
    
    Returns:
        venv: The created environment
    """
    # Create a default environment
    venv = heist.create_venv(
        num=1, start_level=random.randint(1000, 10000), num_levels=0
    )
    state = heist.state_from_venv(venv, 0)
    
    # Get the full grid and its dimensions
    full_grid = state.full_grid(with_mouse=False)
    height, width = full_grid.shape
    
    # Calculate the middle positions
    middle_y = height // 2
    middle_x = width // 2
    
    # Create a new grid with walls everywhere
    new_grid = np.full_like(full_grid, 51)  # 51 is BLOCKED
    
    # Create the basic horizontal corridor
    new_grid[middle_y, middle_x-length//2:middle_x+length//2+1] = 100  # 100 is EMPTY
    
    # Add the vertical part based on orientation
    if orientation == "up":
        new_grid[middle_y-length//2:middle_y, middle_x] = 100
        player_x = middle_x - length // 4
        player_y = middle_y
    elif orientation == "down":
        new_grid[middle_y:middle_y+length//2+1, middle_x] = 100
        player_x = middle_x - length // 4
        player_y = middle_y
    elif orientation == "left":
        new_grid[middle_y, middle_x-length:middle_x] = 100
        player_x = middle_x + length // 4
        player_y = middle_y
    elif orientation == "right":
        new_grid[middle_y, middle_x:middle_x+length] = 100
        player_x = middle_x - length // 4
        player_y = middle_y
    else:
        raise ValueError("Orientation must be 'up', 'down', 'left', or 'right'")
    
    # Set the new grid
    state.set_grid(new_grid)
    
    # Remove all entities
    state.remove_all_entities()
    
    # Set the player position
    state.set_mouse_pos(player_y, player_x)
    
    # Place entities if specified
    if entity_positions:
        for entity_type, entity_color, x, y in entity_positions:
            entity_type_id = ENTITY_TYPES[entity_type]
            entity_color_id = ENTITY_COLORS[entity_color]
            state.set_entity_position(entity_type_id, entity_color_id, y, x)
    
    # Update the environment with the new state
    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        venv.reset()
    
    return venv

# %%
def create_fork_corridor(orientation="left_right", length=9, entity_positions=None):
    """
    Creates a forked corridor environment where the player starts in one path and has to choose between two directions.
    
    Args:
        orientation (str): Direction of the fork - "left_right", "up_down"
        length (int): Length of the corridors
        entity_positions (list): List of tuples (entity_type, entity_color, x, y) for entity placement
    
    Returns:
        venv: The created environment
    """
    # Create a default environment
    venv = heist.create_venv(
        num=1, start_level=random.randint(1000, 10000), num_levels=0
    )
    state = heist.state_from_venv(venv, 0)
    
    # Get the full grid and its dimensions
    full_grid = state.full_grid(with_mouse=False)
    height, width = full_grid.shape
    
    # Calculate the middle positions
    middle_y = height // 2
    middle_x = width // 2
    
    # Create a new grid with walls everywhere
    new_grid = np.full_like(full_grid, 51)  # 51 is BLOCKED
    
    if orientation == "left_right":
        # Create a vertical corridor for the stem
        new_grid[middle_y-length//2:middle_y+1, middle_x] = 100
        # Create left and right branches
        new_grid[middle_y-length//2, middle_x-length//4:middle_x+length//4+1] = 100
        player_x = middle_x
        player_y = middle_y
    elif orientation == "up_down":
        # Create a horizontal corridor for the stem
        new_grid[middle_y, middle_x-length//2:middle_x+1] = 100
        # Create up and down branches
        new_grid[middle_y-length//4:middle_y+length//4+1, middle_x-length//2] = 100
        player_x = middle_x - length // 4
        player_y = middle_y
    else:
        raise ValueError("Orientation must be 'left_right' or 'up_down'")
    
    # Set the new grid
    state.set_grid(new_grid)
    
    # Remove all entities
    state.remove_all_entities()
    
    # Set the player position
    state.set_mouse_pos(player_y, player_x)
    
    # Place entities if specified
    if entity_positions:
        for entity_type, entity_color, x, y in entity_positions:
            entity_type_id = ENTITY_TYPES[entity_type]
            entity_color_id = ENTITY_COLORS[entity_color]
            state.set_entity_position(entity_type_id, entity_color_id, y, x)
    
    # Update the environment with the new state
    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        venv.reset()
    
    return venv

# %%
def visualize_environment(venv, save_path=None):
    """
    Visualizes an environment by rendering it.
    
    Args:
        venv: The environment to visualize
        save_path (str): Path to save the visualization
    
    Returns:
        Image array of the rendered environment
    """
    obs = venv.reset()
    img = venv.render(mode="rgb_array")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return img

# %%
def test_environment_generation():
    """
    Generate and visualize example environments for testing
    """
    os.makedirs("corridor_environments", exist_ok=True)
    
    # Test horizontal corridor
    h_corridor = create_straight_corridor(
        direction="horizontal",
        entity_positions=[
            ("key", "red", 15, 12),
        ]
    )
    visualize_environment(h_corridor, "corridor_environments/horizontal_corridor.png")
    
    # Test vertical corridor
    v_corridor = create_straight_corridor(
        direction="vertical",
        entity_positions=[
            ("key", "green", 12, 15),
        ]
    )
    visualize_environment(v_corridor, "corridor_environments/vertical_corridor.png")
    
    # Test T-junction
    t_junction = create_T_junction(
        orientation="up",
        entity_positions=[
            ("key", "blue", 12, 8),
            ("key", "red", 15, 12),
        ]
    )
    visualize_environment(t_junction, "corridor_environments/t_junction.png")
    
    # Test fork corridor
    fork = create_fork_corridor(
        orientation="left_right",
        entity_positions=[
            ("key", "blue", 8, 8),
            ("key", "red", 16, 8),
        ]
    )
    visualize_environment(fork, "corridor_environments/fork_corridor.png")

# %%
# Run this to test environment generation
test_environment_generation()
# %%

from utils.steering_experiments import create_specific_l_shaped_maze_env
def test_choice_mazes():
    """
    Test the straight vs L-choice mazes and the new maze permutations
    """
    os.makedirs("choice_mazes", exist_ok=True)
    
    # Test with reward at end of straight path
    straight_reward = create_straight_vs_L_maze(reward_location="straight")
    visualize_environment(straight_reward, "choice_mazes/reward_straight.png")
    
    # Test with reward at end of L-shaped path
    l_reward = create_straight_vs_L_maze(reward_location="L")
    visualize_environment(l_reward, "choice_mazes/reward_L.png")
    
    # Test the original maze permutation (variant 0)
    original_maze = create_specific_l_shaped_maze_env(maze_variant=0, entity_type="gem", entity_color="red")
    visualize_environment(original_maze, "choice_mazes/original_maze.png")
    
    # Test a different maze permutation (variant 4 - S-shaped)
    s_maze = create_specific_l_shaped_maze_env(maze_variant=4, entity_type="key", entity_color="blue")
    visualize_environment(s_maze, "choice_mazes/s_shaped_maze.png")
    
    print("Done testing choice mazes with different permutations.")

# %%
# Run this to test the choice mazes
# test_choice_mazes()

# %%

def test_maze_permutations():
    """
    Generate and visualize all 8 variants of the L-shaped mazes
    """
    os.makedirs("maze_permutations", exist_ok=True)
    
    # Test each maze variant with different entity types/colors for variety
    configurations = [
        (0, "key", "blue"),      # Original L-maze
        (1, "key", "blue"),     # Flipped horizontally
        (2, "key", "blue"),    # Flipped vertically
        (3, "key", "blue"),      # Rotated 180 degrees
        (4, "key", "blue"),     # S-shaped path
        (5, "key", "blue"),    # Inverted L
        (6, "key", "blue"),      # T-junction
        (7, "key", "blue")      # U-shaped path
    ]
    
    for i, (variant, entity_type, entity_color) in enumerate(configurations):
        # Create the maze variant
        maze = create_specific_l_shaped_maze_env(
            maze_variant=variant,
            entity_type=entity_type,
            entity_color=entity_color
        )
        
        # Visualize and save
        visualize_environment(
            maze, 
            f"maze_permutations/variant_{variant}_{entity_type}_{entity_color}.png"
        )
        
        # Close the environment
        maze.close()
        
    print(f"Generated 8 maze permutations in the 'maze_permutations' directory")

# %%
# Run this to test all maze permutations
test_maze_permutations()


# %%
