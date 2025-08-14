"""
Bias-corrected maze configurations for ablation experiments.
Creates fork mazes where the stem points in the bias direction, 
so the agent moves away from entities by default.
"""

import numpy as np
import random
from utils.create_intervention_mazes import create_custom_maze_sequence


def create_bias_corrected_fork_maze(bias_direction="up"):
    """
    Create a fork maze where the stem points in the bias direction.
    This ensures the agent's bias moves it away from the entities.
    
    Args:
        bias_direction: The model's dominant bias direction ("up", "down", "left", "right")
    
    Returns:
        tuple: (observations, venv) - List of observations and final environment
    """
    
    # Define fork patterns for each bias direction
    # The stem should point in the bias direction, entities at the fork ends
    
    if bias_direction == "up":
        # Stem points up, entities at bottom of fork
        # Fix coordinate system: in the environment, lower Y coordinates are at the top
        pattern = np.array([
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [3, 0, 4, 0, 5, 0, 6]
        ])
    
    elif bias_direction == "down":
        # Stem points down, entities at top of fork
                pattern = np.array([
            [3, 0, 4, 0, 5, 0, 6],  # Entities at fork ends (bottom)
            [1, 0, 1, 0, 1, 0, 1],  # Fork ends
            [1, 1, 1, 1, 1, 1, 1],  # Fork junction
            [0, 0, 0, 1, 0, 0, 0],  # Stem
            [0, 0, 0, 1, 0, 0, 0],  # Stem
            [0, 0, 0, 1, 0, 0, 0],  # Stem
            [0, 0, 0, 2, 0, 0, 0]   # Player at stem end (top)
        ])

    
    elif bias_direction == "left":
        # Stem points left, entities at right side of fork (proper fork shape)
        pattern = np.array([
            [0, 0, 0, 0, 1, 1, 3],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 4],
            [2, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 5],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 6]
        ])
    
    elif bias_direction == "right":
        # Stem points right, entities at left side of fork (proper fork shape)
        pattern = np.array([
            [3, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [4, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 2],
            [5, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [6, 1, 1, 0, 0, 0, 0]
        ])
    
    else:
        # Default to up if unknown bias direction
        print(f"Warning: Unknown bias direction '{bias_direction}', using 'up' as default")
        return create_bias_corrected_fork_maze("up")
    
    # Randomize entity positions (same as original fork maze)
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


def create_bias_corrected_corners_maze(bias_direction="up"):
    """
    Create a corners maze where the player is positioned opposite to the bias direction.
    
    Args:
        bias_direction: The model's dominant bias direction ("up", "down", "left", "right")
    
    Returns:
        tuple: (observations, venv) - List of observations and final environment
    """
    
    if bias_direction == "up":
        # Player at bottom, entities at top corners
        pattern = np.array([
            [3, 0, 0, 0, 0, 0, 4],  # Entities at top corners
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],  # Middle corridor
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [5, 0, 0, 2, 0, 0, 6]   # Player at bottom center, entities at bottom corners
        ])
    
    elif bias_direction == "down":
        # Player at top, entities at bottom corners
        pattern = np.array([
            [3, 0, 0, 2, 0, 0, 4],  # Player at top center, entities at top corners
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],  # Middle corridor
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [5, 0, 0, 0, 0, 0, 6]   # Entities at bottom corners
        ])
    
    elif bias_direction == "left":
        # Player at right, entities at left corners
        pattern = np.array([
            [3, 0, 0, 0, 0, 0, 4],  # Entities at top corners
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [5, 1, 1, 1, 1, 1, 2],  # Player at right, entity at left
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [6, 0, 0, 0, 0, 0, 0]   # Entity at bottom-left
        ])
    
    elif bias_direction == "right":
        # Player at left, entities at right corners
        pattern = np.array([
            [3, 0, 0, 0, 0, 0, 4],  # Entities at top corners
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [2, 1, 1, 1, 1, 1, 5],  # Player at left, entity at right
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 6]   # Entity at bottom-right
        ])
    
    else:
        # Default to up if unknown bias direction
        print(f"Warning: Unknown bias direction '{bias_direction}', using 'up' as default")
        return create_bias_corrected_corners_maze("up")
    
    # Randomize entity positions
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


# Test function to visualize the patterns
def visualize_bias_corrected_patterns():
    """Print ASCII representation of bias-corrected patterns for debugging."""
    directions = ["up", "down", "left", "right"]
    
    for direction in directions:
        print(f"\n=== FORK MAZE - Bias Direction: {direction} ===")
        # Create a test pattern without randomization
        if direction == "up":
            pattern = np.array([
                [0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [3, 0, 4, 0, 5, 0, 6]
            ])
        elif direction == "down":
            pattern = np.array([
                [3, 0, 4, 0, 5, 0, 6],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0]
            ])
        elif direction == "left":
            pattern = np.array([
                [0, 0, 0, 1, 0, 0, 3],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 4],
                [2, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 5],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 6]
            ])
        elif direction == "right":
            pattern = np.array([
                [3, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [4, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 2],
                [5, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [6, 0, 0, 1, 0, 0, 0]
            ])
        
        # Print pattern with symbols
        symbol_map = {0: '█', 1: ' ', 2: 'P', 3: 'G', 4: 'B', 5: 'R', 6: 'Y'}
        for row in pattern:
            print(''.join(symbol_map.get(cell, '?') for cell in row))
        print("Legend: P=Player, G=Gem, B=Blue, R=Red, Y=Yellow, █=Wall, =Path")


if __name__ == "__main__":
    visualize_bias_corrected_patterns()