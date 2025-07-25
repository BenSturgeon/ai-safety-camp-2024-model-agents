#!/usr/bin/env python
# coding: utf-8
#This file acts as a means to decode and edit the underlying state of the heist environment for the purposes of our interpretability work.

# The code is based off Monte's code in procgen tools with modifications to account for the differences in environment.

import os
import gym
import random
import numpy as np
import imageio
import matplotlib.pyplot as plt
import typing
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
from procgen import ProcgenGym3Env
import struct
import typing
from typing import Tuple, Dict, Callable, List, Optional
from dataclasses import dataclass
import sys
import os



from policies_impala import ImpalaCNN
from procgen_tools.procgen_wrappers import VecExtractDictObs, TransposeFrame, ScaledFloatFrame

from gym3 import ToBaselinesVecEnv


ordered_layer_names  = {
    1: 'conv1a',
    2: 'pool1',
    3: 'conv2a',
    4: 'conv2b',
    5: 'pool2',
    6: 'conv3a',
    7: 'pool3',
    8: 'conv4a',
    9: 'pool4',
    10: 'fc1',
    11: 'fc2',
    12: 'fc3',
    13: 'value_fc',
    14: 'dropout_conv',
    15: 'dropout_fc'
}


ENTITY_COLORS = {
    "blue": 0,
    "green": 1,
    "red": 2
}

ENTITY_TYPES = {
    "key": 2,
    "lock": 1,
    "gem": 9,
    "player": 0
}


# Constants
KEY_COLORS = {0: 'blue',1: 'green',2: 'red',}
MOUSE = 0
KEY = 2
LOCKED_DOOR = 1
WORLD_DIM = 25
EMPTY = 100
BLOCKED = 51
GEM= 9
KEY_ON_RING = 11  # HUD key icon entity


HEIST_STATE_DICT_TEMPLATE = [
    ["int", "SERIALIZE_VERSION"],
    ["string", "game_name"],
    ["int", "options.paint_vel_info"],
    ["int", "options.use_generated_assets"],
    ["int", "options.use_monochrome_assets"],
    ["int", "options.restrict_themes"],
    ["int", "options.use_backgrounds"],
    ["int", "options.center_agent"],
    ["int", "options.debug_mode"],
    ["int", "options.distribution_mode"],
    ["int", "options.use_sequential_levels"],
    ["int", "options.use_easy_jump"],
    ["int", "options.plain_assets"],
    ["int", "options.physics_mode"],
    ["int", "grid_step"],
    ["int", "level_seed_low"],
    ["int", "level_seed_high"],
    ["int", "game_type"],
    ["int", "game_n"],
    ["int", "level_seed_rand_gen.is_seeded"],
    ["string", "level_seed_rand_gen.str"],
    ["int", "rand_gen.is_seeded"],
    ["string", "rand_gen.str"],
    ["float", "step_data.reward"],
    ["int", "step_data.done"],
    ["int", "step_data.level_complete"],
    ["int", "action"],
    ["int", "timeout"],
    ["int", "current_level_seed"],
    ["int", "prev_level_seed"],
    ["int", "episodes_remaining"],
    ["int", "episode_done"],
    ["int", "last_reward_timer"],
    ["float", "last_reward"],
    ["int", "default_action"],
    ["int", "fixed_asset_seed"],
    ["int", "cur_time"],
    ["int", "is_waiting_for_step"],
    ["int", "grid_size"],
    ["int", "ents.size"],
    [
        "loop",
        "ents",
        "ents.size",
        [
            ["float", "x"],
            ["float", "y"],
            ["float", "vx"],
            ["float", "vy"],
            ["float", "rx"],
            ["float", "ry"],
            ["int", "type"],
            ["int", "image_type"],
            ["int", "image_theme"],
            ["int", "render_z"],
            ["int", "will_erase"],
            ["int", "collides_with_entities"],
            ["float", "collision_margin"],
            ["float", "rotation"],
            ["float", "vrot"],
            ["int", "is_reflected"],
            ["int", "fire_time"],
            ["int", "spawn_time"],
            ["int", "life_time"],
            ["int", "expire_time"],
            ["int", "use_abs_coords"],
            ["float", "friction"],
            ["int", "smart_step"],
            ["int", "avoids_collisions"],
            ["int", "auto_erase"],
            ["float", "alpha"],
            ["float", "health"],
            ["float", "theta"],
            ["float", "grow_rate"],
            ["float", "alpha_decay"],
            ["float", "climber_spawn_x"],
        ],
    ],
    ["int", "use_procgen_background"],
    ["int", "background_index"],
    ["float", "bg_tile_ratio"],
    ["float", "bg_pct_x"],
    ["float", "char_dim"],
    ["int", "last_move_action"],
    ["int", "move_action"],
    ["int", "special_action"],
    ["float", "mixrate"],
    ["float", "maxspeed"],
    ["float", "max_jump"],
    ["float", "action_vx"],
    ["float", "action_vy"],
    ["float", "action_vrot"],
    ["float", "center_x"],
    ["float", "center_y"],
    ["int", "random_agent_start"],
    ["int", "has_useful_vel_info"],
    ["int", "step_rand_int"],
    ["int", "asset_rand_gen.is_seeded"],
    ["string", "asset_rand_gen.str"],
    ["int", "main_width"],
    ["int", "main_height"],
    ["int", "out_of_bounds_object"],
    ["float", "unit"],
    ["float", "view_dim"],
    ["float", "x_off"],
    ["float", "y_off"],
    ["float", "visibility"],
    ["float", "min_visibility"],
    ["int", "w"],
    ["int", "h"],
    ["int", "data.size"],
    ["loop", "data", "data.size", [["int", "i"]]],
    ["int", "num_keys"],
    ["int", "world_dim"],
    ["int", "_has_keys_size"],
    ["loop", "has_keys", "_has_keys_size", [["int", "key_state"]]],
    ["int", "num_locked_doors"],
    ["loop", "locked_doors", "num_locked_doors", [
        ["float", "x"],
        ["float", "y"],
        ["int", "door_state"],
    ]],
    ["int", "END_OF_BUFFER"],
]


@dataclass
class StateValue:
    val: typing.Any
    idx: int


StateValues = typing.Dict[
    str, typing.Any
]  

Square = typing.Tuple[int, int]





def load_model( model_path = '../model_final.pt'):
    env_name = "procgen:procgen-heist-v0"  

    env = gym.make(env_name, start_level=100, num_levels=200, render_mode="rgb_array", distribution_mode="easy") #remove render mode argument to go faster but not produce images 
    observation_space = env.observation_space
    action_space = env.action_space.n
    model = ImpalaCNN(observation_space, action_space)
    model.load_from_file(model_path, device="cpu")
    return model



def wrap_venv(venv) -> ToBaselinesVecEnv:
    "Wrap a vectorized env, making it compatible with the gym apis, transposing, scaling, etc."

    venv = ToBaselinesVecEnv(venv)  
    venv = VecExtractDictObs(venv, "rgb")

    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv  

def create_venv(
    num: int, start_level: int, num_levels: int, num_threads: int = 1, distribution_mode: str = "easy"
):
    """
    Create a wrapped venv. See https://github.com/openai/procgen#environment-options for params

    num=1 - The number of parallel environments to create in the vectorized env.

    num_levels=0 - The number of unique levels that can be generated. Set to 0 to use unlimited levels.

    start_level=0 - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
    """
    venv = ProcgenGym3Env(
        num=num,
        env_name="heist",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        num_threads=num_threads,
        render_mode="rgb_array",
    )
    venv = wrap_venv(venv)
    return venv
    



DEBUG = (
    False  
)

def inner_grid(grid: np.ndarray, assert_=True) -> np.ndarray:
    """
    Get the inside of the maze, ie. the stuff within the outermost walls.
    inner_grid(inner_grid(x)) = inner_grid(x) for all x.
    """
    # Find the amount of padding on the maze, where padding is BLOCKED
    # Use numpy to test if each square is BLOCKED
    # If it is, then it's part of the padding
    bl = 0
    # Check if the top, bottom, left, and right are all blocked
    while (
        (grid[bl, :] == BLOCKED).all()
        and (grid[-bl - 1, :] == BLOCKED).all()
        and (grid[:, bl] == BLOCKED).all()
        and (grid[:, -bl - 1] == BLOCKED).all()
    ):
        bl += 1

    return (
        grid[bl:-bl, bl:-bl] if bl > 0 else grid
    )  


def outer_grid(grid: np.ndarray, assert_=True) -> np.ndarray:
    """
    The inverse of inner_grid(). Could also be called "pad_grid".
    """
    bl = (WORLD_DIM - len(grid)) // 2
    outer = np.pad(grid, bl, "constant", constant_values=BLOCKED)
    if assert_:
        assert (inner_grid(outer, assert_=False) == grid).all()
    return outer

def get_legal_mouse_positions(grid: np.ndarray, entities: List[Dict[str, StateValue]]):
    """Return a list of legal mouse positions in the grid, returned as a list of tuples."""
    occupied_positions = set()
    for entity in entities:
        x = entity["x"].val
        y = entity["y"].val
        
        if math.isnan(x):
            x = 0  
        if math.isnan(y):
            y = 0 
        
        ex = math.floor(x)
        ey = math.floor(y)
        occupied_positions.add((ex, ey))

    legal_positions = [
        (x, y)
        for x in range(grid.shape[0])
        for y in range(grid.shape[1])
        if grid[x, y] == EMPTY and (x, y) not in occupied_positions
    ]
    
    return legal_positions

def get_cheese_venv_pair(
    seed: int, has_cheese_tup: Tuple[bool, bool] = (True, False)
):
    "Return a venv of 2 environments from a seed, with cheese in the first environment if has_cheese_tup[0] and in the second environment if has_cheese_tup[1]."
    venv = create_venv(num=2, start_level=seed, num_levels=1)

    for idx in range(2):
        if has_cheese_tup[idx]:
            continue  # Skip if we want cheese in this environment
        remove_cheese(venv, idx=idx)

    return venv

def copy_venv(venv, idx: int):
    "Return a copy of venv number idx. WARNING: After level is finished, the copy will be reset."
    sb = venv.env.callmethod("get_state")[idx]
    env = create_venv(num=1, start_level=0, num_levels=1)
    env.env.callmethod("set_state", [sb])
    return env

class EnvState:
    def __init__(self, state_bytes: bytes):
        self.state_bytes = state_bytes
        self.key_indices = {"blue": 0, "green": 1, "red": 2}


    @property
    def state_vals(self):
        return _parse_maze_state_bytes_handling_buffer_error(self.state_bytes)

    @property
    def world_dim(self):
        return self.state_vals["world_dim"].val

    def full_grid(self, with_mouse=True):
        "Get numpy (world_dim, world_dim) grid of the maze. Includes the mouse by default."
        world_dim = self.world_dim
        grid = np.array(
            [dd["i"].val for dd in self.state_vals["data"]]
        ).reshape(world_dim, world_dim)
        if with_mouse:
            grid[self.mouse_pos] = MOUSE

        return grid

    def inner_grid(self, with_mouse=True):
        "Get inner grid of the maze. Includes the mouse by default."
        return inner_grid(self.full_grid(with_mouse=with_mouse))

    @property
    def mouse_pos(self) -> Tuple[int, int]:
        "Get (x, y) position of mouse in grid."
        ents = self.state_vals["ents"][0]
        # flipped turns out to be oriented right for grid.
        return int(ents["y"].val), int(ents["x"].val)

    def set_mouse_pos(self, x: int, y: int):
        """
        Set the mouse position in the maze state bytes. Much more optimized than parsing and serializing the whole state.
        *WARNING*: This uses *outer coordinates*, not inner.
        """
        # FIXME(slow): grabbing state_vals requires a call to parse the state bytes.
        state_vals = self.state_vals
        state_vals["ents"][0]["x"].val = float(y) + 0.5
        state_vals["ents"][0]["y"].val = float(x) + 0.5
        self.state_bytes = _serialize_maze_state(state_vals)

    def get_key_colors(self):
        key_colors = []
        state_values = self.state_vals
        
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2:  # Check if the entity is a key
                key_index = ents["image_theme"].val
                key_color = KEY_COLORS.get(key_index, "Unknown")
                if key_color not in key_colors:
                    key_colors.append(key_color)
        
        return key_colors

    def add_entity(self, entity_type, entity_theme, x, y):
        """
        Add a new entity to the environment.
        
        :param entity_type: The type of entity (e.g., ENTITY_TYPES['key'])
        :param entity_theme: The theme/color of the entity (e.g., ENTITY_COLORS['blue'])
        :param x: The x-coordinate of the entity
        :param y: The y-coordinate of the entity
        """
        state_values = self.state_vals
        new_entity = {
            "x": StateValue(float(y) + 0.5, 0),
            "y": StateValue(float(x) + 0.5, 0),
            "vx": StateValue(0.0, 0),
            "vy": StateValue(0.0, 0),
            "rx": StateValue(0.0, 0),
            "ry": StateValue(0.0, 0),
            "type": StateValue(entity_type, 0),
            "image_type": StateValue(entity_type, 0),
            "image_theme": StateValue(entity_theme, 0),
            "render_z": StateValue(0, 0),
            "will_erase": StateValue(0, 0),
            "collides_with_entities": StateValue(1, 0),
            "collision_margin": StateValue(0.0, 0),
            "rotation": StateValue(0.0, 0),
            "vrot": StateValue(0.0, 0),
            "is_reflected": StateValue(0, 0),
            "fire_time": StateValue(0, 0),
            "spawn_time": StateValue(0, 0),
            "life_time": StateValue(0, 0),
            "expire_time": StateValue(0, 0),
            "use_abs_coords": StateValue(0, 0),
            "friction": StateValue(0.0, 0),
            "smart_step": StateValue(0, 0),
            "avoids_collisions": StateValue(0, 0),
            "auto_erase": StateValue(0, 0),
            "alpha": StateValue(1.0, 0),
            "health": StateValue(1.0, 0),
            "theta": StateValue(0.0, 0),
            "grow_rate": StateValue(0.0, 0),
            "alpha_decay": StateValue(0.0, 0),
            "climber_spawn_x": StateValue(0.0, 0),
        }
        
        state_values["ents"].append(new_entity)
        state_values["ents.size"].val += 1
        self.state_bytes = _serialize_maze_state(state_values)



    def set_key_position(self, key_index, x, y):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val== 2:
                if key_index == ents["image_theme"].val:
                    ents["x"].val = float(x) 
                    ents["y"].val = float(y) 
        self.state_bytes = _serialize_maze_state(state_values)

    def set_lock_position(self, key_index, x, y):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val== 1:
                if key_index == ents["image_theme"].val:
                    ents["x"].val = float(y) + .5
                    ents["y"].val = float(x) + .5
        self.state_bytes = _serialize_maze_state(state_values)

    def set_gem_position(self, x, y):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val== 9:
                ents["x"].val = float(y) + .5
                ents["y"].val = float(x) + .5
        self.state_bytes = _serialize_maze_state(state_values)
    
    def set_grid(self, grid: np.ndarray, pad=False):
        """
        Set the grid of the maze.
        """
        if pad:
            grid = outer_grid(grid, assert_=False)
        assert grid.shape == (self.world_dim, self.world_dim)

        state_vals = self.state_vals
        grid = grid.copy()  # might need to remove mouse if in grid
        if (grid == MOUSE).sum() > 0:
            x, y = get_mouse_pos(grid)

            state_vals["ents"][0]["x"].val = (
                float(y) + 0.5
            )  # flip again to get back to original orientation
            state_vals["ents"][0]["y"].val = float(x) + 0.5

            grid[x, y] = EMPTY

        world_dim = state_vals["world_dim"].val
        assert grid.shape == (world_dim, world_dim)
        for i, dd in enumerate(state_vals["data"]):
            dd["i"].val = int(grid.ravel()[i])

        self.state_bytes = _serialize_maze_state(state_vals)
    
    def entity_exists(self, entity_type, entity_theme=None):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == entity_type:
                if entity_theme is None or ents["image_theme"].val == entity_theme:
                    # Check if the entity is on the screen
                    if ents["x"].val >= 0 and ents["y"].val >= 0:
                        return True
        return False
    
    def remove_all_entities(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            ents["x"].val = -1
            ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)
    
    def clear_entities(self):
        """
        Properly removes all entities from the state, by clearing the 'ents' list.
        """
        state_values = self.state_vals
        state_values["ents"] = []
        state_values["ents.size"].val = 0
        self.state_bytes = _serialize_maze_state(state_values)

    def remove_gem(self):
        state_values = self.state_vals  
        for ents in state_values["ents"]:
            if ents["image_type"].val== 9:
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def remove_player(self):
        state_values = self.state_vals  
        for ents in state_values["ents"]:
            if ents["image_type"].val== 0:
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_specific_keys_and_locks(self, colors_to_delete):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val in [1, 2]:  # Check if the entity is a key or lock
                if ents["image_theme"].val in [KEY_COLORS[color] for color in colors_to_delete]:
                    ents["x"].val = -1
                    ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_keys(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2:  # Check if the entity is a key
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_locks(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 1:  # Check if the entity is a lock
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_key(self, key_index):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2 and ents["image_theme"].val == key_index:  # Check if the entity is a key
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_keys_and_locks(self, stage):

        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val in [1, 2]:  # Check if the entity is a key or lock
                if stage == 2 and ents["image_theme"].val == self.key_indices["blue"]:
                    ents["x"].val = -1
                    ents["y"].val = -1
                elif stage == 3 and ents["image_theme"].val in [self.key_indices["blue"], self.key_indices["green"]]:
                    ents["x"].val = -1
                    ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def get_lock_positions(self):
        lock_positions = []
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 1:  # Check if the entity is a lock
                lock_positions.append((ents["x"].val, ents["y"].val))
        return lock_positions

    def get_key_position(self, key_index):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2 and ents["image_theme"].val == key_index:
                return (ents["x"].val, ents["y"].val)
        raise ValueError("Key not found")

    def get_key_positions(self):
        key_positions = []
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2:  # Check if the entity is a key
                key_positions.append((ents["x"].val, ents["y"].val))
        return key_positions

    def get_lock_statuses(self):
        lock_statuses = []
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 1:  # Check if the entity is a lock
                lock_statuses.append(ents)
        return lock_statuses
    
    def delete_specific_keys(self, key_indices: list):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2 and ents["image_theme"].val in key_indices:
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_specific_locks(self, lock_indices: list):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 1 and ents["image_theme"].val in lock_indices:
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)
    
    def remove_entity(self, entity_type, entity_theme=None):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == entity_type:
                if entity_theme is None or ents["image_theme"].val == entity_theme:
                    ents["x"].val = -1
                    ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def set_entity_position(self, entity_type, entity_theme, x, y):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == entity_type:
                if entity_theme is None or ents["image_theme"].val == entity_theme:
                    ents["x"].val = float(y) + .5
                    ents["y"].val = float(x) + .5
        self.state_bytes = _serialize_maze_state(state_values)

    def move_entity_offscreen(self, entity_type, entity_theme=None):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == entity_type:
                if entity_theme is None or ents["image_theme"].val == entity_theme:
                    self.original_position = (ents["x"].val, ents["y"].val)
                    ents["x"].val = -1  # Move far off-screen
                    ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)
    
    def get_entity_position(self, entity_type, entity_theme=None):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == entity_type:
                if entity_theme is None or ents["image_theme"].val == entity_theme:
                    return (ents["x"].val, ents["y"].val)
        return None  # Return None if the entity is not found

    def restore_entity_position(self, entity_type, entity_theme=None, original_position=None):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == entity_type:
                if entity_theme is None or ents["image_theme"].val == entity_theme:
                    ents["x"].val, ents["y"].val = original_position
        self.state_bytes = _serialize_maze_state(state_values)
    
    def count_entities(self, entity_type, entity_theme=None):
        count = 0
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == entity_type:
                if entity_theme is None or ents["image_theme"].val == entity_theme:
                    # Check if the entity is on the screen
                    if ents["x"].val >= 0 and ents["y"].val >= 0:
                        count += 1
        return count

    def get_entities(self, on_screen_only: bool = True) -> List[Dict[str, StateValue]]:
        """
        Returns a list of all entities in the current state.
        Each entity is a dictionary of its properties (StateValue objects).
        By default, only returns entities that are on-screen (x.val >= 0 and y.val >= 0, and not NaN).
        """
        all_raw_entities = self.state_vals.get("ents")
        if not all_raw_entities:
            return []

        if not on_screen_only:
            return all_raw_entities # Return all entities as they are

        # Filter for on-screen entities
        on_screen_entities: List[Dict[str, StateValue]] = []
        for entity_data in all_raw_entities:
            x_state_value = entity_data.get("x")
            y_state_value = entity_data.get("y")

            x_val = float('nan')
            y_val = float('nan')

            if x_state_value is not None:
                x_val = x_state_value.val
            if y_state_value is not None:
                y_val = y_state_value.val
            
            is_valid_and_on_screen = (
                not math.isnan(x_val) and \
                not math.isnan(y_val) and \
                x_val >= 0 and \
                y_val >= 0
            )
            
            if is_valid_and_on_screen:
                on_screen_entities.append(entity_data)
        return on_screen_entities
    
    def set_ring_key_position(self, colour_idx: int, x: float, y: float):
        """Move the HUD key icon (KEY_ON_RING) for given colour to normalized screen coords (x,y)."""
        KEY_ON_RING = 11
        state_vals = self.state_vals  # Get fresh reference at start
        for ent in state_vals["ents"]:
            if ent["type"].val == KEY_ON_RING and ent["image_theme"].val == colour_idx:
                ent["x"].val = float(x)
                ent["y"].val = float(y)
                break
        self.state_bytes = _serialize_maze_state(state_vals)

    def set_hud_keys(self, blue_visible=False, green_visible=False, red_visible=False, use_natural_positions=True):
        """
        Show/hide HUD key icons with simple boolean flags.
        
        Now that we've fixed the template with the missing length field, this works perfectly:
        - has_keys[0] = 0/1 (blue key flag)
        - has_keys[1] = 0/1 (green key flag)  
        - has_keys[2] = 0/1 (red key flag)
        
        Args:
            blue_visible (bool): Whether blue key icon should be visible. Default False.
            green_visible (bool): Whether green key icon should be visible. Default False.
            red_visible (bool): Whether red key icon should be visible. Default False.
            use_natural_positions (bool): If True, positions keys at their natural game locations
                                        in the top-right corner. Default True.
        
        Example:
            # Show blue and red keys
            state.set_hud_keys(blue_visible=True, red_visible=True)
            venv.env.callmethod("set_state", [state.state_bytes])
            
            # Show all three keys
            state.set_hud_keys(blue_visible=True, green_visible=True, red_visible=True)
            venv.env.callmethod("set_state", [state.state_bytes])
            
            # Hide all keys  
            state.set_hud_keys()
            venv.env.callmethod("set_state", [state.state_bytes])
        """
        KEY_ON_RING = 11
        
        # Natural HUD key positions (discovered from gameplay analysis)
        NATURAL_POSITIONS = {
            0: (0.962, 0.022),  # Blue key - rightmost
            1: (0.902, 0.022),  # Green key - middle
            2: (0.843, 0.022),  # Red key - leftmost
        }
        
        # Get fresh state_vals reference (CRITICAL working pattern)
        state_vals = self.state_vals
        
        # Position HUD keys in natural locations if requested
        if use_natural_positions:
            for ent in state_vals["ents"]:
                if ent["type"].val == KEY_ON_RING:
                    theme = ent["image_theme"].val
                    if theme in NATURAL_POSITIONS:
                        x, y = NATURAL_POSITIONS[theme]
                        ent["x"].val = float(x)
                        ent["y"].val = float(y)
        
        # Set the boolean flags directly - simple and clean!
        # Leave num_keys and _has_keys_size alone, only touch the actual flags
        if len(state_vals["has_keys"]) >= 1:  # Blue key slot
            state_vals["has_keys"][0]["key_state"].val = 1 if blue_visible else 0
        if len(state_vals["has_keys"]) >= 2:  # Green key slot  
            state_vals["has_keys"][1]["key_state"].val = 1 if green_visible else 0
        if len(state_vals["has_keys"]) >= 3:  # Red key slot
            state_vals["has_keys"][2]["key_state"].val = 1 if red_visible else 0
        
        # Serialize using the working pattern
        self.state_bytes = _serialize_maze_state(state_vals)




def _get_neighbors(x, y):
    "Get the neighbors of (x, y) in the grid"
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _ingrid(grid: np.ndarray, n):
    "Is (x, y) in the grid?"
    return 0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1]

def get_empty_neighbors(grid: np.ndarray, x, y):
    "Get the empty neighbors of (x, y) in the grid"
    return [
        n
        for n in _get_neighbors(x, y)
        if _ingrid(grid, n) and grid[n] != BLOCKED
    ]

def get_gem_pos(grid: np.ndarray, flip_y: bool = False) -> Square:
    "Get (row, col) position of the cheese in the grid. Note that the numpy grid is flipped along the y-axis, relative to rendered images."
    num_cheeses = (grid == GEM).sum()
    if num_cheeses == 0:
        return None
    row, col = np.where(grid == GEM)
    row, col = row[0], col[0]
    return ((WORLD_DIM - 1) - row if flip_y else row), col

def get_mouse_pos_sv(state_vals: StateValues) -> Square:
    """Get the mouse position from state_vals"""
    ents = state_vals["ents"][0]
    return int(ents["y"].val), int(ents["x"].val)

def get_mouse_pos(
    grid: np.ndarray, flip_y: bool = False
) -> typing.Tuple[int, int]:
    "Get (x, y) position of the mouse in the grid"
    num_mouses = (grid == MOUSE).sum()
    assert num_mouses == 1, f"{num_mouses} mice, should be 1"
    row, col = np.where(grid == MOUSE)
    row, col = row[0], col[0]
    return ((WORLD_DIM - 1) - row if flip_y else row), col


def state_from_venv(venv, idx: int = 0) -> EnvState:
    """
    Get the maze state from the venv.
    """
    state_bytes_list = venv.env.callmethod("get_state")
    return EnvState(state_bytes_list[idx])

def _parse_maze_state_bytes(state_bytes: bytes, assert_=DEBUG) -> StateValues:


    def read_fixed(sb, idx, fmt):
        sz = struct.calcsize(fmt)
        if idx + sz > len(sb):
            # print(f"Warning: Buffer underflow at index {idx} with size {sz}, buffer length {len(sb)}. Returning default value.")
            if fmt == '@i':  # Default for integers
                default_val = 0
            elif fmt == '@f':  # Default for floats
                default_val = float('nan')  # or 0.0 depending on what makes sense for your context
    
            # Decide whether to advance idx or not
            # Option 1: Do not advance idx if you want to try reading again later or handle this case differently
            # Option 2: Advance idx to skip the expected number of bytes (more risky if data is critical)
            idx += sz  # Uncomment this line if you choose to advance idx
    
            return default_val, idx
    
        val = struct.unpack(fmt, sb[idx : (idx + sz)])[0]
        idx += sz
        return val, idx

    read_int = lambda sb, idx: read_fixed(sb, idx, "@i")
    read_float = lambda sb, idx: read_fixed(sb, idx, "@f")

    def read_string(sb, idx):
        sz, idx = read_int(sb, idx)
        val = sb[idx : (idx + sz)].decode("ascii")
        idx += sz
        return val, idx


    # Function to process a value definition and return a value (called recursively for loops)
    def parse_value(vals, val_def, idx):
        typ = val_def[0]
        name = val_def[1]
        # print((typ, name))
        if typ == "int":
            val, idx = read_int(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "float":
            val, idx = read_float(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "string":
            val, idx = read_string(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "loop":
            len_name = val_def[2]
            loop_val_defs = val_def[3]
            loop_len = vals[len_name].val
            vals[name] = []
            for _ in range(loop_len):
                vals_this = {}
                for loop_val_def in loop_val_defs:
                    idx = parse_value(vals_this, loop_val_def, idx)
                vals[name].append(vals_this)
        return idx

    # Dict to hold values
    vals = {}

    # Loop over list of value defs, parsing each
    idx = 0
    for val_def in HEIST_STATE_DICT_TEMPLATE:
        idx = parse_value(vals, val_def, idx)

    if idx != len(state_bytes):
        print(f"[heist.py WARNING] _parse_maze_state_bytes: Final index {idx} does not match state_bytes length {len(state_bytes)}. Potential corruption or incomplete parse.")
    if assert_:
        assert (
            _serialize_maze_state(vals, assert_=False) == state_bytes
        ), "serialize(deserialize(state_bytes)) != state_bytes"
    return vals

def _parse_maze_state_bytes_handling_buffer_error(state_bytes: bytes, assert_=DEBUG) -> StateValues:


    def read_fixed(sb, idx, fmt):
        sz = struct.calcsize(fmt)
        if idx + sz > len(sb):
            # print(f"Warning: Buffer underflow at index {idx} with size {sz}, buffer length {len(sb)}. Returning default value.")
            if fmt == '@i':  # Default for integers
                default_val = 0
            elif fmt == '@f':  # Default for floats
                default_val = float('nan')  # or 0.0 depending on what makes sense for your context
    
            # Decide whether to advance idx or not
            # Option 1: Do not advance idx if you want to try reading again later or handle this case differently
            # Option 2: Advance idx to skip the expected number of bytes (more risky if data is critical)
            # idx += sz  # Uncomment this line if you choose to advance idx
    
            return default_val, idx
    
        val = struct.unpack(fmt, sb[idx : (idx + sz)])[0]
        idx += sz
        return val, idx

    read_int = lambda sb, idx: read_fixed(sb, idx, "@i")
    read_float = lambda sb, idx: read_fixed(sb, idx, "@f")

    def read_string(sb, idx):
        sz, idx = read_int(sb, idx)
        val = sb[idx : (idx + sz)].decode("ascii")
        idx += sz
        return val, idx


    # Function to process a value definition and return a value (called recursively for loops)
    def parse_value(vals, val_def, idx):
        typ = val_def[0]
        name = val_def[1]
        # print((typ, name))
        if typ == "int":
            val, idx = read_int(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "float":
            val, idx = read_float(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "string":
            val, idx = read_string(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "loop":
            len_name = val_def[2]
            loop_val_defs = val_def[3]
            loop_len = vals[len_name].val
            vals[name] = []
            for _ in range(loop_len):
                vals_this = {}
                for loop_val_def in loop_val_defs:
                    idx = parse_value(vals_this, loop_val_def, idx)
                vals[name].append(vals_this)
        return idx

    # Dict to hold values
    vals = {}

    # Loop over list of value defs, parsing each
    idx = 0
    for val_def in HEIST_STATE_DICT_TEMPLATE:
        idx = parse_value(vals, val_def, idx)


    if assert_:
        assert (
            _serialize_maze_state(vals, assert_=False) == state_bytes
        ), "serialize(deserialize(state_bytes)) != state_bytes"
    return vals
def _serialize_maze_state(state_vals: StateValues, assert_=DEBUG) -> bytes:
    # Serialize any value to a bytes object
    def serialize_val(val):
        if isinstance(val, StateValue):
            val = val.val
        if isinstance(val, int):
            return struct.pack("@i", val)
        elif isinstance(val, float):
            return struct.pack("@f", val)
        elif isinstance(val, str):
            return serialize_val(len(val)) + val.encode("ascii")
        else:
            raise ValueError(f"type(val)={type(val)} not handled")

    # Flatten the nested values into a single list of primitives
    def flatten_vals(vals, flat_list=None):
        if flat_list is None:
            flat_list = []
        if isinstance(vals, dict):
            for val in vals.values():
                flatten_vals(val, flat_list)
        elif isinstance(vals, list):
            for val in vals:
                flatten_vals(val, flat_list)
        else:
            flat_list.append(vals)

    # Flatten the values, then serialize
    flat_vals = []
    flatten_vals(state_vals, flat_vals)

    state_bytes = b"".join([serialize_val(val) for val in flat_vals])

    if assert_:
        assert (
            _parse_maze_state_bytes_handling_buffer_error(state_bytes, assert_=False) == state_vals
        ), "deserialize(serialize(state_vals)) != state_vals"
    return state_bytes





def get_maze_structure(data):
    positions = []
    for i, cell in enumerate(data):
        if 'i' in cell and cell['i'].val == 51:
            positions.append(i)
    return positions

def get_keys(state_values):
    positions = {}
    for ents in state_values["ents"]:
        if ents["image_type"].val== 2:
            if ents["x"].val > 1 or ents["y"].val > 1: 
                positions[ents["image_theme"].val] = {"x" :ents["x"].val, "y" :ents["y"].val}
    return positions




import random

def create_key_states(key_color_combinations, num_samples=5, num_levels=100):
    observations_list = [[] for _ in range(num_samples)]
    key_indices = {"blue": 0, "green": 1, "red": 2}
    sample_idx = 0
    while sample_idx < num_samples:
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)

        if not all(color in state.get_key_colors() for color in ['blue', 'green', 'red']):
            venv.close()
            continue
        


        for key_colors in key_color_combinations:
            state.remove_all_entities()
            full_grid = state.full_grid(with_mouse=False)
            entities = state.state_vals["ents"]
            legal_mouse_positions = get_legal_mouse_positions(full_grid, entities)

            for i, color in enumerate(key_colors):
                key_index = key_indices[color]
                x, y = legal_mouse_positions[random.randint(0, len(legal_mouse_positions) - 1)]
                state.set_key_position(key_index, x, y)

            full_grid = state.full_grid(with_mouse=False)
            entities = state.state_vals["ents"]
            legal_mouse_positions = get_legal_mouse_positions(full_grid, entities)
            player_pos = legal_mouse_positions[random.randint(0, len(legal_mouse_positions) - 1)]
            state.set_mouse_pos(*player_pos)
            state_bytes = state.state_bytes
            if state_bytes is not None:
                        # Add player to the environment
                venv.env.callmethod("set_state", [state_bytes])
                obs = venv.reset()
                observations_list[sample_idx].append(obs[0])

        venv.close()
        sample_idx += 1

    return observations_list

def create_gem_states(num_samples=5, num_levels=100):
    observations_list = []

    for _ in range(num_samples):
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)
        full_grid = state.full_grid(with_mouse=False)
        entities = state.state_vals["ents"]
        legal_mouse_positions = get_legal_mouse_positions(full_grid, entities)

        state.remove_all_entities()

        # Add player to the environment
        player_pos = legal_mouse_positions[random.randint(0, len(legal_mouse_positions) - 1)]
        state.set_mouse_pos(*player_pos)

        # Add gem to the environment
        gem_pos = legal_mouse_positions[random.randint(0, len(legal_mouse_positions) - 1)]
        state.set_gem_position(*gem_pos)

        state_bytes = state.state_bytes
        if state_bytes is not None:
            venv.env.callmethod("set_state", [state_bytes])
            obs = venv.reset()
            observations_list.append(obs[0])

        venv.close()

    return observations_list


def create_classified_dataset(num_samples_per_category=5, num_levels=0):
    dataset = {
        "gem": [],
        "blue_key": [],
        "green_key": [],
        "red_key": [],
        "blue_lock": [],
        "green_lock": [],
        "red_lock": []
    }

    key_indices = {"blue": 0, "green": 1, "red": 2}

    

    while any(len(samples) < num_samples_per_category for samples in dataset.values()):
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)
        key_colors = state.get_key_colors()

        if not key_colors:
            if len(dataset["gem"]) < num_samples_per_category:
                state_bytes = state.state_bytes
                if state_bytes is not None:
                    venv.env.callmethod("set_state", [state_bytes])
                    obs = venv.reset()
                    dataset["gem"].append(obs[0].transpose(1,2,0))
        else:
            if "red" in key_colors:
                if len(dataset["red_key"]) < num_samples_per_category:
                    state.delete_keys_and_locks(3)
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        obs = venv.reset()
                        dataset["red_key"].append(obs[0].transpose(1,2,0))
                if len(dataset["red_lock"]) < num_samples_per_category:
                    state.delete_keys()
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        obs = venv.reset()
                        dataset["red_lock"].append(obs[0].transpose(1,2,0))
            elif "green" in key_colors:
                if len(dataset["green_key"]) < num_samples_per_category:
                    state.delete_keys_and_locks(2)
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        obs = venv.reset()
                        dataset["green_key"].append(obs[0].transpose(1,2,0))
                if len(dataset["green_lock"]) < num_samples_per_category:
                    state.delete_keys()
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        obs = venv.reset()
                        dataset["green_lock"].append(obs[0].transpose(1,2,0))
            elif "blue" in key_colors:
                if len(dataset["blue_key"]) < num_samples_per_category:
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        obs = venv.reset()
                        dataset["blue_key"].append(obs[0].transpose(1,2,0))
                if len(dataset["blue_lock"]) < num_samples_per_category:
                    state.delete_keys()
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        obs = venv.reset()
                        dataset["blue_lock"].append(obs[0].transpose(1,2,0))

        venv.close()

    return dataset

def create_classified_dataset_venvs(num_samples_per_category=5, num_levels=0):
    dataset = {
        "gem": [],
        "blue_key": [],
        "green_key": [],
        "red_key": [],
        "blue_lock": [],
        "green_lock": [],
        "red_lock": []
    }

    key_indices = {"blue": 0, "green": 1, "red": 2}

    while any(len(samples) < num_samples_per_category for samples in dataset.values()):
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)
        key_colors = state.get_key_colors()

        if not key_colors:
            if len(dataset["gem"]) < num_samples_per_category:
                state_bytes = state.state_bytes
                if state_bytes is not None:
                    venv.env.callmethod("set_state", [state_bytes])
                    venv.reset()
                    dataset["gem"].append(venv)
                else:
                    venv.close()
        else:
            if "red" in key_colors:
                if len(dataset["red_key"]) < num_samples_per_category:
                    state.delete_keys_and_locks(3)  # Remove blue and green, keep red
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        venv.reset()
                        dataset["red_key"].append(venv)
                    else:
                        venv.close()
                elif len(dataset["red_lock"]) < num_samples_per_category:
                    state.delete_keys()
                    state.delete_specific_locks([key_indices["blue"], key_indices["green"]])  # Remove blue and green locks
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        venv.reset()
                        dataset["red_lock"].append(venv)
                    else:
                        venv.close()
                else:
                    venv.close()
            elif "green" in key_colors:
                if len(dataset["green_key"]) < num_samples_per_category:
                    state.delete_keys_and_locks(2)  # Remove blue, keep green
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        venv.reset()
                        dataset["green_key"].append(venv)
                    else:
                        venv.close()
                elif len(dataset["green_lock"]) < num_samples_per_category:
                    state.delete_keys()
                    state.delete_specific_locks([key_indices["blue"]])  # Remove blue locks
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        venv.reset()
                        dataset["green_lock"].append(venv)
                    else:
                        venv.close()
                else:
                    venv.close()
            elif "blue" in key_colors:
                if len(dataset["blue_key"]) < num_samples_per_category:
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        venv.reset()
                        dataset["blue_key"].append(venv)
                    else:
                        venv.close()
                elif len(dataset["blue_lock"]) < num_samples_per_category:
                    state.delete_keys()
                    state_bytes = state.state_bytes
                    if state_bytes is not None:
                        venv.env.callmethod("set_state", [state_bytes])
                        venv.reset()
                        dataset["blue_lock"].append(venv)
                    else:
                        venv.close()
                else:
                    venv.close()
            else:
                venv.close()

    return dataset

def create_empty_maze_dataset(num_samples_per_category=5, num_levels=0, keep_player=True):
    dataset = {
        "empty_maze": []
    }

    key_indices = {"blue": 0, "green": 1, "red": 2}

    

    while any(len(samples) < num_samples_per_category for samples in dataset.values()):
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)

        state.remove_gem()
        state.delete_keys()
        state.delete_locks()
        if not keep_player:
            state.remove_player()
        state_bytes = state.state_bytes
        if state_bytes is not None:
            venv.env.callmethod("set_state", [state_bytes])
            obs = venv.reset()
            dataset["empty_maze"].append(obs[0].transpose(1,2,0))


        venv.close()

    return dataset

def venv_with_all_mouse_positions(venv):
    """
    From a venv with a single env, create a new venv with one env for each legal mouse position.

    Returns venv_all, (legal_mouse_positions, inner_grid_without_mouse)
    Typically you'd call this with `venv_all, _ = venv_with_all_mouse_positions(venv)`,
    The extra return values are useful for conciseness sometimes.
    """
    assert (
        venv.num_envs == 1
    ), f"Did you forget to use copy_venv to get a single env?"

    sb_back = venv.env.callmethod("get_state")[0]
    env_state = EnvState(sb_back)

    grid = env_state.inner_grid(with_mouse=False)
    entities = env_state.state_vals["ents"]
    legal_mouse_positions = get_legal_mouse_positions(grid,entities)


    # convert coords from inner to outer grid coordinates
    padding = get_padding(grid)

    # create a venv for each legal mouse position
    state_bytes_list = []
    for mx, my in legal_mouse_positions:
        # we keep a backup of the state bytes for efficiency, as calling set_mouse_pos
        # implicitly calls _parse_state_bytes, which is slow. this is a hack.
        # NOTE: Object orientation hurts us here. It would be better to have functions.
        noisy_mx = mx + random.uniform(-0.2, 0.2)
        noisy_my = my + random.uniform(-0.2, 0.2)
        env_state.set_mouse_pos(noisy_mx, noisy_my)
        state_bytes_list.append(env_state.state_bytes)
        env_state.state_bytes = sb_back

    threads = 1 if len(legal_mouse_positions) < 100 else os.cpu_count()
    venv_all = create_venv(
        num=len(legal_mouse_positions),
        num_threads=threads,
        num_levels=1,
        start_level=1,
    )
    venv_all.env.callmethod("set_state", state_bytes_list)
    return venv_all, (legal_mouse_positions, grid)

def get_padding(grid: np.ndarray) -> int:
    """Return the padding of the (inner) grid, i.e. the number of walls around the maze."""
    return (WORLD_DIM - grid.shape[0]) // 2

def set_mouse_to_center(state):
    # Find the legal positions for the mouse
    full_grid = state.full_grid(with_mouse=False)
    entities = state.state_vals["ents"]
    legal_mouse_positions = get_legal_mouse_positions(full_grid, entities)

    if not legal_mouse_positions:
        return None

    # Calculate the middle point based on legal positions
    middle_x = sum(pos[0] for pos in legal_mouse_positions) / len(legal_mouse_positions)
    middle_y = sum(pos[1] for pos in legal_mouse_positions) / len(legal_mouse_positions)

    # Find the legal position closest to the middle point
    closest_position = min(legal_mouse_positions, key=lambda pos: (pos[0] - middle_x) ** 2 + (pos[1] - middle_y) ** 2)

    # Set the mouse position to the closest legal position
    state.set_mouse_pos(*closest_position)

    return state

def create_direction_dataset(num_samples_per_category=5, num_levels=100):
    dataset = {
        "top_left": [],
        "top_right": [],
        "bottom_left": [],
        "bottom_right": []
    }

    while any(len(samples) < num_samples_per_category for samples in dataset.values()):
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)

        # Set the mouse position to the center based on legal positions
        state.remove_all_entities()
        state = set_mouse_to_center(state)
        if state is None:
            venv.close()
            continue

        # Find the legal positions for the target
        full_grid = state.full_grid(with_mouse=False)
        entities = state.state_vals["ents"]
        legal_positions = get_legal_mouse_positions(full_grid, entities)

        if len(legal_positions) < 4:
            venv.close()
            continue

        # Find the farthest corners based on legal positions
        top_left_corner = min(legal_positions, key=lambda pos: pos[0] + pos[1])
        top_right_corner = max(legal_positions, key=lambda pos: pos[0] - pos[1])
        bottom_left_corner = max(legal_positions, key=lambda pos: -pos[0] + pos[1])
        bottom_right_corner = max(legal_positions, key=lambda pos: pos[0] + pos[1])

        # Generate samples for each corner position
        for corner_pos, category in zip(
            [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner],
            ["top_left", "top_right", "bottom_left", "bottom_right"]
        ):
            if len(dataset[category]) < num_samples_per_category:
                state.set_gem_position(*corner_pos)
                state_bytes = state.state_bytes
                if state_bytes is not None:
                    venv.env.callmethod("set_state", [state_bytes])
                    obs = venv.reset()
                    dataset[category].append(obs[0].transpose(1, 2, 0))

        venv.close()

    return dataset
