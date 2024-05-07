#!/usr/bin/env python
# coding: utf-8
#This file acts as a means to decode and edit the underlying state of the heist environment for the purposes of our interpretability work.

# The code is based off Monte's code in procgen tools with modifications to account for the differences in environment.


import gym
import random
import numpy as np
from helpers import generate_action
from procgen import ProcgenGym3Env
import imageio
import matplotlib.pyplot as plt
import typing
import math

from procgen import ProcgenGym3Env
import struct
import typing
from typing import Tuple, Dict, Callable, List, Optional
from dataclasses import dataclass
from src.policies_modified import ImpalaCNN
from procgen_tools.procgen_wrappers import VecExtractDictObs, TransposeFrame, ScaledFloatFrame

from gym3 import ToBaselinesVecEnv



# Constants
KEY_COLORS = {0: 'blue',1: 'green',2: 'red',}
MOUSE = 0
KEY = 2
LOCKED_DOOR = 1
WORLD_DIM = 25
EMPTY = 100
BLOCKED = 51

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
    ["loop", "has_keys", "num_keys", [["int", "key_state"]]],
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





def load_model( model_path = '../model_1501.0_interpretable.pt'):
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
    num: int, start_level: int, num_levels: int, num_threads: int = 1
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
        distribution_mode="easy",
        num_threads=num_threads,
        render_mode="rgb_array",
    )
    venv = wrap_venv(venv)
    return venv
    


model = load_model()



env = ProcgenGym3Env(num=1, env_name="heist", render_mode="rgb_array")
state = env.callmethod("get_state")[0]


modified_state = bytearray(state)





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


class EnvState:
    def __init__(self, state_bytes: bytes):
        self.state_bytes = state_bytes

    @property
    def state_vals(self):
        return _parse_maze_state_bytes(self.state_bytes)

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


    def set_key_position(self, key_index, x, y):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val== 2:
                if key_index == ents["image_theme"].val:
                    ents["x"].val = float(y) + .5
                    ents["y"].val = float(x) + .5
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
    
    def remove_all_entities(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            ents["x"].val = -1
            ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)
    
    def remove_gem(self):
        state_values = self.state_vals  
        for ents in state_values["ents"]:
            if ents["image_type"].val== 9:
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
    # Functions to read values of different types
    def read_fixed(sb, idx, fmt):
        sz = struct.calcsize(fmt)
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
    def flatten_vals(vals, flat_list=[]):
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
            _parse_maze_state_bytes(state_bytes, assert_=False) == state_vals
        ), "deserialize(serialize(state_vals)) != state_vals"
    return state_bytes





def get_maze_structure(data):
    positions = []
    for i, cell in enumerate(data):
        if 'i' in cell and cell['i'].val == 51:
            positions.append(i)
    return positions

def get_keys(state_values):
    positions = []
    for ents in state_values["ents"]:
        if ents["image_type"].val== 2:
            print(ents)
            positions.append({"x" :ents["x"].val, "y" :ents["y"].val, "alpha" : ents["alpha"].val})
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

# def create_classified_dataset(num_samples_per_category=5, num_levels=100):
#     dataset = {
#         "gem": [],
#         "blue_key": [],
#         "green_key": [],
#         "red_key": []
#     }

#     while any(len(samples) < num_samples_per_category for samples in dataset.values()):
#         venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
#         state = state_from_venv(venv, 0)
#         key_colors = state.get_key_colors()

#         if not key_colors:
#             if len(dataset["gem"]) < num_samples_per_category:
#                 obs = venv.reset()
#                 dataset["gem"].append(obs[0].transpose(1,2,0))
#         else:
#             for color in key_colors:
#                 if len(dataset[f"{color}_key"]) < num_samples_per_category:
#                     if color == "blue":
#                         state.delete_specific_keys_and_locks([])
#                     elif color == "green":
#                         state.delete_specific_keys_and_locks([0])
#                     elif color == "red":
#                         state.delete_specific_keys_and_locks([0, 1])

#                     state_bytes = state.state_bytes
#                     if state_bytes is not None:
#                         venv.env.callmethod("set_state", [state_bytes])
#                         obs = venv.reset()
#                         dataset[f"{color}_key"].append(obs[0].transpose(1,2,0))

#         venv.close()

#     return dataset

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

    def delete_keys(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2:  # Check if the entity is a key
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_keys_and_locks(self, stage):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val in [1, 2]:  # Check if the entity is a key or lock
                if stage == 2 and ents["image_theme"].val == key_indices["blue"]:
                    ents["x"].val = -1
                    ents["y"].val = -1
                elif stage == 3 and ents["image_theme"].val in [key_indices["blue"], key_indices["green"]]:
                    ents["x"].val = -1
                    ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    EnvState.delete_keys = delete_keys
    EnvState.delete_keys_and_locks = delete_keys_and_locks

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