from src.utils import heist
from src.utils import helpers
import torch

# import random
# from helpers import generate_action, load_model
# import imageio
# import typing
# import math
# import struct
# import typing
# from typing import Tuple, Dict, Callable, List, Optional
# from dataclasses import dataclass

# from gym3 import ToBaselinesVecEnv
# import random


from src.utils.steering_experiments import run_entity_steering_experiment

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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


# model_path = "../model_interpretable.pt"
# modification_value = -2
# total_episodes = 200
# objectives = {}

# entity = ["key", "lock", "gem"]
# entity_colors = ["blue", "red", "green"]
layers_to_test = range(1, 14)

# Read the objectives object from the pickle file
with open("objectives.pkl", "rb") as f:
    objectives = pickle.load(f)

metrics = ["avg_total_reward", "avg_steps_until_pickup", "total_count_pickups"]
entity_combinations = [
    f"{e}{c}" for e in ["key", "lock"] for c in ["blue", "red", "green"]
] + ["gem"]

for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    data = []
    for layer in layers_to_test:
        row = []
        for entity in entity_combinations:
            if entity == "gem":
                value = objectives[layer]["gem"][metric]
            else:
                value = objectives[layer][entity][metric]

            # Convert to float if it's a numpy array or other non-float type
            if isinstance(value, np.ndarray):
                value = float(value.item())
            elif not isinstance(value, (int, float)):
                value = float(value)

            row.append(value)
        data.append(row)

    df = pd.DataFrame(data, columns=entity_combinations, index=list(layers_to_test))

    # Rename entities to capture colour and lock or key
    new_entity_names = [
        "Blue Key",
        "Red Key",
        "Green Key",
        "Blue Lock",
        "Red Lock",
        "Green Lock",
        "Gem",
    ]
    df.columns = new_entity_names

    sns.heatmap(df, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={"label": metric})
    plt.title(
        f"Entity Steering Results Across Layers: {metric.replace('_', ' ').title()}"
    )
    plt.xlabel("Entity")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(f"entity_steering_heatmap_{metric}.png")
    plt.close()


metrics = ["avg_total_reward", "avg_steps_until_pickup", "total_count_pickups"]
titles = [
    "Average Total Reward",
    "Average Steps Until Pickup",
    "Total Count of Pickups",
]
all_entities = [
    "keyblue",
    "keyred",
    "keygreen",
    "lockblue",
    "lockred",
    "lockgreen",
    "gem",
]

for i, metric in enumerate(metrics):
    plt.figure(figsize=(12, 8))

    for entity in all_entities:
        values = [objectives[layer][entity][metric] for layer in layers_to_test]

        # Convert numpy array to float if necessary
        values = [
            float(v.item()) if isinstance(v, np.ndarray) else float(v) for v in values
        ]

        # Create a more readable label
        if entity == "gem":
            label = "Gem"
        else:
            entity_type = "Key" if entity.startswith("key") else "Lock"
            color = entity[3:] if entity.startswith("key") else entity[4:]
            label = f"{color.capitalize()} {entity_type}"

        # Convert values to percentages
        max_value = max(values)
        percentage_values = [(v / (max_value + 0.0001)) * 100 for v in values]

        plt.plot(list(layers_to_test), percentage_values, label=label, marker="o")

    plt.title(titles[i])
    plt.xlabel("Layer Number")
    plt.ylabel(f'{metric.replace("_", " ").title()} (% of Max)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"entity_steering_{metric}_percentage.png")
    plt.close()
