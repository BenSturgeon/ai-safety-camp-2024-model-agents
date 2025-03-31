#!/usr/bin/env python
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
from utils.environment_modification_experiments import create_specific_l_shaped_maze_env
from utils import helpers
from sae_cnn import ordered_layer_names


class InterventionExperiment:
    """Base class for intervention experiments on model activations"""
    
    def __init__(self, model_path, layer_number, device=None):
        """
        Initialize the experiment with model and target layer

        Args:
            model_path (str): Path to the model checkpoint
            layer_number (int): Layer number to target (corresponds to ordered_layer_names)
            device (torch.device): Device to run on (defaults to CUDA if available, else CPU)
        """
        # Set up device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = helpers.load_interpretable_model(model_path=model_path).to(self.device)
        
        # Set layer info
        self.layer_number = layer_number
        self.layer_name = ordered_layer_names[layer_number]
        print(f"Targeting layer: {self.layer_name} (layer {layer_number})")
        
        # Get the module for this layer
        self.module = self._get_module(self.model, self.layer_name)
        
        # Initialize buffers
        self.original_activations = []
        self.modified_activations = []
        self.intervention_active = False
        self.intervention_config = None
        self.dynamic_intervention = None
        
    def _get_module(self, model, layer_name):
        """Get a module from the model using its layer name"""
        module = model
        for element in layer_name.split("."):
            if "[" in element:
                base, idx = element.rstrip("]").split("[")
                module = getattr(module, base)[int(idx)]
            else:
                module = getattr(module, element)
        return module
    
    def _hook_activations(self, module, input, output):
        """
        Base hook method to capture and/or modify activations
        This should be overridden by derived classes
        """
        raise NotImplementedError("This method should be overridden by derived classes")

    def set_intervention(self, config):
        """
        Set up a static intervention configuration
        
        Args:
            config (list): List of dicts with intervention parameters:
                - channel (int): Channel to intervene on
                - position (tuple): (y, x) position to intervene at
                - value (float): Value to set at that position
                - radius (int, optional): Radius of intervention (default 0 for single point)
                - scale_factor (float, optional): Scale factor for radius-based interventions
        """
        self.intervention_config = config
        self.intervention_active = True
        self.dynamic_intervention = None
        print(f"Static intervention configured for {len(config)} points")
    
    def set_dynamic_intervention(self, intervention_function):
        """
        Set up a dynamic intervention that can change based on the agent's state
        
        Args:
            intervention_function (callable): Function that takes (step, original_activations, modified_activations)
                                             and returns modified activations
        """
        self.dynamic_intervention = intervention_function
        self.intervention_active = False
        self.intervention_config = None
        print("Dynamic intervention configured")
        
    def disable_intervention(self):
        """Disable any active interventions"""
        self.intervention_active = False
        self.intervention_config = None
        self.dynamic_intervention = None
        print("Interventions disabled")
    
    def run_maze_experiment(self, maze_variant=0, max_steps=100, save_gif=True, 
                           output_path="intervention_results", save_gif_freq=1,
                           max_gif_frames=0, entity1_code=4, entity2_code=None, use_box_maze=False):
        """
        Run an experiment with a specific maze variant.
        
        Args:
            maze_variant (int): Maze variant to use (0-7)
            max_steps (int): Maximum number of steps to run
            save_gif (bool): Whether to save a GIF of the experiment
            output_path (str): Directory to save results
            save_gif_freq (int): Save every Nth timestep in activation GIFs (default: 1 = all frames)
            max_gif_frames (int): Maximum number of frames to include in GIFs (0=all frames)
            entity1_code (int): Primary entity code to use in box maze (default: 4, blue key)
            entity2_code (int): Secondary entity code for box maze (optional)
            use_box_maze (bool): Whether to use box maze instead of L-shaped maze
            
        Returns:
            dict: Results of the experiment
        """
        # Create appropriate maze environment
        if use_box_maze:
            # Import box maze only when needed
            from utils.environment_modification_experiments import create_box_maze
            print(f"Creating box maze with entity codes: primary={entity1_code}, secondary={entity2_code}")
            observations, venv = create_box_maze(entity1=entity1_code, entity2=entity2_code)
        else:
            # Use L-shaped maze
            print(f"Creating L-shaped maze variant {maze_variant}")
            venv = create_specific_l_shaped_maze_env(maze_variant=maze_variant)
            observations = None  # We'll collect these in the loop
        
        # Reset buffers
        self.original_activations = []
        self.modified_activations = []
        
        # Register hook for activations
        handle = self.module.register_forward_hook(self._hook_activations)
        
        # Run the agent in the environment
        observation = venv.reset()
        frames = []
        
        # Initialize observations list if needed
        if not use_box_maze or observations is None:
            observations = []
        
        total_reward = 0
        steps = 0
        done = False
        
        # Store initial state
        if save_gif:
            frames.append(venv.render(mode="rgb_array"))
            
        # For L-shaped maze, we need to collect observations
        if not use_box_maze:
            observations.append(observation[0])
        
        intervention_type = "none"
        if self.intervention_active:
            intervention_type = "static"
        elif self.dynamic_intervention is not None:
            intervention_type = "dynamic"
        
        print(f"Running agent with {intervention_type} intervention for max {max_steps} steps")
        
        while not done and steps < max_steps:
            # Prepare observation for the model
            obs = observation[0]
            
            # For L-shaped maze, collect observations
            if not use_box_maze:
                observations.append(obs)
            
            # Convert observation to RGB
            converted_obs = helpers.observation_to_rgb(obs)
            obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(self.device)
            
            # Make sure obs_tensor has 4 dimensions (add batch dimension if needed)
            if obs_tensor.ndim == 3:
                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
            
            # Get model output with intervention (if active)
            with torch.no_grad():
                outputs = self.model(obs_tensor)
            
            # Get action from model output
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0].logits
                
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            action = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            
            # Take action in environment
            observation, reward, done, info = venv.step(action.cpu().numpy())
            
            # Record frame
            if save_gif:
                frames.append(venv.render(mode="rgb_array"))
                
            total_reward += reward
            steps += 1
            
        # Remove hook
        handle.remove()
        
        # Close environment
        venv.close()
        
        print(f"Episode complete: {steps} steps, reward {total_reward}")
        
        # Save results
        if save_gif and frames:
            os.makedirs(output_path, exist_ok=True)
            
            # Create descriptive filename
            if intervention_type == "static":
                channels_str = "_".join([str(cfg["channel"]) for cfg in self.intervention_config])
                filename = f"maze_{maze_variant}_static_ch{channels_str}"
            elif intervention_type == "dynamic":
                filename = f"maze_{maze_variant}_dynamic"
            else:
                filename = f"maze_{maze_variant}_baseline"
            
            # Add entity info to filename if using box maze
            if use_box_maze:
                entity_info = f"entity{entity1_code}"
                if entity2_code is not None:
                    entity_info += f"_entity{entity2_code}"
                filename = f"{filename}_{entity_info}"
            
            gif_path = f"{output_path}/{filename}.gif"
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"Saved GIF to {gif_path}")
            
            # Save visualization of activations
            if len(self.original_activations) > 0:
                self._visualize_activations(output_path, maze_variant, intervention_type)
                # Also create GIFs of activation changes over time
                self._create_activation_gifs(output_path, maze_variant, intervention_type, 
                                            frames=frames, save_gif_freq=save_gif_freq,
                                            max_gif_frames=max_gif_frames)
        
        results = {
            "maze_variant": maze_variant,
            "total_steps": steps,
            "total_reward": total_reward,
            "intervention_type": intervention_type,
            "intervention_config": self.intervention_config,
            "output_path": output_path
        }
        
        # Add entity info to results if using box maze
        if use_box_maze:
            results["entity1_code"] = entity1_code
            results["entity2_code"] = entity2_code
            results["use_box_maze"] = True
        
        return results
    
    def _visualize_activations(self, output_path, maze_variant, intervention_type):
        """
        Create visualization of activations for the first frame
        Should be implemented by derived classes
        """
        raise NotImplementedError("This method should be overridden by derived classes")
            
    def _create_activation_gifs(self, output_path, maze_variant, intervention_type, 
                               frames=None, save_gif_freq=1, max_gif_frames=0):
        """
        Create GIFs of activation changes over time for specific channels
        Should be implemented by derived classes
        """
        raise NotImplementedError("This method should be overridden by derived classes")


def create_trajectory_based_intervention(channel, start_pos, target_pos, max_steps, value=5.0, radius=1):
    """
    Create a trajectory-based intervention function that moves activation along a path
    
    Args:
        channel (int): Channel to intervene on
        start_pos (tuple): Starting position (y, x)
        target_pos (tuple): Target position (y, x)
        max_steps (int): Steps to complete trajectory
        value (float): Value to set at intervention point
        radius (int): Radius of intervention
        
    Returns:
        function: Intervention function that can be passed to set_dynamic_intervention
    """
    start_y, start_x = start_pos
    target_y, target_x = target_pos
    
    # Calculate trajectory points
    y_steps = np.linspace(start_y, target_y, max_steps)
    x_steps = np.linspace(start_x, target_x, max_steps)
    
    # Store trajectory points with interpolation
    trajectory = [(int(round(y)), int(round(x))) for y, x in zip(y_steps, x_steps)]
    
    def intervention_function(step, original_activations, modified_activations):
        # Calculate current position along trajectory
        if step >= len(trajectory):
            # We've reached the end of the trajectory, stay at final position
            current_pos = trajectory[-1]
        else:
            current_pos = trajectory[step]
        
        # Apply the intervention at the current position
        y, x = current_pos
        
        # Single point modification
        if radius == 0:
            modified_activations[0, channel, y, x] = value
        # Radius-based modification
        else:
            h, w = modified_activations.shape[2], modified_activations.shape[3]
            
            # Create a mask for the circular region
            y_indices = torch.arange(max(0, y-radius), min(h, y+radius+1), device=modified_activations.device)
            x_indices = torch.arange(max(0, x-radius), min(w, x+radius+1), device=modified_activations.device)
            
            # Calculate distances
            Y, X = torch.meshgrid(y_indices, x_indices, indexing='ij')
            distances = torch.sqrt((Y - y)**2 + (X - x)**2)
            
            # Apply modification within radius
            mask = distances <= radius
            for i in range(len(y_indices)):
                for j in range(len(x_indices)):
                    if mask[i, j]:
                        current_y, current_x = y_indices[i], x_indices[j]
                        modified_activations[0, channel, current_y, current_x] = value
        
        return modified_activations
        
    return intervention_function 