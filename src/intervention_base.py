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
    
    def run_maze_experiment(self, venv, max_steps=100, save_gif=True, 
                           output_path="intervention_results", save_gif_freq=1,
                           max_gif_frames=0):
        """
        Run an experiment with a given environment.
        
        Args:
            venv (ProcgenEnv): The pre-initialized Procgen environment.
            max_steps (int): Maximum number of steps to run
            save_gif (bool): Whether to save a GIF of the experiment
            output_path (str): Directory to save results
            save_gif_freq (int): Save every Nth timestep in activation GIFs (default: 1 = all frames)
            max_gif_frames (int): Maximum number of frames to include in GIFs (0=all frames)
            
        Returns:
            dict: Results of the experiment
        """
        # Environment is now passed in, no need to create it here.
        # Ensure output path exists
        os.makedirs(output_path, exist_ok=True)

        # Reset buffers
        self.original_activations = []
        self.modified_activations = []
        
        # Register hook for activations
        handle = self.module.register_forward_hook(self._hook_activations)
        
        # Run the agent in the environment
        observation = venv.reset() # Reset the provided environment
        frames = []
        
        # Initialize observations list (may not be needed if handled by caller)
        observations = []
        
        total_reward = 0
        steps = 0
        done = False
        
        # Store initial state
        if save_gif:
            frames.append(venv.render(mode="rgb_array"))
            
        # We probably always want the first observation
        observations.append(observation[0])
        
        intervention_type = "none"
        if self.intervention_active:
            intervention_type = "static"
        elif self.dynamic_intervention is not None:
            intervention_type = "dynamic"
        
        print(f"Running agent with {intervention_type} intervention for max {max_steps} steps")
        
        try:
            while not done and steps < max_steps:

                if isinstance(observation, tuple):
                    obs = observation[0]
                elif isinstance(observation, dict): # 
                    obs = observation['rgb'] 
                elif isinstance(observation, np.ndarray):
                    obs = observation
                else:
                    raise TypeError(f"Unexpected observation type: {type(observation)}")
                
                observations.append(obs)
                
                converted_obs = helpers.observation_to_rgb(obs)
                obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(self.device)
                
                if obs_tensor.ndim == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)  
                with torch.no_grad():
                    outputs = self.model(obs_tensor)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0].logits
                    
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                action = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
                
                observation, reward, done, info = venv.step(action.cpu().numpy())
                
                if save_gif:
                    frames.append(venv.render(mode="rgb_array"))
                    
                total_reward += reward
                steps += 1
        finally:

            handle.remove()
        
        print(f"Episode complete: {steps} steps, reward {total_reward}")
        
        # Save results
        if save_gif and frames:

            if intervention_type == "static":
                channels_str = "_".join([str(cfg["channel"]) for cfg in self.intervention_config])
                filename = f"static_intervention_ch{channels_str}"
            elif intervention_type == "dynamic":
                filename = f"dynamic_intervention"
            else:
                filename = f"baseline_run"
            
            gif_path = os.path.join(output_path, f"{filename}.gif")
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"Saved GIF to {gif_path}")
            

            if len(self.original_activations) > 0:
                
                maze_placeholder = "box" if "entity_" in output_path else "default" 
                self._visualize_activations(output_path, maze_placeholder, intervention_type)
                self._create_activation_gifs(output_path, maze_placeholder, intervention_type, 
                                            frames=frames, save_gif_freq=save_gif_freq,
                                            max_gif_frames=max_gif_frames)
        
        results = {
            "total_steps": steps,
            "total_reward": total_reward,
            "intervention_type": intervention_type,
            "intervention_config": self.intervention_config,
            "output_path": output_path
        }
        
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