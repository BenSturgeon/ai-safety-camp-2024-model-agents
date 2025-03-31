#!/usr/bin/env python
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
from utils.environment_modification_experiments import create_box_maze
from utils import helpers


class BaseModelInterventionExperiment:
    """Experiment for intervening on base model activations."""
    
    def __init__(self, model_path, target_layer, device=None):
        """
        Initialize the experiment with model and target layer

        Args:
            model_path (str): Path to the model checkpoint
            target_layer (str): Name of the layer to target (e.g., 'conv3a')
            device (torch.device): Device to run on (defaults to CUDA if available, else CPU)
        """
        # Set up device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = helpers.load_interpretable_model(model_path=model_path).to(self.device)
        self.model.eval()
        
        # Set layer info
        self.layer_name = target_layer
        print(f"Targeting layer: {self.layer_name}")
        
        # Get the module for this layer
        self.module = self._get_module(self.layer_name)
        
        # Initialize buffers
        self.original_activations = []
        self.modified_activations = []
        self.intervention_active = False
        self.intervention_config = None
        self.dynamic_intervention = None
        
    def _get_module(self, layer_name):
        """Get a module from the model using its layer name"""
        return dict(self.model.named_modules())[layer_name]
    
    def _hook_activations(self, module, input, output):
        """Hook to capture and/or modify activations"""
        with torch.no_grad():
            # Store original activations
            orig_acts = output.detach().clone()
            self.original_activations.append(orig_acts.squeeze(0).cpu())
            
            # Apply static intervention if active
            if self.intervention_active and self.intervention_config:
                modified_acts = output.clone()  # Start with a copy of the original activations
                
                # Apply the modification at specific positions
                for config in self.intervention_config:
                    channel = config["channel"]
                    position = config["position"]
                    value = config["value"]
                    scale_factor = config.get("scale_factor", 1.0)
                    radius = config.get("radius", 0)
                    
                    # Get batch size (though usually it's 1)
                    batch_size = modified_acts.shape[0]
                    
                    # For each item in batch (usually just one)
                    for b in range(batch_size):
                        # Single point modification
                        if radius == 0:
                            modified_acts[b, channel, position[0], position[1]] = value
                        # Radius-based modification
                        else:
                            h, w = modified_acts.shape[2], modified_acts.shape[3]
                            y, x = position
                            
                            # Create a mask for the circular region
                            y_indices = torch.arange(max(0, y-radius), min(h, y+radius+1), device=self.device)
                            x_indices = torch.arange(max(0, x-radius), min(w, x+radius+1), device=self.device)
                            
                            # Calculate distances
                            Y, X = torch.meshgrid(y_indices, x_indices, indexing='ij')
                            distances = torch.sqrt((Y - y)**2 + (X - x)**2)
                            
                            # Apply modification within radius
                            mask = distances <= radius
                            for i in range(len(y_indices)):
                                for j in range(len(x_indices)):
                                    if mask[i, j]:
                                        current_y, current_x = y_indices[i], x_indices[j]
                                        # Scale value by distance from center
                                        if scale_factor != 1.0:
                                            distance_ratio = 1.0 - (distances[i, j] / radius)
                                            scaled_value = value * (scale_factor ** distance_ratio)
                                        else:
                                            scaled_value = value
                                        modified_acts[b, channel, current_y, current_x] = scaled_value
                
                # Store modified activations
                self.modified_activations.append(modified_acts.squeeze(0).cpu())
                
                # Return modified activations to affect the model's behavior
                return modified_acts
            
            # Apply dynamic intervention if provided
            elif self.dynamic_intervention is not None:
                step = len(self.original_activations) - 1  # Current step in the episode
                modified_acts = output.clone()
                
                # Call the dynamic intervention function with current step and activations
                modified_acts = self.dynamic_intervention(
                    step=step, 
                    original_activations=output, 
                    modified_activations=modified_acts
                )
                
                # Store modified activations
                self.modified_activations.append(modified_acts.squeeze(0).cpu())
                
                # Return modified activations
                return modified_acts
            
            # If no intervention, just append empty modified activation and return original
            self.modified_activations.append(orig_acts.squeeze(0).cpu())
            return output

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
                           output_path="base_model_intervention_results", save_gif_freq=1,
                           max_gif_frames=0, entity1_code=4, entity2_code=None):
        """
        Run an experiment with a specific maze variant.
        
        Args:
            maze_variant (int): Maze variant to use (0-7)
            max_steps (int): Maximum number of steps to run
            save_gif (bool): Whether to save a GIF of the experiment
            output_path (str): Directory to save results
            save_gif_freq (int): Save every Nth timestep in activation GIFs (default: 1 = all frames)
            max_gif_frames (int): Maximum number of frames to include in GIFs (0=all frames)
            entity1_code (int): Code of primary entity to track (default: 4, blue key)
            entity2_code (int): Code of secondary entity (optional)
            
        Returns:
            dict: Results of the experiment
        """
        # Mapping of entity codes to descriptions
        entity_code_description = {
            3: "gem",
            4: "blue_key",
            5: "green_key",
            6: "red_key",
            7: "blue_lock",
            8: "green_lock",
            9: "red_lock"
        }
        
        # Get entity description for the folder name
        entity_desc = entity_code_description.get(entity1_code, f"entity_{entity1_code}")
        
        # Create directory structure: intervention_results/[layer_name]/entity_[code]_[description]
        layer_specific_output = f"{output_path}/{self.layer_name}"
        
        # Create entity-specific subfolder
        entity_specific_output = f"{layer_specific_output}/entity_{entity1_code}_{entity_desc}"
        
        # Create output directory
        os.makedirs(entity_specific_output, exist_ok=True)
        
        print(f"Saving results to: {os.path.abspath(entity_specific_output)}")
        
        # Create maze environment with specified entities
        print(f"Creating maze variant {maze_variant} with entity1_code={entity1_code}, entity2_code={entity2_code}")
        obs, venv = create_box_maze(entity1=entity1_code, entity2=entity2_code)
        
        # Reset buffers
        self.original_activations = []
        self.modified_activations = []
        
        # Register hook for activations
        handle = self.module.register_forward_hook(self._hook_activations)
        
        # Run the agent in the environment
        observation = venv.reset()
        frames = []
        observations = []
        
        total_reward = 0
        steps = 0
        done = False
        
        # Store initial state
        if save_gif:
            frames.append(venv.render(mode="rgb_array"))
        observations.append(observation[0])
        
        # Determine intervention type for reporting
        intervention_type = "none"
        if self.intervention_active:
            intervention_type = "static"
        elif self.dynamic_intervention is not None:
            intervention_type = "dynamic"
        
        print(f"Running agent with {intervention_type} intervention for max {max_steps} steps")
        
        while not done and steps < max_steps:
            # Prepare observation for the model
            obs = observation[0]
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
            os.makedirs(entity_specific_output, exist_ok=True)
            
            # Create descriptive filename
            if intervention_type == "static":
                channels_str = "_".join([str(cfg["channel"]) for cfg in self.intervention_config])
                filename = f"maze_{maze_variant}_static_ch{channels_str}"
            elif intervention_type == "dynamic":
                filename = f"maze_{maze_variant}_dynamic"
            else:
                filename = f"maze_{maze_variant}_baseline"
            
            gif_path = f"{entity_specific_output}/{filename}.gif"
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"Saved GIF to {gif_path}")
            
            # Save visualization of activations
            if len(self.original_activations) > 0:
                self._visualize_activations(entity_specific_output, maze_variant, intervention_type)
                # Also create GIFs of activation changes over time
                self._create_activation_gifs(entity_specific_output, maze_variant, intervention_type, 
                                            frames=frames, save_gif_freq=save_gif_freq,
                                            max_gif_frames=max_gif_frames)
        
        # Return results summary
        results = {
            "maze_variant": maze_variant,
            "total_steps": steps,
            "total_reward": total_reward,
            "intervention_type": intervention_type,
            "intervention_config": self.intervention_config,
            "entity1_code": entity1_code,
            "entity2_code": entity2_code,
            "output_path": entity_specific_output,
            "layer_name": self.layer_name
        }
        
        return results
    
    def _visualize_activations(self, output_path, maze_variant, intervention_type):
        """Create visualization of activations for the first frame"""
        if not self.original_activations:
            return
            
        # Create directory for activation visualizations
        vis_dir = f"{output_path}/activations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Determine which channels to visualize
        channels = []
        if intervention_type == "static" and self.intervention_config:
            channels = [cfg["channel"] for cfg in self.intervention_config]
        else:
            # For baseline or dynamic, just show some channels with highest mean activation
            first_activation = self.original_activations[0]
            # Calculate mean across spatial dimensions for each channel
            channel_means = first_activation.mean(dim=(1, 2))
            # Get top channels by mean activation
            top_channels = torch.argsort(channel_means, descending=True)[:5]
            channels = top_channels.tolist()
        
        # Create figure for the first frame's activations
        for channel in channels:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original activation
            orig_act = self.original_activations[0][channel].numpy()
            im0 = axs[0].imshow(orig_act, cmap='viridis')
            axs[0].set_title(f"Original Channel {channel}")
            fig.colorbar(im0, ax=axs[0])
            
            # Modified activation (if available)
            if len(self.modified_activations) > 0:
                mod_act = self.modified_activations[0][channel].numpy()
                im1 = axs[1].imshow(mod_act, cmap='viridis')
                axs[1].set_title(f"Modified Channel {channel}")
                fig.colorbar(im1, ax=axs[1])
            else:
                axs[1].imshow(orig_act, cmap='viridis', alpha=0.3)
                axs[1].set_title("No modification")
            
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/channel_{channel}_frame0.png", dpi=150)
            plt.close(fig)
    
    def _create_activation_gifs(self, output_path, maze_variant, intervention_type, 
                                frames=None, save_gif_freq=1, max_gif_frames=0):
        """Create GIFs showing activation patterns over time"""
        if not self.original_activations:
            return
            
        # Create directory for activation GIFs
        gif_dir = f"{output_path}/activation_gifs"
        os.makedirs(gif_dir, exist_ok=True)
        
        # Determine which channels to visualize
        channels = []
        if intervention_type == "static" and self.intervention_config:
            channels = [cfg["channel"] for cfg in self.intervention_config]
        else:
            # For baseline or dynamic, show channels with highest variance over time
            all_activations = torch.stack(self.original_activations)  # shape: [frames, channels, height, width]
            
            # Calculate mean activation for each channel across time
            channel_means = all_activations.mean(dim=(0, 2, 3))
            
            # Calculate variance of mean activation over time for each channel
            channel_vars = []
            for c in range(all_activations.shape[1]):
                channel_over_time = all_activations[:, c].mean(dim=(1, 2))  # Mean activation for channel c at each timestep
                channel_vars.append(channel_over_time.var().item())
            
            # Get top channels by variance
            top_channels = np.argsort(channel_vars)[-5:]  # Top 5 channels by variance
            channels = top_channels.tolist()
        
        # Number of frames to process
        n_frames = len(self.original_activations)
        if max_gif_frames > 0 and max_gif_frames < n_frames:
            # Subsample frames evenly
            indices = np.linspace(0, n_frames-1, max_gif_frames, dtype=int)
        else:
            indices = range(0, n_frames, save_gif_freq)
        
        # Create GIF for each selected channel
        for channel in channels:
            channel_frames = []
            
            for i in indices:
                if i >= len(self.original_activations):
                    continue
                    
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original activation
                orig_act = self.original_activations[i][channel].numpy()
                im0 = axs[0].imshow(orig_act, cmap='viridis')
                axs[0].set_title(f"Original Ch {channel} (Frame {i})")
                fig.colorbar(im0, ax=axs[0])
                
                # Modified activation (if available)
                if i < len(self.modified_activations):
                    mod_act = self.modified_activations[i][channel].numpy()
                    im1 = axs[1].imshow(mod_act, cmap='viridis')
                    axs[1].set_title(f"Modified Ch {channel} (Frame {i})")
                    fig.colorbar(im1, ax=axs[1])
                else:
                    axs[1].imshow(orig_act, cmap='viridis', alpha=0.3)
                    axs[1].set_title("No modification")
                
                plt.tight_layout()
                
                # Convert figure to image
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                channel_frames.append(image)
                plt.close(fig)
            
            if channel_frames:
                # Save as GIF
                filename = f"channel_{channel}_{intervention_type}"
                gif_path = f"{gif_dir}/{filename}.gif"
                imageio.mimsave(gif_path, channel_frames, fps=5)
                print(f"Saved activation GIF for channel {channel} to {gif_path}")
    
    def analyze_top_channels(self, output_path, top_n=10):
        """Analyze and save information about top channels"""
        if not self.original_activations:
            return
            
        # Create directory for analysis
        analysis_dir = f"{output_path}/analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Stack all activations over time
        all_activations = torch.stack(self.original_activations)  # shape: [frames, channels, height, width]
        n_frames, n_channels = all_activations.shape[0], all_activations.shape[1]
        
        # Calculate various metrics for each channel
        channel_stats = []
        
        for c in range(n_channels):
            # Get all activations for this channel
            channel_acts = all_activations[:, c]  # shape: [frames, height, width]
            
            # Calculate mean activation over time and space
            mean_activation = channel_acts.mean().item()
            
            # Calculate variance of mean activation over time
            mean_over_time = channel_acts.mean(dim=(1, 2))  # Mean activation at each timestep
            variance_over_time = mean_over_time.var().item()
            
            # Calculate spatial variance (average variance across spatial dimensions)
            spatial_variance = channel_acts.var(dim=(1, 2)).mean().item()
            
            channel_stats.append({
                "channel": c,
                "mean_activation": mean_activation,
                "variance_over_time": variance_over_time,
                "spatial_variance": spatial_variance
            })
        
        # Sort channels by variance over time (most dynamic channels)
        dynamic_channels = sorted(channel_stats, key=lambda x: x["variance_over_time"], reverse=True)[:top_n]
        
        # Sort channels by spatial variance (channels with most spatial structure)
        spatial_channels = sorted(channel_stats, key=lambda x: x["spatial_variance"], reverse=True)[:top_n]
        
        # Sort channels by mean activation (most active channels)
        active_channels = sorted(channel_stats, key=lambda x: x["mean_activation"], reverse=True)[:top_n]
        
        # Save analysis to file
        with open(f"{analysis_dir}/channel_analysis.txt", "w") as f:
            f.write("BASE MODEL CHANNEL ANALYSIS\n")
            f.write("=========================\n\n")
            
            f.write("Most Dynamic Channels (highest variance over time):\n")
            for i, stats in enumerate(dynamic_channels):
                f.write(f"{i+1}. Channel {stats['channel']}: variance = {stats['variance_over_time']:.6f}\n")
            f.write("\n")
            
            f.write("Channels with Most Spatial Structure:\n")
            for i, stats in enumerate(spatial_channels):
                f.write(f"{i+1}. Channel {stats['channel']}: spatial variance = {stats['spatial_variance']:.6f}\n")
            f.write("\n")
            
            f.write("Most Active Channels:\n")
            for i, stats in enumerate(active_channels):
                f.write(f"{i+1}. Channel {stats['channel']}: mean activation = {stats['mean_activation']:.6f}\n")
            f.write("\n")
            
            # Provide some example intervention commands
            f.write("Example Intervention Commands:\n")
            
            if dynamic_channels:
                channel = dynamic_channels[0]["channel"]
                f.write(f"python run_base_model_intervention.py --static --channel {channel} --position 3,3 --value 5.0 --layer_name {self.layer_name}\n\n")
            
            if len(dynamic_channels) >= 3:
                channels = [str(stats["channel"]) for stats in dynamic_channels[:3]]
                f.write(f"Multiple channels intervention:\n")
                f.write(f"python run_base_model_intervention.py --static --channel {','.join(channels)} --position 3,3;4,4;2,2 --value 5.0 --layer_name {self.layer_name}\n")
        
        print(f"Saved channel analysis to {analysis_dir}/channel_analysis.txt")
        return dynamic_channels


def create_trajectory_based_intervention(channel, start_pos, target_pos, max_steps, value=5.0, radius=1):
    """
    Create a dynamic intervention function that moves a high activation region along a trajectory.
    
    Args:
        channel (int): Channel to intervene on
        start_pos (tuple): Starting position (y, x)
        target_pos (tuple): Target position (y, x)
        max_steps (int): Number of steps to complete the trajectory
        value (float): Value to set at the intervention point
        radius (int): Radius of the intervention
        
    Returns:
        function: Dynamic intervention function
    """
    start_y, start_x = start_pos
    target_y, target_x = target_pos
    
    # Precompute trajectory
    y_positions = np.linspace(start_y, target_y, max_steps)
    x_positions = np.linspace(start_x, target_x, max_steps)
    
    def intervention_function(step, original_activations, modified_activations):
        """
        Apply the intervention at the current position along the trajectory.
        
        Args:
            step (int): Current step in the episode
            original_activations (Tensor): Original model activations
            modified_activations (Tensor): Activations to modify
            
        Returns:
            Tensor: Modified activations
        """
        # Determine current position along trajectory
        if step < max_steps:
            current_y = y_positions[step]
            current_x = x_positions[step]
        else:
            current_y = target_y
            current_x = target_x
        
        # Convert to integer positions
        y = int(round(current_y))
        x = int(round(current_x))
        
        # Apply modification within the radius
        h, w = modified_activations.shape[2], modified_activations.shape[3]
        
        # Create a mask for the circular region
        for b in range(modified_activations.shape[0]):  # For each item in batch
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    # Check if within radius and within bounds
                    if dy*dy + dx*dx <= radius*radius:
                        cur_y, cur_x = y + dy, x + dx
                        if 0 <= cur_y < h and 0 <= cur_x < w:
                            modified_activations[b, channel, cur_y, cur_x] = value
        
        return modified_activations
    
    return intervention_function


# Main function will be implemented in run_base_model_intervention.py
if __name__ == "__main__":
    print("This module provides the BaseModelInterventionExperiment class.")
    print("Please use run_base_model_intervention.py to run experiments.")
