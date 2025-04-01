import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
from utils.environment_modification_experiments import create_specific_l_shaped_maze_env, create_box_maze
from utils import helpers
from sae_cnn import load_sae_from_checkpoint, ordered_layer_names


class SAEInterventionExperiment:
    def __init__(self, model_path, sae_checkpoint_path, layer_number, device=None):
        """
        Initialize the experiment with model, SAE, and target layer

        Args:
            model_path (str): Path to the model checkpoint
            sae_checkpoint_path (str): Path to the SAE checkpoint
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
        
        # Load SAE
        assert os.path.exists(sae_checkpoint_path), f"SAE checkpoint path not found: {sae_checkpoint_path}"
        print(f"Loading SAE from {sae_checkpoint_path}")
        self.sae = load_sae_from_checkpoint(sae_checkpoint_path).to(self.device)
        
        # Get the module for this layer
        self.module = self._get_module(self.model, self.layer_name)
        
        # Initialize buffers
        self.sae_activations = []
        self.modified_sae_activations = []
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
    
    def _hook_sae_activations(self, module, input, output):
        """Hook to capture and/or modify SAE activations"""
        with torch.no_grad():
            output = output.to(self.device)
            self.sae.to(self.device)
            
            # Get original SAE activations
            _, _, acts, _ = self.sae(output)
            
            # Store original activations
            orig_acts = acts.squeeze().cpu()
            self.sae_activations.append(orig_acts.clone())
            
            # Apply static intervention if active
            if self.intervention_active and self.intervention_config:
                modified_acts = acts.clone()  # Start with a copy of the original activations
                
                # Apply the modification at specific positions
                for config in self.intervention_config:
                    channel = config["channel"]
                    position = config["position"]
                    value = config["value"]
                    scale_factor = config.get("scale_factor", 1.0)
                    radius = config.get("radius", 0)
                    
                    # Create a new zeros tensor for the entire channel
                    channel_shape = modified_acts[0, channel].shape
                    modified_acts[0, channel] = torch.zeros(channel_shape, device=self.device)
                    
                    # Single point modification
                    if radius == 0:
                        modified_acts[0, channel, position[0], position[1]] = value
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
                                    modified_acts[0, channel, current_y, current_x] = scaled_value
                
                # Store modified activations
                self.modified_sae_activations.append(modified_acts.squeeze().cpu())
                
                # Use the decoder part of the SAE to get the reconstruction
                # The decoder in ConvSAE is the conv_dec layer
                reconstructed_output = self.sae.conv_dec(modified_acts)
                
                # Return the modified reconstruction to affect the model's behavior
                return reconstructed_output
            
            # Apply dynamic intervention if provided
            elif self.dynamic_intervention is not None:
                step = len(self.sae_activations) - 1  # Current step in the episode
                modified_acts = acts.clone()
                
                # Call the dynamic intervention function with current step and activations
                modified_acts = self.dynamic_intervention(
                    step=step, 
                    original_activations=acts, 
                    modified_activations=modified_acts
                )
                
                # Store modified activations
                self.modified_sae_activations.append(modified_acts.squeeze().cpu())
                
                # Use the decoder to get reconstructed output
                reconstructed_output = self.sae.conv_dec(modified_acts)
                
                # Return the modified reconstructions
                return reconstructed_output
            
            # If no intervention, just return the original output
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
                           output_path="maze_intervention_results", save_gif_freq=1,
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
        # Modify output path to include entity code and layer information
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
        
        # Create directory structure: intervention_results/[layer_name]_sae/entity_[code]_[description]
        layer_specific_output = f"{output_path}/{self.layer_name}_sae"
        
        # Create entity-specific subfolder
        entity_specific_output = f"{layer_specific_output}/entity_{entity1_code}_{entity_desc}"
        
        # Create output directory
        os.makedirs(entity_specific_output, exist_ok=True)
        
        print(f"Saving results to: {entity_specific_output}")
        
        # Create maze environment with specified entities
        # if maze_variant >= 0 and maze_variant <= 7:
        print(f"Creating maze variant {maze_variant} with entity1_code={entity1_code}, entity2_code={entity2_code}")
        obs, venv = create_box_maze(entity1_code,entity2_code)
        # else:
        #     # Use create_example_maze_sequence for custom maze with specified entities
        #     print(f"Creating example maze sequence with entity1_code={entity1_code}, entity2_code={entity2_code}")
        #     observations, venv = create_example_maze_sequence(entity1=entity1_code, entity2=entity2_code)
        #     # We'll ignore the observations returned by create_example_maze_sequence
        
        # Reset buffers
        self.sae_activations = []
        self.modified_sae_activations = []
        
        # Register hook for SAE activations
        handle = self.module.register_forward_hook(self._hook_sae_activations)
        
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
            if len(self.sae_activations) > 0:
                self._visualize_activations(entity_specific_output, maze_variant, intervention_type)
                # Also create GIFs of activation changes over time
                self._create_activation_gifs(entity_specific_output, maze_variant, intervention_type, 
                                            frames=frames, save_gif_freq=save_gif_freq,
                                            max_gif_frames=max_gif_frames)
        
        results = {
            "maze_variant": maze_variant,
            "total_steps": steps,
            "total_reward": total_reward,
            "intervention_type": intervention_type,
            "intervention_config": self.intervention_config,
            "entity1_code": entity1_code,
            "entity2_code": entity2_code,
            "output_path": entity_specific_output,
            "layer_number": self.layer_number,
            "layer_name": self.layer_name
        }
        
        return results
    
    def _visualize_activations(self, output_path, maze_variant, intervention_type):
        """Create visualization of activations for the first frame"""
        if not self.sae_activations:
            return
            
        # Create directory for activation visualizations
        vis_dir = f"{output_path}/activations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Determine which channels to visualize
        channels = []
        if intervention_type == "static" and self.intervention_config:
            channels = [cfg["channel"] for cfg in self.intervention_config]
        elif intervention_type == "dynamic":
            # Find the top activated channels in the first frame
            first_activation = self.sae_activations[0]
            channel_means = torch.mean(first_activation, dim=(1, 2))
            top_channels = torch.argsort(channel_means, descending=True)[:5]
            channels = top_channels.tolist()
        else:
            # For baseline, just show some random channels
            channels = list(range(0, 100, 20))[:5]  # Show channels 0, 20, 40, 60, 80
        
        # Create figure for the first frame's activations
        for channel in channels:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original activation
            orig_act = self.sae_activations[0][channel].numpy()
            im0 = axs[0].imshow(orig_act, cmap='viridis')
            axs[0].set_title(f"Original Channel {channel}")
            fig.colorbar(im0, ax=axs[0])
            
            # Modified activation (if available)
            if (intervention_type != "none") and len(self.modified_sae_activations) > 0:
                mod_act = self.modified_sae_activations[0][channel].numpy()
                im1 = axs[1].imshow(mod_act, cmap='viridis')
                axs[1].set_title(f"Modified Channel {channel}")
                fig.colorbar(im1, ax=axs[1])
            else:
                axs[1].imshow(orig_act, cmap='viridis', alpha=0.3)
                axs[1].set_title("No modification")
            
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/maze_{maze_variant}_{intervention_type}_ch{channel}_activations.png")
            plt.close()
            
    def _create_activation_gifs(self, output_path, maze_variant, intervention_type, 
                               frames=None, save_gif_freq=1, max_gif_frames=0):
        """Create GIFs of activation changes over time for specific channels
        
        Args:
            output_path (str): Directory to save results
            maze_variant (int): Maze variant used
            intervention_type (str): Type of intervention used
            frames (list, optional): List of environment frames
            save_gif_freq (int): Save every Nth timestep (default: 1 = all frames)
            max_gif_frames (int): Maximum number of frames to include in GIFs (0=all frames)
        """
        if not self.sae_activations or len(self.sae_activations) < 2:
            print("Not enough activations to create GIFs")
            return
            
        # Create directory for activation GIFs
        gifs_dir = f"{output_path}/activation_gifs"
        os.makedirs(gifs_dir, exist_ok=True)
        
        # Apply frame sampling based on save_gif_freq
        sampled_activations = self.sae_activations[::save_gif_freq]
        sampled_modified = self.modified_sae_activations[::save_gif_freq] if self.modified_sae_activations else []
        sampled_frames = frames[::save_gif_freq] if frames is not None else None
        
        # Limit number of frames if max_gif_frames is specified
        if max_gif_frames > 0 and len(sampled_activations) > max_gif_frames:
            print(f"Limiting GIFs to {max_gif_frames} frames (from {len(sampled_activations)} sampled frames)")
            sampled_activations = sampled_activations[:max_gif_frames]
            if sampled_modified:
                sampled_modified = sampled_modified[:max_gif_frames]
            if sampled_frames is not None:
                sampled_frames = sampled_frames[:max_gif_frames]
        
        # Print diagnostic info about frames
        print(f"Creating activation GIFs with {len(sampled_activations)} frames:")
        print(f"  - Sampling frequency: {save_gif_freq}")
        print(f"  - Max gif frames: {max_gif_frames if max_gif_frames > 0 else 'unlimited'}")
        print(f"  - Total original frames: {len(self.sae_activations)}")
        
        print(f"Creating activation GIFs with sampling frequency {save_gif_freq} "
              f"({len(sampled_activations)} frames out of {len(self.sae_activations)})")
        
        # self._create_all_channels_grid_gif(
        #     sampled_activations, 
        #     sampled_modified, 
        #     sampled_frames, 
        #     save_gif_freq, 
        #     gifs_dir,
        #     maze_variant, 
        #     intervention_type
        # )
        
        # Determine which specific channels to visualize in detail
        channels = []
        if intervention_type == "static" and self.intervention_config:
            channels = [cfg["channel"] for cfg in self.intervention_config]
        elif intervention_type == "dynamic":
            # Find channels with the highest variance over time
            # This identifies channels that change the most during the episode
            all_activations = torch.stack(self.sae_activations)
            channel_variance = torch.var(all_activations, dim=0).mean(dim=(1, 2))
            top_channels = torch.argsort(channel_variance, descending=True)[:5]
            channels = top_channels.tolist()
        else:
            # For baseline, show channels with highest variance
            all_activations = torch.stack(self.sae_activations)
            channel_variance = torch.var(all_activations, dim=0).mean(dim=(1, 2))
            top_channels = torch.argsort(channel_variance, descending=True)[:5]
            channels = top_channels.tolist()
            
        print(f"Creating detailed activation GIFs for channels: {channels}")
        
        # Create GIF for each channel
        for channel in channels:
            activation_frames = []
            
            # Create frames for each timestep
            for i, activation in enumerate(sampled_activations):
                # Create a figure with the original observation and the activation map
                fig, axs = plt.subplots(1, 2 if sampled_frames is None else 3, figsize=(16, 6))
                
                # Channel activation
                act = activation[channel].numpy()
                im0 = axs[0].imshow(act, cmap='viridis')
                axs[0].set_title(f"Channel {channel} (Step {i*save_gif_freq})")
                fig.colorbar(im0, ax=axs[0])
                
                # If we have modified activations, show those too
                if len(sampled_modified) > i:
                    mod_act = sampled_modified[i][channel].numpy()
                    im1 = axs[1].imshow(mod_act, cmap='viridis')
                    axs[1].set_title(f"Modified Ch {channel} (Step {i*save_gif_freq})")
                    fig.colorbar(im1, ax=axs[1])
                else:
                    axs[1].imshow(act, cmap='viridis', alpha=0.3)
                    axs[1].set_title("No modification")
                
                # If we have environment frames, show the corresponding frame
                if sampled_frames is not None and i < len(sampled_frames):
                    axs[2].imshow(sampled_frames[i])
                    axs[2].set_title(f"Environment (Step {i*save_gif_freq})")
                    axs[2].axis('off')
                
                plt.tight_layout()
                
                # Convert figure to image
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                activation_frames.append(img)
                plt.close(fig)
            
            # Save as GIF if we have frames
            if activation_frames:
                gif_path = f"{gifs_dir}/maze_{maze_variant}_{intervention_type}_ch{channel}_over_time.gif"
                imageio.mimsave(gif_path, activation_frames, fps=5)  # Slower fps for easier viewing
                print(f"Saved activation GIF for channel {channel} to {gif_path}")
        
        # Create a composite GIF showing multiple channels side by side
        if len(channels) > 1:
            composite_frames = []
            
            # Take first few channels for composite view
            display_channels = channels[:min(3, len(channels))]
            
            # Create frames for each timestep
            for i in range(len(sampled_activations)):
                # Create a figure with multiple channels side by side
                fig, axs = plt.subplots(1, len(display_channels) + (1 if sampled_frames is not None else 0), 
                                       figsize=(6 * (len(display_channels) + (1 if sampled_frames is not None else 0)), 6))
                
                # Show each channel
                for j, channel in enumerate(display_channels):
                    act = sampled_activations[i][channel].numpy()
                    im = axs[j].imshow(act, cmap='viridis')
                    axs[j].set_title(f"Channel {channel} (Step {i*save_gif_freq})")
                    fig.colorbar(im, ax=axs[j])
                
                # Show environment frame if available
                if sampled_frames is not None and i < len(sampled_frames):
                    axs[-1].imshow(sampled_frames[i])
                    axs[-1].set_title(f"Environment (Step {i*save_gif_freq})")
                    axs[-1].axis('off')
                
                plt.tight_layout()
                
                # Convert figure to image
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                composite_frames.append(img)
                plt.close(fig)
            
            # Save as GIF
            if composite_frames:
                gif_path = f"{gifs_dir}/maze_{maze_variant}_{intervention_type}_multi_channel.gif"
                imageio.mimsave(gif_path, composite_frames, fps=5)
                print(f"Saved multi-channel activation GIF to {gif_path}")
        
        return

    def _create_all_channels_grid_gif(self, activations, modified_activations, frames, save_gif_freq, output_dir, maze_variant, intervention_type):
        """Create a grid visualization of all channels in the SAE
        
        Args:
            activations (list): List of activation tensors for each timestep
            modified_activations (list): List of modified activation tensors for each timestep
            frames (list): List of environment frames
            save_gif_freq (int): Sampling frequency
            output_dir (str): Directory to save output
            maze_variant (int): Maze variant used
            intervention_type (str): Type of intervention (static, dynamic, etc.)
        """
        if not activations:
            return
            
        all_frames = []
        
        # Determine how many channels we have
        num_channels = activations[0].shape[0]
        print(f"Creating grid visualization for all {num_channels} channels")
        
        # Calculate grid dimensions (aiming for roughly square grid)
        grid_size = int(np.ceil(np.sqrt(num_channels)))
        
        # Process each timestep
        for i, activation in enumerate(activations):
            # Create a figure for the grid of channels
            fig = plt.figure(figsize=(20, 20))
            fig.suptitle(f"All SAE Channels (Step {i*save_gif_freq})", fontsize=16)
            
            # Add environment frame if available
            if frames is not None and i < len(frames):
                # Add environment frame in a separate subplot
                ax_frame = fig.add_subplot(grid_size+1, grid_size+1, 1)
                ax_frame.imshow(frames[i])
                ax_frame.set_title("Environment")
                ax_frame.axis('off')
            
            # Add intervention info if available
            if intervention_type == "static" and self.intervention_config:
                for config in self.intervention_config:
                    channel = config["channel"]
                    position = config["position"]
                    value = config["value"]
                    info_text = f"Intervention: Ch {channel} at {position} = {value}"
                    fig.text(0.5, 0.95, info_text, ha='center', fontsize=14)
            
            # Add each channel to the grid
            for j in range(num_channels):
                # Calculate grid position
                row = j // grid_size
                col = j % grid_size
                
                # Add subplot for this channel
                ax = fig.add_subplot(grid_size, grid_size, j+1)
                
                # Display the activation map
                act = activation[j].numpy()
                im = ax.imshow(act, cmap='viridis')
                
                # Add minimal labeling (just channel number)
                ax.set_title(f"Ch {j}", fontsize=8)
                ax.axis('off')
                
                # Highlight intervention channel if applicable
                if intervention_type == "static" and self.intervention_config:
                    for config in self.intervention_config:
                        if config["channel"] == j:
                            # Highlight this channel with a colored border
                            for spine in ax.spines.values():
                                spine.set_color('red')
                                spine.set_linewidth(3)
                            # Mark the intervention point
                            y, x = config["position"]
                            ax.plot(x, y, 'rx', markersize=8)
                            break
            
            plt.tight_layout()
            
            # Convert figure to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            all_frames.append(img)
            plt.close(fig)
        
        # Save as GIF
        if all_frames:
            grid_gif_path = f"{output_dir}/maze_{maze_variant}_{intervention_type}_all_channels_grid.gif"
            imageio.mimsave(grid_gif_path, all_frames, fps=3)  # Slower fps for easier viewing
            print(f"Saved grid visualization of all channels to {grid_gif_path}")

    def analyze_activation_patterns(self, maze_variants=[0, 1, 6, 7], channels=None, output_path="activation_analysis"):
        """
        Analyze activation patterns across different maze variants to identify
        channels that respond to specific spatial directions.
        
        Args:
            maze_variants (list): List of maze variants to analyze
            channels (list): Specific channels to analyze, or None to find automatically
            output_path (str): Directory to save analysis results
        
        Returns:
            dict: Analysis results with channels that respond to different directions
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Configure maze variants by direction
        directions = {
            "right": [0, 1],  # Variants where agent needs to move right
            "up": [2, 3],     # Variants where agent needs to move up
            "left": [4, 5],   # Variants where agent needs to move left
            "down": [6, 7]    # Variants where agent needs to move down
        }
        
        # Create subdirectory for this analysis
        analysis_dir = f"{output_path}/maze_variant_analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Store activations for each variant
        variant_activations = {}
        
        # Disable any interventions for this analysis
        self.disable_intervention()
        
        # Collect activations from each maze variant
        for variant in maze_variants:
            # Reset activation buffers
            self.sae_activations = []
            
            # Register hook for SAE activations (no intervention)
            handle = self.module.register_forward_hook(self._hook_sae_activations)
            
            # Create maze environment and get first observation
            venv = create_specific_l_shaped_maze_env(maze_variant=variant)
            observation = venv.reset()
            
            # Run model on first observation to get activations
            obs = observation[0]
            converted_obs = helpers.observation_to_rgb(obs)
            obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                self.model(obs_tensor)
            
            # Store activations for this variant
            if self.sae_activations:
                variant_activations[variant] = self.sae_activations[0]
            
            # Remove hook and close environment
            handle.remove()
            venv.close()
        
        # If no specific channels provided, analyze all channels
        if channels is None:
            num_channels = variant_activations[maze_variants[0]].shape[0]
            channels = list(range(num_channels))
        
        # Analyze activation patterns by direction
        direction_responsive_channels = {direction: [] for direction in directions}
        
        for direction, variants in directions.items():
            # Only use variants we've collected
            variants = [v for v in variants if v in variant_activations]
            if not variants:
                continue
                
            # For each channel, compare activations across variants
            for channel in channels:
                # Extract channel activations for this direction's variants
                direction_acts = [variant_activations[v][channel] for v in variants if v in variant_activations]
                
                if not direction_acts:
                    continue
                
                # Calculate mean activation for this direction
                mean_act = torch.mean(torch.stack(direction_acts), dim=0)
                
                # Calculate activations for other directions
                other_variants = [v for v in variant_activations if v not in variants]
                if not other_variants:
                    continue
                    
                other_acts = [variant_activations[v][channel] for v in other_variants]
                other_mean_act = torch.mean(torch.stack(other_acts), dim=0)
                
                # Compare activation patterns
                direction_strength = torch.mean(mean_act) / (torch.mean(other_mean_act) + 1e-6)
                
                # If this channel responds more strongly to this direction
                if direction_strength > 1.5:  # Threshold for significance
                    direction_responsive_channels[direction].append({
                        "channel": channel,
                        "strength": direction_strength.item()
                    })
        
        # Sort channels by strength
        for direction in direction_responsive_channels:
            direction_responsive_channels[direction].sort(key=lambda x: x["strength"], reverse=True)
        
        # Visualize top channels for each direction
        for direction, channels_info in direction_responsive_channels.items():
            if not channels_info:
                continue
                
            # Take top 3 channels for each direction
            top_channels = channels_info[:3]
            
            # Create visualization
            fig, axs = plt.subplots(len(top_channels), len(maze_variants), 
                                   figsize=(len(maze_variants)*3, len(top_channels)*3))
            
            # Ensure axs is 2D even with only one channel
            if len(top_channels) == 1:
                axs = np.array([axs])
            
            for i, channel_info in enumerate(top_channels):
                channel = channel_info["channel"]
                
                for j, variant in enumerate(maze_variants):
                    if variant not in variant_activations:
                        continue
                        
                    ax = axs[i, j]
                    activation = variant_activations[variant][channel].numpy()
                    
                    # Plot activation
                    im = ax.imshow(activation, cmap='viridis')
                    ax.set_title(f"Variant {variant}, Ch {channel}")
                    
                    # Add direction indicator
                    for dir_name, vars_in_dir in directions.items():
                        if variant in vars_in_dir:
                            variant_dir = dir_name
                            break
                    else:
                        variant_dir = "unknown"
                    
                    # Add subplot border color based on direction
                    color = "red" if variant_dir == direction else "gray"
                    for spine in ax.spines.values():
                        spine.set_color(color)
                        spine.set_linewidth(2 if variant_dir == direction else 1)
            
            plt.tight_layout()
            plt.savefig(f"{analysis_dir}/{direction}_responsive_channels.png")
            plt.close()
        
        return direction_responsive_channels


def create_trajectory_based_intervention(channel, start_pos, target_pos, max_steps, value=5.0, radius=1):
    """
    Creates a dynamic intervention function that moves the activation point along
    a trajectory from start to target position over time.
    
    Args:
        channel (int): Channel to intervene on
        start_pos (tuple): Starting position (y, x)
        target_pos (tuple): Target position (y, x)
        max_steps (int): Number of steps to complete the trajectory
        value (float): Value to set at intervention point
        radius (int): Radius of intervention
        
    Returns:
        function: Dynamic intervention function
    """
    # Calculate step sizes for each dimension
    start_y, start_x = start_pos
    target_y, target_x = target_pos
    
    step_y = (target_y - start_y) / max_steps
    step_x = (target_x - start_x) / max_steps
    
    def intervention_function(step, original_activations, modified_activations):
        # Calculate current position along trajectory
        progress = min(step, max_steps) / max_steps
        current_y = start_y + step_y * step
        current_x = start_x + step_x * step
        
        # Ensure position is within bounds
        h, w = modified_activations.shape[2], modified_activations.shape[3]
        current_y = max(0, min(h-1, current_y))
        current_x = max(0, min(w-1, current_x))
        
        # Convert to integers for indexing
        y, x = int(current_y), int(current_x)
        
        # Apply modification (similar to static intervention)
        if radius == 0:
            modified_activations[0, channel, y, x] = value
        else:
            # Create indices for the circular region
            device = modified_activations.device
            y_indices = torch.arange(max(0, y-radius), min(h, y+radius+1), device=device)
            x_indices = torch.arange(max(0, x-radius), min(w, x+radius+1), device=device)
            
            # Calculate distances
            Y, X = torch.meshgrid(y_indices, x_indices, indexing='ij')
            distances = torch.sqrt((Y - y)**2 + (X - x)**2)
            
            # Apply modification within radius
            mask = distances <= radius
            for i in range(len(y_indices)):
                for j in range(len(x_indices)):
                    if mask[i, j]:
                        curr_y, curr_x = y_indices[i].item(), x_indices[j].item()
                        # Scale value by distance from center
                        distance_ratio = 1.0 - (distances[i, j] / radius)
                        scaled_value = value * (0.8 ** (1.0 - distance_ratio.item()))
                        modified_activations[0, channel, curr_y, curr_x] = scaled_value
        
        return modified_activations
    
    return intervention_function


def run_maze_intervention_experiment(direction_channels=None):
    """Run predefined experiments with spatial interventions on SAE channels"""
    # Specify paths
    model_path = "../model_interpretable.pt"
    sae_checkpoint_path = "checkpoints/layer_6_conv3a/sae_checkpoint_step_1000000.pt"
    output_path = "maze_intervention_results"
    
    # Create experiment
    experiment = SAEInterventionExperiment(
        model_path=model_path,
        sae_checkpoint_path=sae_checkpoint_path,
        layer_number=6  # conv3a
    )
    
    # If no direction-responsive channels provided, analyze them
    if direction_channels is None:
        print("Analyzing direction-responsive channels...")
        direction_channels = experiment.analyze_activation_patterns(
            maze_variants=[0, 1, 2, 3, 4, 5, 6, 7],
            output_path=output_path
        )
    
    # Step 1: Run baseline without intervention
    print("\nRunning baseline experiment...")
    experiment.disable_intervention()
    baseline_results = experiment.run_maze_experiment(
        maze_variant=0,  # Original L-shaped maze
        max_steps=100,
        save_gif=True,
        output_path=output_path
    )
    
    # Step 2: Static intervention on channel 95 (the requested channel)
    print("\nRunning static intervention on channel 95...")
    static_config = [{"channel": 95, "position": (4, 4), "value": 5.0, "radius": 1}]
    experiment.set_intervention(static_config)
    static_results = experiment.run_maze_experiment(
        maze_variant=0,
        max_steps=100,
        save_gif=True,
        output_path=output_path
    )
    
    # Step 3: Try direction-specific channels
    print("\nRunning interventions on direction-specific channels...")
    direction_results = {}
    
    for direction, channels in direction_channels.items():
        if not channels:
            continue
            
        # Use the top channel for this direction
        top_channel = channels[0]["channel"]
        
        # Configure intervention based on direction
        if direction == "right":
            # Encourage rightward movement in variant 0
            config = [{"channel": top_channel, "position": (4, 6), "value": 5.0, "radius": 1}]
        elif direction == "left":
            # Encourage leftward movement in variant 0 (against natural direction)
            config = [{"channel": top_channel, "position": (4, 2), "value": 5.0, "radius": 1}]
        elif direction == "up":
            # Encourage upward movement in variant 0
            config = [{"channel": top_channel, "position": (2, 4), "value": 5.0, "radius": 1}]
        elif direction == "down":
            # Encourage downward movement in variant 0
            config = [{"channel": top_channel, "position": (6, 4), "value": 5.0, "radius": 1}]
        
        experiment.set_intervention(config)
        result = experiment.run_maze_experiment(
            maze_variant=0,
            max_steps=100,
            save_gif=True,
            output_path=output_path
        )
        
        direction_results[direction] = result
    
    # Step 4: Try dynamic intervention (moving activation point)
    print("\nRunning dynamic intervention with moving activation point...")
    
    # Find a right-responsive channel
    right_channel = 95  # Default
    if "right" in direction_channels and direction_channels["right"]:
        right_channel = direction_channels["right"][0]["channel"]
    
    # Create trajectory-based intervention to guide agent right then up
    dynamic_intervention = create_trajectory_based_intervention(
        channel=right_channel,
        start_pos=(4, 2),    # Start in the middle-left
        target_pos=(2, 6),   # End at the top-right
        max_steps=20,        # Complete the trajectory in 20 steps
        value=5.0,
        radius=1
    )
    
    experiment.set_dynamic_intervention(dynamic_intervention)
    dynamic_results = experiment.run_maze_experiment(
        maze_variant=0,
        max_steps=100,
        save_gif=True,
        output_path=output_path
    )
    
    # Compare results
    print("\nExperiment Results:")
    print(f"Baseline (no intervention): {baseline_results['total_steps']} steps, reward {baseline_results['total_reward']}")
    print(f"Static intervention on ch95: {static_results['total_steps']} steps, reward {static_results['total_reward']}")
    
    for direction, result in direction_results.items():
        channel = result["intervention_config"][0]["channel"]
        print(f"{direction.capitalize()} channel ({channel}): {result['total_steps']} steps, reward {result['total_reward']}")
    
    print(f"Dynamic intervention: {dynamic_results['total_steps']} steps, reward {dynamic_results['total_reward']}")
    
    return {
        "baseline": baseline_results,
        "static_ch95": static_results,
        "direction_specific": direction_results,
        "dynamic": dynamic_results,
        "direction_channels": direction_channels
    }


if __name__ == "__main__":
    run_maze_intervention_experiment() 