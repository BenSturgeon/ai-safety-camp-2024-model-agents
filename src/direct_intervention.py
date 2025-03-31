#!/usr/bin/env python
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from intervention_base import InterventionExperiment

# Import visualization functions from SAE implementation
from sae_spatial_intervention import SAEInterventionExperiment


class DirectInterventionExperiment(InterventionExperiment):
    """
    Experiment class for direct interventions on model layer activations
    without using SAEs.
    """
    
    def __init__(self, model_path, layer_number, device=None):
        """
        Initialize the experiment with model and target layer

        Args:
            model_path (str): Path to the model checkpoint
            layer_number (int): Layer number to target (corresponds to ordered_layer_names)
            device (torch.device): Device to run on (defaults to CUDA if available, else CPU)
        """
        super().__init__(model_path, layer_number, device)
        print(f"Initialized DirectInterventionExperiment targeting layer {self.layer_name}")
        
        # Get the shape of activations from this layer
        self._determine_activation_shape()
    
    def _determine_activation_shape(self):
        """Determine the shape of activations from the target layer"""
        # Create a dummy input to get activation shapes
        dummy_input = torch.zeros((1, 3, 64, 64), device=self.device)
        
        # Define a hook to capture output shape
        activation_shape = [None]
        def shape_hook(module, input, output):
            activation_shape[0] = output.shape
            return output
        
        # Register hook, run model, remove hook
        handle = self.module.register_forward_hook(shape_hook)
        with torch.no_grad():
            self.model(dummy_input)
        handle.remove()
        
        self.activation_shape = activation_shape[0]
        print(f"Target layer activation shape: {self.activation_shape}")
    
    def _hook_activations(self, module, input, output):
        """Hook to capture and/or modify raw model activations"""
        with torch.no_grad():
            # Store original activations for visualization
            orig_acts = output.clone()
            self.original_activations.append(orig_acts.cpu())
            
            # VERY IMPORTANT: Create a CLONE of the activations to avoid modifying the original
            modified_acts = output.clone()
            
            # Apply intervention if active and config exists
            if self.intervention_active and self.intervention_config:
                # Debug info on first step
                if len(self.original_activations) == 1:
                    print("\n======== APPLYING CHANNEL INTERVENTION ========")
                    for cfg in self.intervention_config:
                        ch = cfg["channel"]
                        pos = cfg["position"]
                        val = cfg.get("value", 0.0)
                        radius = cfg.get("radius", 0)
                        print(f"  → Channel {ch}: zeroing entire channel and setting {val} at position {pos} with radius {radius}")
                
                # For each channel in config, zero it out and set the target value
                for cfg in self.intervention_config:
                    channel = cfg["channel"]
                    position = cfg["position"]
                    value = cfg.get("value", 0.0)
                    radius = cfg.get("radius", 0)
                    
                    # CRITICAL DEBUG: Verify channel shape and tensor structure
                    print(f"Step {len(self.original_activations)}: Processing channel {channel}")
                    print(f"  Modified acts shape: {modified_acts.shape}")
                    
                    # First, zero out the entire channel with a new zeros tensor
                    # (This guarantees we're not keeping any original values)
                    channel_shape = modified_acts[0, channel].shape
                    print(f"  Channel shape: {channel_shape}")
                    
                    # Create a zeros tensor for the channel
                    zeros_tensor = torch.zeros(channel_shape, device=output.device)
                    
                    # Zero out the channel
                    modified_acts[0, channel] = zeros_tensor
                    
                    # Then, set the specific value at target position (with radius if specified)
                    if radius > 0:
                        # Apply to all positions within radius
                        y, x = position
                        h, w = modified_acts.shape[2], modified_acts.shape[3]
                        
                        # Create a mask for the circular region
                        y_indices = torch.arange(max(0, y-radius), min(h, y+radius+1), device=modified_acts.device)
                        x_indices = torch.arange(max(0, x-radius), min(w, x+radius+1), device=modified_acts.device)
                        
                        # Calculate distances
                        Y, X = torch.meshgrid(y_indices, x_indices, indexing='ij')
                        distances = torch.sqrt((Y - y)**2 + (X - x)**2)
                        
                        # Apply modification within radius
                        mask = distances <= radius
                        for i in range(len(y_indices)):
                            for j in range(len(x_indices)):
                                if mask[i, j]:
                                    current_y, current_x = y_indices[i], x_indices[j]
                                    modified_acts[0, channel, current_y, current_x] = value
                    else:
                        # Simply set value at the exact position
                        y, x = position
                        modified_acts[0, channel, y, x] = value
                    
                    # Verify the intervention
                    if radius == 0:
                        y, x = position
                        orig_val = orig_acts[0, channel, y, x].item()
                        mod_val = modified_acts[0, channel, y, x].item()
                        print(f"  At position {position}: Original={orig_val:.4f}, Modified={mod_val:.4f}")
                        
                        # Check if channel is properly zeroed except for target position
                        channel_sum = modified_acts[0, channel].sum().item()
                        if abs(channel_sum - value) > 1e-5:
                            print(f"  WARNING: Channel sum ({channel_sum:.4f}) doesn't match target value ({value:.4f})")
                        else:
                            print(f"  SUCCESS: Channel sum ({channel_sum:.4f}) matches target value ({value:.4f})")
            
            # Store modified activations for visualization
            self.modified_activations.append(modified_acts.cpu())
            
            # CRUCIAL: Return the modified tensor to affect the model's forward pass
            # When debugging, print before returning to ensure modified_acts is different from output
            if self.intervention_active and len(self.original_activations) == 1:
                print(f"Original output norm: {output.norm().item()}")
                print(f"Modified output norm: {modified_acts.norm().item()}")
                if torch.allclose(output, modified_acts):
                    print("WARNING: Output and modified_acts are identical!")
                else:
                    print("SUCCESS: Output has been modified")
            
            # Always return the modified tensor to affect the model
            return modified_acts
    
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
        if not self.original_activations or len(self.original_activations) < 2:
            print("Not enough activations to create GIFs")
            return
        
        # Create directory for activation GIFs
        gifs_dir = f"{output_path}/activation_gifs"
        os.makedirs(gifs_dir, exist_ok=True)
        
        # Model type and layer info for labeling
        model_type = "Base Model"
        layer_info = f"Layer {self.layer_number} ({self.layer_name})"
        
        # Sample frames based on save_gif_freq
        sampled_activations = self.original_activations[::save_gif_freq]
        sampled_modified = self.modified_activations[::save_gif_freq] if self.modified_activations else []
        sampled_frames = frames[::save_gif_freq] if frames is not None else None
        
        # Limit number of frames if max_gif_frames is specified
        if max_gif_frames > 0 and len(sampled_activations) > max_gif_frames:
            print(f"Limiting GIFs to {max_gif_frames} frames (from {len(sampled_activations)} sampled frames)")
            sampled_activations = sampled_activations[:max_gif_frames]
            if sampled_modified:
                sampled_modified = sampled_modified[:max_gif_frames]
            if sampled_frames is not None:
                sampled_frames = sampled_frames[:max_gif_frames]
        
        # Debug info about intervention state
        print(f"\nBefore creating GIFs:")
        print(f"  Intervention type: {intervention_type}")
        print(f"  intervention_active: {self.intervention_active}")
        print(f"  intervention_config available: {self.intervention_config is not None}")
        if self.intervention_config:
            print(f"  Number of channels in config: {len(self.intervention_config)}")
            for i, cfg in enumerate(self.intervention_config):
                print(f"    Channel {i+1}: {cfg['channel']}, position: {cfg['position']}")
        
        # Determine channels to visualize - for static interventions, ALWAYS use config
        if intervention_type == "static" and self.intervention_config:
            channels = [cfg["channel"] for cfg in self.intervention_config]
            positions = {cfg["channel"]: cfg["position"] for cfg in self.intervention_config}
            radii = {cfg["channel"]: cfg.get("radius", 0) for cfg in self.intervention_config}
            print(f"Creating GIFs for intervention channels: {channels}")
        elif self.intervention_config:
            # Use config if available (even for baseline with config)
            channels = [cfg["channel"] for cfg in self.intervention_config]
            positions = {cfg["channel"]: cfg["position"] for cfg in self.intervention_config}
            radii = {cfg["channel"]: cfg.get("radius", 0) for cfg in self.intervention_config}
            print(f"Creating GIFs using stored config channels: {channels}")
        else:
            # Fallback case - this should rarely happen
            print("WARNING: No intervention_config found! Using fallback channels.")
            channels = [29]  # Default fallback to channel 29
            positions = {}
            radii = {}
        
        # Ensure channels is a list
        if not isinstance(channels, list):
            channels = [channels]
        
        # Create GIF for each channel
        for channel in channels:
            activation_frames = []
            
            # Create frames for each timestep
            for i, activation in enumerate(tqdm(sampled_activations, desc=f"Creating frames for channel {channel}")):
                # Create a figure with the original activation and environment
                n_cols = 3 if sampled_frames is not None else 2
                fig, axs = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
                
                # Make sure axs is iterable even with single plot
                if n_cols == 1:
                    axs = [axs]
                    
                # Original channel activation
                orig_act = activation[0, channel].numpy()
                im0 = axs[0].imshow(orig_act, cmap='viridis')
                axs[0].set_title(f"Channel {channel} (Original)")
                fig.colorbar(im0, ax=axs[0])
                
                # Highlight position if specified for this channel
                if channel in positions:
                    y, x = positions[channel]
                    radius = radii.get(channel, 0)
                    
                    # Mark the position
                    axs[0].plot(x, y, 'rx', markersize=10)
                    
                    # Add circle for radius
                    if radius > 0:
                        circle = plt.Circle((x, y), radius, fill=False, color='r', linewidth=2)
                        axs[0].add_patch(circle)
                
                # Modified activation if available
                if len(sampled_modified) > i:
                    mod_act = sampled_modified[i][0, channel].numpy()
                    im1 = axs[1].imshow(mod_act, cmap='viridis')
                    axs[1].set_title(f"Channel {channel} (Modified)")
                    fig.colorbar(im1, ax=axs[1])
                    
                    # Also highlight position in modified activation
                    if channel in positions:
                        y, x = positions[channel]
                        radius = radii.get(channel, 0)
                        
                        # Mark the position
                        axs[1].plot(x, y, 'rx', markersize=10)
                        
                        # Add circle for radius
                        if radius > 0:
                            circle = plt.Circle((x, y), radius, fill=False, color='r', linewidth=2)
                            axs[1].add_patch(circle)
                else:
                    # No modifications available - show original with lower opacity
                    axs[1].imshow(orig_act, cmap='viridis', alpha=0.3)
                    axs[1].set_title("No modification")
                
                # Environment frame if available
                if sampled_frames is not None and i < len(sampled_frames):
                    axs[2].imshow(sampled_frames[i])
                    axs[2].set_title(f"Environment (Step {i*save_gif_freq})")
                    axs[2].axis('off')
                
                # Set overall title with model type and layer info
                fig.suptitle(f"{model_type} - {layer_info}\nChannel {channel} - Step {i*save_gif_freq}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust to make room for the title
                
                # Convert figure to image
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                activation_frames.append(img)
                plt.close(fig)
            
            # Save as GIF if we have frames
            if activation_frames:
                # Include model type and layer in the filename
                model_layer_str = f"base_model_L{self.layer_number}"
                gif_path = f"{gifs_dir}/maze_{maze_variant}_{intervention_type}_{model_layer_str}_ch{channel}_over_time.gif"
                imageio.mimsave(gif_path, activation_frames, fps=3)  # Slower fps for easier viewing
                print(f"Saved activation GIF for channel {channel} to {gif_path}")
        
        # If multiple channels specified, create a combined GIF
        if len(channels) > 1:
            self._create_multichannel_gif(
                channels=channels,
                positions=positions,
                radii=radii,
                sampled_activations=sampled_activations,
                sampled_modified=sampled_modified,
                sampled_frames=sampled_frames,
                save_gif_freq=save_gif_freq,
                gifs_dir=gifs_dir,
                maze_variant=maze_variant,
                intervention_type=intervention_type,
                model_type=model_type,
                layer_info=layer_info
            )
        
        return

    def _create_multichannel_gif(self, channels, positions, radii, sampled_activations, 
                               sampled_modified, sampled_frames, save_gif_freq, 
                               gifs_dir, maze_variant, intervention_type, model_type, layer_info):
        """Create a GIF showing multiple channels side by side"""
        print(f"Creating combined GIF for channels: {channels}")
        
        # Limit to a reasonable number of channels for the grid
        display_channels = channels[:min(3, len(channels))]
        
        # Create frames for the GIF
        combined_frames = []
        
        # Process each timestep
        for i in range(len(sampled_activations)):
            # Create a figure with channels side by side
            n_cols = len(display_channels) + (1 if sampled_frames is not None else 0)
            fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
            
            # Make axs iterable even with single subplot
            if n_cols == 1:
                axs = [axs]
            
            # Display each channel's activation
            for j, channel in enumerate(display_channels):
                # Choose to display original or modified based on intervention type
                if intervention_type != "none" and len(sampled_modified) > i:
                    act = sampled_modified[i][0, channel].numpy()
                    label = "Modified"
                else:
                    act = sampled_activations[i][0, channel].numpy()
                    label = "Original"
                
                # Plot the activation
                im = axs[j].imshow(act, cmap='viridis')
                axs[j].set_title(f"Channel {channel} ({label})")
                fig.colorbar(im, ax=axs[j], fraction=0.046, pad=0.04)
                
                # Highlight position if specified
                if channel in positions:
                    y, x = positions[channel]
                    radius = radii.get(channel, 0)
                    
                    # Mark position
                    axs[j].plot(x, y, 'rx', markersize=10)
                    
                    # Add circle for radius
                    if radius > 0:
                        circle = plt.Circle((x, y), radius, fill=False, color='r', linewidth=2)
                        axs[j].add_patch(circle)
            
            # Add environment frame if available
            if sampled_frames is not None and i < len(sampled_frames):
                axs[-1].imshow(sampled_frames[i])
                axs[-1].set_title(f"Environment (Step {i*save_gif_freq})")
                axs[-1].axis('off')
            
            # Add overall title with model type and layer info
            fig.suptitle(f"{model_type} - {layer_info}\nMultiple Channels - Step {i*save_gif_freq}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust to make room for the title
            
            # Convert figure to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            combined_frames.append(img)
            plt.close(fig)
        
        # Save as GIF
        if combined_frames:
            # Include model type and layer in the filename
            model_layer_str = f"base_model_L{self.layer_number}"
            gif_path = f"{gifs_dir}/maze_{maze_variant}_{intervention_type}_{model_layer_str}_multichannel.gif"
            imageio.mimsave(gif_path, combined_frames, fps=3)
            print(f"Saved multi-channel GIF to {gif_path}")
    
    def _visualize_activations(self, output_path, maze_variant, intervention_type):
        """Create visualization of activations for the first frame
        
        Args:
            output_path (str): Directory to save results
            maze_variant (int): Maze variant used
            intervention_type (str): Type of intervention used
        """
        if not self.original_activations:
            return
        
        # Create directory for activation visualizations
        vis_dir = f"{output_path}/activations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Only visualize channels from intervention_config
        if not self.intervention_config:
            print("No intervention config found, skipping activation visualization")
            return
            
        channels = [cfg["channel"] for cfg in self.intervention_config]
        print(f"Visualizing specified channels: {channels}")
        
        # Create figure for each channel
        for channel in channels:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original activation
            orig_act = self.original_activations[0][0, channel].numpy()
            im0 = axs[0].imshow(orig_act, cmap='viridis')
            axs[0].set_title(f"Original Channel {channel}")
            fig.colorbar(im0, ax=axs[0])
            
            # Modified activation (if available)
            if len(self.modified_activations) > 0:
                mod_act = self.modified_activations[0][0, channel].numpy()
                im1 = axs[1].imshow(mod_act, cmap='viridis')
                axs[1].set_title(f"Modified Channel {channel}")
                fig.colorbar(im1, ax=axs[1])
                
                # Add some diff stats
                max_diff = np.max(np.abs(mod_act - orig_act))
                plt.figtext(0.5, 0.01, f"Max difference: {max_diff:.4f}",
                           ha="center", fontsize=12, bbox={"facecolor":"yellow", "alpha":0.2, "pad":5})
            else:
                # Show placeholder if no modifications
                axs[1].text(0.5, 0.5, "No modified activations available", 
                           ha='center', va='center', transform=axs[1].transAxes)
                axs[1].set_title("No modification")
            
            # Highlight positions from intervention_config
            for cfg in self.intervention_config:
                if cfg["channel"] == channel:
                    y, x = cfg["position"]
                    
                    # Add marker to both subplots
                    for ax in axs:
                        ax.plot(x, y, 'rx', markersize=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"{vis_dir}/maze_{maze_variant}_{intervention_type}_ch{channel}_activations.png")
            plt.close()

    def set_intervention(self, config):
        """
        Set up a static intervention configuration
        
        Args:
            config (list): List of dicts with intervention parameters:
                - channel (int): Channel to intervene on
                - position (tuple): (y, x) position to intervene at
                - value (float, optional): Value to set at that position (default 0.0)
                - radius (int, optional): Radius of intervention (default 0 for single point)
                - scale_factor (float, optional): Scale factor for radius-based interventions
        """
        # Ensure all configs have required fields
        for cfg in config:
            if "channel" not in cfg:
                raise ValueError("Each config entry must include 'channel'")
            if "position" not in cfg:
                raise ValueError("Each config entry must include 'position'")
            # Ensure value field exists, default to 0.0
            if "value" not in cfg:
                cfg["value"] = 0.0
        
        # Explicitly set both properties - this ensures they're set in this class
        self.intervention_config = config
        self.intervention_active = True
        
        print(f"Static intervention configured for {len(config)} points")
        
        # Print configuration details
        for cfg in config:
            channel = cfg["channel"]
            position = cfg["position"]
            value = cfg.get("value", 0.0)
            radius = cfg.get("radius", 0)
            print(f"  Channel {channel}: zero channel and set value {value} at position {position} (radius: {radius})")
        
        print(f"DirectInterventionExperiment: intervention_active={self.intervention_active}, config has {len(self.intervention_config)} items")


