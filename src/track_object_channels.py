#!/usr/bin/env python
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import argparse
from PIL import Image
import imageio
from tqdm import tqdm

# Import local modules
from utils.environment_modification_experiments import create_example_maze_sequence
from utils import helpers
from sae_cnn import load_sae_from_checkpoint, ordered_layer_names

class EntityTrackingExperiment:
    def __init__(self, model_path, device=None):
        """
        Initialize the experiment to track SAE channels that correlate with entity positions.
        
        Args:
            model_path (str): Path to the model checkpoint
            device (torch.device): Device to run on (defaults to CUDA if available, else CPU)
        """
        # Set up device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = helpers.load_interpretable_model(model_path=model_path).to(self.device)
        
        # Define the SAE layers we want to analyze
        self.target_layers = {
            1: "conv1a",
            3: "conv2a",
            4: "conv2b",
            6: "conv3a",
            8: "conv4a",
        }
        
        # Store loaded SAEs
        self.saes = {}
        
        # Store module handles
        self.handles = []
        
        # Storage for activations
        self.all_activations = {}
        
    def load_sae(self, layer_number, step=1000000):
        """Load an SAE for a specific layer"""
        layer_name = self.target_layers[layer_number]
        checkpoint_path = f"checkpoints/layer_{layer_number}_{layer_name}/sae_checkpoint_step_{step}.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: SAE checkpoint not found at {checkpoint_path}")
            return False
        
        print(f"Loading SAE for layer {layer_name} from {checkpoint_path}")
        try:
            sae = load_sae_from_checkpoint(checkpoint_path).to(self.device)
            self.saes[layer_number] = sae
            return True
        except Exception as e:
            print(f"Error loading SAE: {e}")
            return False
    
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
    
    def _hook_sae_activations(self, layer_number):
        """Create a hook function for the specified layer"""
        def hook(module, input, output):
            with torch.no_grad():
                output = output.to(self.device)
                sae = self.saes[layer_number].to(self.device)
                _, _, acts, _ = sae(output)
                
                layer_name = self.target_layers[layer_number]
                if layer_name not in self.all_activations:
                    self.all_activations[layer_name] = []
                
                self.all_activations[layer_name].append(acts.squeeze().cpu())
            
            return output
        return hook
    
    def register_hooks(self):
        """Register hooks for all loaded SAEs"""
        # First remove any existing hooks
        self.remove_hooks()
        
        # Register new hooks for each loaded SAE
        for layer_number in self.saes:
            layer_name = ordered_layer_names[layer_number]
            module = self._get_module(self.model, layer_name)
            
            # Create and register the hook
            hook_fn = self._hook_sae_activations(layer_number)
            handle = module.register_forward_hook(hook_fn)
            self.handles.append(handle)
            
            print(f"Registered hook for layer {self.target_layers[layer_number]}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("Removed all hooks")
    
    def reset_activations(self):
        """Clear stored activations"""
        self.all_activations = {}
    
    def run_entity_tracking_experiment(self, entity1=3, entity2=4, output_path="entity_tracking_results"):
        """
        Run an experiment to track how SAE activations correlate with entity positions.
        
        Args:
            entity1 (int): Entity code for the first entity (default: 3 = gem)
            entity2 (int): Entity code for the second entity (default: 4 = blue key)
            output_path (str): Directory to save results
        
        Returns:
            dict: Results showing which channels correlate with entity positions
        """
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Reset activations
        self.reset_activations()
        
        # Make sure hooks are registered
        if not self.handles:
            self.register_hooks()
        
        # Create maze sequence with the moving entity
        print(f"Creating maze sequence with entity1={entity1}, entity2={entity2}")
        observations, venv = create_example_maze_sequence(entity1, entity2)
        
        # Store the positions of entity2 (the one we're tracking)
        # Instead of using frame indices, use known positions from create_example_maze_sequence
        # These positions correspond to how the blue key moves in the maze pattern
        known_positions = [
            (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), 
            (4, 5), (4, 4), (4, 3), (4, 2), (4, 1), (3, 1)
        ]
        
        entity_positions = []
        entity_x_positions = []
        entity_y_positions = []
        
        # Process each observation
        print("Processing observations...")
        for i, obs in enumerate(tqdm(observations)):
            # Use frame index for position tracking, but also track actual x,y coordinates
            entity_positions.append(i)
            
            # Get the coordinates from known positions or default to center
            if i < len(known_positions):
                y_pos, x_pos = known_positions[i]
            else:
                # Default position if we go beyond known positions
                y_pos, x_pos = 3.5, 3.5
                
            entity_x_positions.append(x_pos)
            entity_y_positions.append(y_pos)
            
            # Convert observation to tensor for the model
            converted_obs = helpers.observation_to_rgb(obs)
            obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(self.device)
            
            # Make sure obs_tensor has a batch dimension
            if obs_tensor.ndim == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Run model to trigger activation hooks
            with torch.no_grad():
                self.model(obs_tensor)
        
        # Close the environment
        venv.close()
        
        # Save the actual positions for reference
        position_file = os.path.join(output_path, "entity_positions.csv")
        with open(position_file, "w") as f:
            f.write("frame,x,y\n")
            for frame, x, y in zip(entity_positions, entity_x_positions, entity_y_positions):
                f.write(f"{frame},{x},{y}\n")
        
        # Analyze correlations between activations and entity positions
        results = self.analyze_correlations_with_positions(
            entity_positions, entity_x_positions, entity_y_positions, output_path
        )
        
        # Create visualizations
        self.visualize_results(observations, entity_positions, results, output_path)
        
        return results
    
    def analyze_correlations_with_positions(self, frame_positions, x_positions, y_positions, output_path):
        """
        Analyze correlations between SAE activations and entity positions.
        
        Args:
            frame_positions (list): List of frame indices
            x_positions (list): List of x-coordinates
            y_positions (list): List of y-coordinates
            output_path (str): Directory to save results
        
        Returns:
            dict: Results showing which channels correlate with entity positions
        """
        results = {}
        
        # Convert position lists to numpy arrays
        frame_pos = np.array(frame_positions)
        x_pos = np.array(x_positions)
        y_pos = np.array(y_positions)
        
        print("Analyzing correlations with entity position...")
        
        # For each layer
        for layer_name, activations_list in self.all_activations.items():
            print(f"Analyzing layer {layer_name}...")
            
            # Stack all activations for this layer
            all_acts = torch.stack(activations_list)
            num_channels = all_acts.shape[1]
            
            # Calculate correlation for each channel with X and Y positions
            x_correlations = []
            y_correlations = []
            frame_correlations = []
            
            for channel in range(num_channels):
                # Get mean activation for this channel across all spatial positions
                channel_acts = all_acts[:, channel].mean(dim=(1, 2)).numpy()
                
                # Calculate correlation with position
                if len(channel_acts) > 1:  # Need at least 2 points for correlation
                    try:
                        # Correlation with X position
                        x_corr, x_p_value = pearsonr(x_pos, channel_acts)
                        x_correlations.append((channel, x_corr, x_p_value))
                        
                        # Correlation with Y position
                        y_corr, y_p_value = pearsonr(y_pos, channel_acts)
                        y_correlations.append((channel, y_corr, y_p_value))
                        
                        # Correlation with frame index (for reference)
                        frame_corr, frame_p_value = pearsonr(frame_pos, channel_acts)
                        frame_correlations.append((channel, frame_corr, frame_p_value))
                    except:
                        x_correlations.append((channel, 0, 1.0))
                        y_correlations.append((channel, 0, 1.0))
                        frame_correlations.append((channel, 0, 1.0))
            
            # Sort by absolute correlation value (descending)
            x_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            y_correlations.sort(key=lambda y: abs(y[1]), reverse=True)
            frame_correlations.sort(key=lambda f: abs(f[1]), reverse=True)
            
            # Store results for this layer
            results[layer_name] = {
                "x_correlations": x_correlations,
                "y_correlations": y_correlations,
                "frame_correlations": frame_correlations,
                "top_x_channels": [c[0] for c in x_correlations[:10]],
                "top_y_channels": [c[0] for c in y_correlations[:10]],
                "top_frame_channels": [c[0] for c in frame_correlations[:10]]
            }
            
            # Save correlation results to CSV
            with open(f"{output_path}/{layer_name}_x_correlations.csv", "w") as f:
                f.write("Channel,Correlation,P-Value\n")
                for channel, corr, p_value in x_correlations:
                    f.write(f"{channel},{corr},{p_value}\n")
            
            with open(f"{output_path}/{layer_name}_y_correlations.csv", "w") as f:
                f.write("Channel,Correlation,P-Value\n")
                for channel, corr, p_value in y_correlations:
                    f.write(f"{channel},{corr},{p_value}\n")
            
            with open(f"{output_path}/{layer_name}_frame_correlations.csv", "w") as f:
                f.write("Channel,Correlation,P-Value\n")
                for channel, corr, p_value in frame_correlations:
                    f.write(f"{channel},{corr},{p_value}\n")
        
        return results
    
    def visualize_results(self, observations, positions, results, output_path):
        """
        Create visualizations of the results.
        
        Args:
            observations (list): List of observations
            positions (list): List of frame indices (entity_positions)
            results (dict): Results from analyze_correlations_with_positions
            output_path (str): Directory to save visualizations
        """
        print("Creating visualizations...")
        
        # Create a directory for visualizations
        vis_dir = f"{output_path}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load position data from the CSV file we saved
        positions_file = os.path.join(output_path, "entity_positions.csv")
        x_positions = []
        y_positions = []
        
        with open(positions_file, "r") as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    x_positions.append(float(parts[1]))
                    y_positions.append(float(parts[2]))
        
        # Plot entity trajectory
        plt.figure(figsize=(10, 10))
        plt.plot(x_positions, y_positions, 'b-o', label='Blue Key Path')
        plt.xlim(0, 7)
        plt.ylim(0, 7)
        plt.grid(True)
        plt.title('Blue Key Path')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.savefig(f"{vis_dir}/entity_path.png", dpi=150)
        plt.close()
        
        # Create heatmaps of correlations for each layer
        for layer_name, layer_results in results.items():
            # Create X correlation chart
            plt.figure(figsize=(12, 8))
            top_channels = layer_results["x_correlations"][:20]
            channels = [c[0] for c in top_channels]
            correlations = [c[1] for c in top_channels]
            plt.barh(range(len(channels)), [abs(c) for c in correlations], color=['r' if c < 0 else 'g' for c in correlations])
            plt.yticks(range(len(channels)), channels)
            plt.xlabel("X Position Correlation (absolute)")
            plt.ylabel("Channel")
            plt.title(f"Top X-Correlating Channels for Layer {layer_name}")
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/{layer_name}_x_correlations.png", dpi=150)
            plt.close()
            
            # Create Y correlation chart
            plt.figure(figsize=(12, 8))
            top_channels = layer_results["y_correlations"][:20]
            channels = [c[0] for c in top_channels]
            correlations = [c[1] for c in top_channels]
            plt.barh(range(len(channels)), [abs(c) for c in correlations], color=['r' if c < 0 else 'g' for c in correlations])
            plt.yticks(range(len(channels)), channels)
            plt.xlabel("Y Position Correlation (absolute)")
            plt.ylabel("Channel")
            plt.title(f"Top Y-Correlating Channels for Layer {layer_name}")
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/{layer_name}_y_correlations.png", dpi=150)
            plt.close()
            
            # Create frame correlation chart
            plt.figure(figsize=(12, 8))
            top_channels = layer_results["frame_correlations"][:20]
            channels = [c[0] for c in top_channels]
            correlations = [c[1] for c in top_channels]
            plt.barh(range(len(channels)), [abs(c) for c in correlations], color=['r' if c < 0 else 'g' for c in correlations])
            plt.yticks(range(len(channels)), channels)
            plt.xlabel("Frame Index Correlation (absolute)")
            plt.ylabel("Channel")
            plt.title(f"Top Frame-Correlating Channels for Layer {layer_name}")
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/{layer_name}_frame_correlations.png", dpi=150)
            plt.close()
        
        # Create activation visualizations for the top channels
        for layer_name, layer_results in results.items():
            activations_list = self.all_activations[layer_name]
            
            # Get top X and Y correlated channels (avoiding duplicates)
            top_x_channels = layer_results["top_x_channels"][:3]
            top_y_channels = [c for c in layer_results["top_y_channels"][:3] if c not in top_x_channels]
            
            # Combined unique list of top channels
            top_channels = top_x_channels + top_y_channels
            if len(top_channels) > 5:
                top_channels = top_channels[:5]
            
            # Create a visualization showing how these channels change over time
            frames = []
            
            for i, (obs, acts) in enumerate(zip(observations, activations_list)):
                fig, axs = plt.subplots(2, 3, figsize=(18, 12))
                
                # Show the original observation with the entity's position
                axs[0, 0].imshow(obs.squeeze().transpose(1, 2, 0))
                if i < len(x_positions) and i < len(y_positions):
                    axs[0, 0].plot(x_positions[i], y_positions[i], 'ro', markersize=10)
                axs[0, 0].set_title(f"Frame {i}")
                axs[0, 0].axis("off")
                
                # Show activations for top channels
                for j, channel in enumerate(top_channels):
                    row, col = (j+1) // 3, (j+1) % 3
                    ax = axs[row, col]
                    
                    # Get activation for this channel
                    act = acts[channel].numpy()
                    
                    # Plot activation
                    im = ax.imshow(act, cmap='viridis')
                    # Add entity position on the activation map
                    if i < len(x_positions) and i < len(y_positions):
                        # Scale the position to the activation map dimensions
                        h, w = act.shape
                        scaled_x = x_positions[i] * w / 7  # Assuming 7x7 maze
                        scaled_y = y_positions[i] * h / 7
                        ax.plot(scaled_x, scaled_y, 'ro', markersize=5)
                    
                    # Determine which dimension(s) this channel correlates with
                    corr_type = []
                    if channel in layer_results["top_x_channels"][:5]:
                        corr_type.append("X")
                    if channel in layer_results["top_y_channels"][:5]:
                        corr_type.append("Y")
                    if channel in layer_results["top_frame_channels"][:5]:
                        corr_type.append("Frame")
                    
                    corr_label = ",".join(corr_type)
                    
                    ax.set_title(f"Channel {channel} ({corr_label})")
                    fig.colorbar(im, ax=ax)
                
                # Save the figure to a buffer
                fig.tight_layout()
                fig.canvas.draw()
                
                # Convert figure to numpy array
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                frames.append(img)
                plt.close(fig)
            
            # Create a GIF of the visualizations
            if frames:
                imageio.mimsave(f"{vis_dir}/{layer_name}_top_channels.gif", frames, fps=2)
        
        # Create one aggregated visualization showing the top channel from each layer
        if len(results) > 0:
            print("Creating aggregated visualization across layers...")
            frames = []
            
            # Determine how many layers we have
            n_layers = len(results)
            
            for i, obs in enumerate(observations):
                # Create a figure with the observation and top channel from each layer
                fig, axs = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))
                
                # Show the original observation with the entity position
                axs[0].imshow(obs.squeeze().transpose(1, 2, 0))
                if i < len(x_positions) and i < len(y_positions):
                    axs[0].plot(x_positions[i], y_positions[i], 'ro', markersize=10)
                axs[0].set_title(f"Frame {i}")
                axs[0].axis("off")
                
                # Show top channel from each layer
                for j, (layer_name, layer_results) in enumerate(results.items()):
                    if layer_name in self.all_activations and i < len(self.all_activations[layer_name]):
                        # Get activation for the top channel - prefer X position channels
                        if "top_x_channels" in layer_results and layer_results["top_x_channels"]:
                            top_channel = layer_results["top_x_channels"][0]
                        elif "top_y_channels" in layer_results and layer_results["top_y_channels"]:
                            top_channel = layer_results["top_y_channels"][0]
                        else:
                            top_channel = layer_results["top_frame_channels"][0]
                            
                        act = self.all_activations[layer_name][i][top_channel].numpy()
                        
                        # Plot activation
                        im = axs[j+1].imshow(act, cmap='viridis')
                        axs[j+1].set_title(f"{layer_name}\nCh {top_channel}")
                        fig.colorbar(im, ax=axs[j+1])
                        
                        # Mark the entity position
                        if i < len(x_positions) and i < len(y_positions):
                            # Scale to activation dimensions
                            h, w = act.shape
                            scaled_x = x_positions[i] * w / 7
                            scaled_y = y_positions[i] * h / 7
                            axs[j+1].plot(scaled_x, scaled_y, 'ro', markersize=5)
                
                # Save the figure to a buffer
                fig.tight_layout()
                fig.canvas.draw()
                
                # Convert figure to numpy array
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                frames.append(img)
                plt.close(fig)
            
            # Create a GIF of the aggregated visualization
            if frames:
                imageio.mimsave(f"{vis_dir}/all_layers_top_channels.gif", frames, fps=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Track which SAE channels correlate with entity positions")
    
    parser.add_argument("--model_path", type=str, default="../model_interpretable.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--entity1", type=int, default=3,
                        help="Entity code for the first entity (default: 3 = gem)")
    parser.add_argument("--entity2", type=int, default=4,
                        help="Entity code for the second entity (default: 4 = blue key)")
    parser.add_argument("--output", type=str, default="entity_tracking_results",
                        help="Directory to save results")
    parser.add_argument("--sae_step", type=int, default=1000000,
                        help="Step number for SAE checkpoints (default: 1000000)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 80)
    print("ENTITY TRACKING EXPERIMENT")
    print("=" * 80)
    print(f"This experiment will analyze which SAE channels correlate with entity movement.")
    print(f"We'll track the blue key ({args.entity2}) as it moves in a predefined path around the maze.")
    print(f"For each layer's SAE, we'll analyze which channels' activations correlate with:")
    print(f"  - X-position of the blue key")
    print(f"  - Y-position of the blue key")
    print(f"  - Frame sequence (time)")
    print("-" * 80)
    
    # Create experiment
    experiment = EntityTrackingExperiment(model_path=args.model_path)
    
    # Load SAEs for all target layers
    print("Loading SAE models for each layer:")
    loaded_layers = []
    for layer_number in experiment.target_layers:
        layer_name = experiment.target_layers[layer_number]
        print(f"  - Attempting to load layer {layer_number} ({layer_name})...")
        success = experiment.load_sae(layer_number, step=args.sae_step)
        if success:
            loaded_layers.append(layer_name)
    
    if not loaded_layers:
        print("ERROR: No SAE models could be loaded. Make sure the checkpoints exist.")
        return
    
    print(f"Successfully loaded {len(loaded_layers)} SAE models: {', '.join(loaded_layers)}")
    print("-" * 80)
    
    # Run the experiment
    print(f"Running experiment with entity1={args.entity1}, entity2={args.entity2}")
    results = experiment.run_entity_tracking_experiment(
        entity1=args.entity1,
        entity2=args.entity2,
        output_path=args.output
    )
    
    # Print top correlating channels for each layer
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for layer_name, layer_results in results.items():
        print(f"\n[Layer {layer_name}]")
        
        print("  Top X-position correlating channels:")
        for i, (channel, corr, p_value) in enumerate(layer_results["x_correlations"][:5]):
            corr_type = "+" if corr > 0 else "-"
            print(f"    {i+1}. Channel {channel}: correlation = {corr:.4f} ({corr_type}), p-value = {p_value:.4f}")
        
        print("\n  Top Y-position correlating channels:")
        for i, (channel, corr, p_value) in enumerate(layer_results["y_correlations"][:5]):
            corr_type = "+" if corr > 0 else "-"
            print(f"    {i+1}. Channel {channel}: correlation = {corr:.4f} ({corr_type}), p-value = {p_value:.4f}")
        
        print("\n  Top frame-sequence correlating channels:")
        for i, (channel, corr, p_value) in enumerate(layer_results["frame_correlations"][:5]):
            corr_type = "+" if corr > 0 else "-"
            print(f"    {i+1}. Channel {channel}: correlation = {corr:.4f} ({corr_type}), p-value = {p_value:.4f}")
    
    # Remove hooks before exiting
    experiment.remove_hooks()
    
    print("\n" + "-" * 80)
    print(f"Results saved to {args.output}")
    print(f"Visualizations saved to {os.path.join(args.output, 'visualizations')}")
    print("\nRECOMMENDED NEXT STEPS:")
    print("1. Review the GIFs in the visualizations directory to see how channels activate")
    print("2. Try running the spatial intervention experiment with the top correlating channels:")
    print("   python run_sae_intervention.py --static --channel <channel_number> --position 4,4 --value 8.0")
    print("=" * 80)

if __name__ == "__main__":
    main() 