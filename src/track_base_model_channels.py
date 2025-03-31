#!/usr/bin/env python
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import csv
from tqdm import tqdm

# Import local modules
from utils.environment_modification_experiments import create_box_maze
from utils import helpers

# Import analysis functions from track_object_channels.py
from track_object_channels import EntityTrackingExperiment

entity_code_description = {
    3: "gem",
    4: "blue_key",
    5: "green_key",
    6: "red_key",
    7: "blue_lock",
    8: "green_lock",
    9: "red_lock"
}

class BaseModelTrackingExperiment:
    """Experiment to track base model channels that correlate with entity positions."""
    
    def __init__(self, target_layers=None):
        """
        Initialize the experiment.
        
        Args:
            target_layers (list): List of layer names to analyze
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = helpers.load_interpretable_model(model_path="../model_interpretable.pt")
        self.model.eval()
        self.model = self.model.to(self.device)
        
        if target_layers is None:
            self.target_layers = ['conv1a', 'conv2a', 'conv2b', 'conv3a', 'conv4a']
        else:
            self.target_layers = target_layers
            
        self.hooks = []
        self.all_activations = {layer: [] for layer in self.target_layers}
        
        # Create a reference to the EntityTrackingExperiment for using its analysis methods
        self.entity_tracker = EntityTrackingExperiment(model_path="../model_interpretable.pt")
    
    def _get_module(self, name):
        """Get a named module from the model."""
        return dict(self.model.named_modules())[name]
    
    def _hook_layer_activations(self, name):
        """Create a hook function for the given layer."""
        def hook_fn(module, input, output):
            # Store the output activations
            # For convolutional layers, output shape is [batch, channels, height, width]
            self.all_activations[name].append(output.detach())
        return hook_fn
    
    def register_hooks(self):
        """Register hooks for all target layers."""
        for layer_name in self.target_layers:
            module = self._get_module(layer_name)
            hook = module.register_forward_hook(self._hook_layer_activations(layer_name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def reset_activations(self):
        """Reset stored activations."""
        self.all_activations = {layer: [] for layer in self.target_layers}

    def run_entity_tracking_experiment(self, entity1_code=4, entity2_code=None, output_path="base_model_tracking_results"):
        """
        Run an experiment to track how base model activations correlate with entity positions.
        
        Args:
            entity1_code (int): Code of primary entity to track (default: 4, blue key)
            entity2_code (int): Code of secondary entity (optional)
            output_path (str): Directory to save results
        
        Returns:
            dict: Results showing which channels correlate with entity positions
        """

        
        entity_desc_filename = entity_code_description.get(entity1_code, f"entity_{entity1_code}")
        entity_output_path = output_path
        os.makedirs(entity_output_path, exist_ok=True)
        
        print(f"Results will be saved to: {os.path.abspath(entity_output_path)}")
        
        self.reset_activations()
        
        if not self.hooks:
            self.register_hooks()
        
        print(f"Creating box maze with entity codes: primary={entity1_code}, secondary={entity2_code}")
        observations, venv = create_box_maze(entity1=entity1_code, entity2=entity2_code)
        
        # Known positions for the entity in the box maze
        # Format: (y, x) - y is row, x is column (0-indexed)
        known_positions = [
            (2, 1),  # Starting position in the box maze
            (2, 2),  # Moving right
            (2, 3),  # Center area
            (2, 4),  # Continuing right
            (2, 5),  # Far right
            (3, 5),  # Moving down
            (4, 5),  # Bottom right
            (4, 4),  # Moving left
            (4, 3),  # Bottom center
            (4, 2),  # Continuing left
            (4, 1),  # Bottom left
            (3, 1)   # Moving up
        ]
        
        entity_positions = []
        entity_x_positions = []
        entity_y_positions = []
        
        print("Processing observations...")
        for i, obs in enumerate(tqdm(observations)):
            entity_positions.append(i)
            
            if i < len(known_positions):
                y_pos, x_pos = known_positions[i]
            else:
                y_pos, x_pos = 3, 3
                
            entity_x_positions.append(x_pos)
            entity_y_positions.append(y_pos)
            
            # Convert observation to the format expected by the model
            converted_obs = helpers.observation_to_rgb(obs)
            obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(self.device)
            
            # Add batch dimension if needed
            if obs_tensor.ndim == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            with torch.no_grad():
                self.model(obs_tensor)
        
        venv.close()
        
        position_file = os.path.join(entity_output_path, f"{entity_desc_filename}_positions.csv")
        with open(position_file, "w") as f:
            f.write("frame,x,y\n")
            for frame, x, y in zip(entity_positions, entity_x_positions, entity_y_positions):
                f.write(f"{frame},{x},{y}\n")
        
        entity_display_desc = entity_code_description.get(entity1_code, f"entity {entity1_code}")
        entity_display_desc = entity_display_desc.replace("_", " ")
        
        entity_info = {
            "entity_code": entity1_code,
            "secondary_code": entity2_code,
            "description": entity_display_desc,
            "filename_desc": entity_desc_filename
        }
        
        
        print("Reformatting activations for analysis...")
        
        for layer_name, activations_list in self.all_activations.items():
            if not activations_list:
                continue
                
            print(f"Layer {layer_name} original shape: {activations_list[0].shape}")
            
            if len(activations_list[0].shape) != 4: 
                print(f"Skipping layer {layer_name} - unexpected shape")
                continue
                
            # For each frame's activation:
            # - Squeeze out batch dimension to get [channels, height, width]
            # - Ensure it's on CPU for analysis
            processed_activations = []
            for act in activations_list:
                # Remove batch dimension (our batch size is 1)
                act = act.squeeze(0).cpu()
                processed_activations.append(act)
                
            # Make sure the layer exists in the entity_tracker activations dict
            if layer_name not in self.entity_tracker.all_activations:
                print(f"Creating activations entry for layer {layer_name}")
                self.entity_tracker.all_activations[layer_name] = []
                
            self.entity_tracker.all_activations[layer_name] = processed_activations
            
            sample_act = processed_activations[0]
            print(f"Processed activation shape for {layer_name}: {sample_act.shape} (frames: {len(processed_activations)})")
                

        self.entity_tracker.target_layers = list(self.entity_tracker.all_activations.keys())
        
        print("Running analysis using EntityTrackingExperiment methods...")
        

        results = self.entity_tracker.analyze_correlations_with_positions(
            entity_positions, 
            entity_x_positions, 
            entity_y_positions, 
            entity_output_path, 
            entity_info
        )
        
        # Use the imported visualization function
        self.entity_tracker.visualize_results(
            observations,
            entity_positions, 
            results, 
            entity_output_path, 
            entity_info
        )
        

        self._print_results_summary(results, entity_output_path, entity_info)
        
        return results
        
    def _print_results_summary(self, results, output_path, entity_info):
        """Print a summary of the experiment results"""
        entity_desc = entity_info['description']
        entity_code = entity_info['entity_code']
        
        print("\n" + "=" * 80)
        print(f"RESULTS SUMMARY FOR {entity_desc.upper()} (CODE {entity_code})")
        print("=" * 80)
        

        for layer_name, layer_results in results.items():
            print(f"\nLayer {layer_name} Top Results:")
            
            metrics = [
                ("IOU (Spatial Overlap)", "iou_scores"),
                ("TopK Ratio", "topk_ratios"),
                ("SNR", "snr_scores"),
                ("Equivariance", "equivariance_scores")
            ]
            
            for metric_name, metric_key in metrics:
                if metric_key in layer_results and layer_results[metric_key]:
                    top_channel, score = layer_results[metric_key][0]
                    score_fmt = f"{abs(score):.4f}"
                    if metric_key == "equivariance_scores" and score < 0:
                        score_fmt = f"-{score_fmt}"
                    print(f"  Top {metric_name}: Channel {top_channel} = {score_fmt}")
        
        print("\n" + "-" * 80)
        print(f"Detailed results saved to: {os.path.abspath(output_path)}")
        print(f"  - Analysis report: {entity_info['filename_desc']}_detailed_results.txt")
        print(f"  - Visualizations in: {output_path}/visualizations/")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track entity correlations in base model activations")
    parser.add_argument("--entity1_code", type=int, default=5, help="Entity code for primary entity (default: 5 for green key)")
    parser.add_argument("--entity2_code", type=int, default=None, help="Entity code for secondary entity (optional)")
    parser.add_argument("--output", type=str, default="output/base_model_tracking", help="Output directory")
    parser.add_argument("--layers", type=str, default="conv1a,conv2a,conv2b,conv3a,conv4a", 
                        help="Comma-separated list of layers to analyze")
    
    args = parser.parse_args()
    
    # Convert layers string to list
    target_layers = args.layers.split(",")
    
    # Create and run the experiment
    experiment = BaseModelTrackingExperiment(target_layers=target_layers)
    
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    
    entity1_desc = entity_code_description.get(args.entity1_code, f"entity{args.entity1_code}")
    entity2_desc = entity_code_description.get(args.entity2_code, f"entity{args.entity2_code}") if args.entity2_code else None
    
    entity_output = os.path.join(output_path, entity1_desc)
    os.makedirs(entity_output, exist_ok=True)
    
    print(f"Starting tracking experiment for {entity1_desc} (code {args.entity1_code})")
    if args.entity2_code:
        print(f"Secondary entity: {entity2_desc} (code {args.entity2_code})")
    
    # Run the experiment
    experiment.run_entity_tracking_experiment(
        entity1_code=args.entity1_code,
        entity2_code=args.entity2_code,
        output_path=entity_output
    ) 