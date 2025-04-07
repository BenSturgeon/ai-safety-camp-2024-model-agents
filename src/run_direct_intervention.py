#!/usr/bin/env python
import argparse
import os
from direct_intervention import DirectInterventionExperiment
from intervention_base import create_trajectory_based_intervention
from sae_cnn import ordered_layer_names
import imageio

from procgen import ProcgenEnv
from procgen.gym_registration import make_env

from utils.environment_modification_experiments import create_box_maze

ENTITY_CODE_DESCRIPTION = {
    3: "gem",
    4: "blue_key",
    5: "green_key",
    6: "red_key",
    7: "blue_lock",
    8: "green_lock",
    9: "red_lock"
}
ENTITY_DESCRIPTION_CODE = {v: k for k, v in ENTITY_CODE_DESCRIPTION.items()}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run direct model layer intervention experiments")
    
    # Basic configuration
    parser.add_argument("--model_path", type=str, default="../model_interpretable.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--layer", type=int, default=8,
                        help="Layer number to target (default: 8 for conv4a)")
    parser.add_argument("--layer_name", type=str, default=None,
                        help="Direct layer name (e.g., 'conv3a', 'conv4a') - overrides --layer if provided")
    parser.add_argument("--output", type=str, default="direct_intervention_results",
                        help="Directory to save results")
    parser.add_argument("--analyze", action="store_true",
                        help="Run direction channel analysis before interventions")
    
    # Entity configuration
    parser.add_argument("--entity1_code", type=int, default=4, choices=[3,4,5,6,7,8,9],
                        help="Primary entity code to use in the maze (default: 4, blue key)")
    parser.add_argument("--entity2_code", type=int, default=None, choices=[None,3,4,5,6,7,8,9],
                        help="Secondary entity code (optional)")
    
    # Experiment settings
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Maximum number of steps to run")
    parser.add_argument("--save_gif_freq", type=int, default=1,
                        help="Save every Nth timestep in activation GIFs (1=all frames, 2=every other frame, etc.)")
    parser.add_argument("--max_gif_frames", type=int, default=0,
                        help="Maximum number of frames to include in GIFs (0=all frames)")
    
    # Intervention configuration
    parser.add_argument("--channel", type=str, default="95",
                        help="Channel(s) to intervene on. Use comma-separated values for multiple channels, e.g., '95,96,97'")
    parser.add_argument("--position", type=str, default="4,4",
                        help="Position(s) (y,x) to intervene at. For multiple channels, provide comma-separated pairs: '4,4;5,6;3,2'")
    parser.add_argument("--value", type=str, default="5.0",
                        help="Value(s) to set at intervention points. For multiple channels, provide comma-separated values: '5.0,8.0,3.0'")
    parser.add_argument("--radius", type=str, default="1",
                        help="Radius of intervention(s) (0 for single point). For multiple channels, provide comma-separated values: '1,0,2'")
    
    # Environment type
    parser.add_argument("--box_maze", action="store_true",
                        help="Use box maze environment (with specific entities) instead of default L-shaped Procgen maze")
    
    return parser.parse_args()

def parse_position(pos_str):
    """Parse position string 'y,x' into tuple (y, x)"""
    try:
        y, x = map(int, pos_str.split(','))
        return (y, x)
    except:
        raise ValueError(f"Invalid position format: {pos_str}. Use 'y,x' format.")

def parse_multi_channels(args):
    """Parse arguments for multiple channel interventions"""
    # Parse channels
    channels = list(map(int, args.channel.split(',')))
    
    # Parse positions
    if ';' in args.position:
        # Multiple positions provided
        positions = [parse_position(pos) for pos in args.position.split(';')]
    else:
        # Single position, replicate for all channels
        pos = parse_position(args.position)
        positions = [pos] * len(channels)
    
    # Parse values
    if ',' in args.value:
        # Multiple values provided
        values = list(map(float, args.value.split(',')))
    else:
        # Single value, replicate for all channels
        value = float(args.value)
        values = [value] * len(channels)
    
    # Parse radii
    if ',' in args.radius:
        # Multiple radii provided
        radii = list(map(int, args.radius.split(',')))
    else:
        # Single radius, replicate for all channels
        radius = int(args.radius)
        radii = [radius] * len(channels)
    
    # Ensure all lists have the same length
    n_channels = len(channels)
    if len(positions) < n_channels:
        positions.extend([positions[-1]] * (n_channels - len(positions)))
    if len(values) < n_channels:
        values.extend([values[-1]] * (n_channels - len(values)))
    if len(radii) < n_channels:
        radii.extend([radii[-1]] * (n_channels - len(radii)))
    
    # Truncate lists if needed
    positions = positions[:n_channels]
    values = values[:n_channels]
    radii = radii[:n_channels]
    
    # Create configuration list
    config = []
    for i in range(n_channels):
        config.append({
            "channel": channels[i],
            "position": positions[i],
            "value": values[i],
            "radius": radii[i]
        })
    
    return config

def main():
    """Main function to run experiments"""
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Determine layer information
    layer_number = args.layer
    if args.layer_name:
        # If layer_name is directly provided, find the corresponding index
        try:
            # Search through layer names to find matching layer
            matching_layers = [(idx, name) for idx, name in ordered_layer_names.items() 
                              if name == args.layer_name or name.endswith(args.layer_name)]
            
            if matching_layers:
                layer_number, layer_name = matching_layers[0]
                print(f"Using layer {layer_name} (layer number {layer_number})")
            else:
                # If no exact match, show available layers
                raise IndexError("No matching layer found")
        except IndexError:
            print(f"Warning: Could not find layer with name '{args.layer_name}'. Valid layers are:")
            for idx, name in ordered_layer_names.items():
                print(f"  {idx}: {name}")
            print(f"Falling back to layer number {layer_number}")
    
    # Create experiment
    print(f"Creating DirectInterventionExperiment with layer {layer_number}")
    experiment = DirectInterventionExperiment(
        model_path=args.model_path,
        layer_number=layer_number
    )
    
    # Parse multi-channel configurations
    config = parse_multi_channels(args)
        
    # Configure intervention
    channels_str = ', '.join([str(cfg["channel"]) for cfg in config])
    positions_str = ', '.join([str(cfg["position"]) for cfg in config])
    values_str = ', '.join([str(cfg["value"]) for cfg in config])
    
    print(f"Running intervention on channels {channels_str} at positions {positions_str} with values {values_str}")
    
    # Set intervention configuration
    experiment.set_intervention(config)
    
    # Verify that intervention was set properly
    print(f"Intervention active: {experiment.intervention_active}")
    if experiment.intervention_config:
        print(f"Configured {len(experiment.intervention_config)} channel interventions:")
        for i, cfg in enumerate(experiment.intervention_config):
            ch = cfg["channel"]
            pos = cfg["position"]
            val = cfg.get("value", 0.0)
            rad = cfg.get("radius", 0)
            print(f"  {i+1}. Channel {ch}: zero out and set value {val} at position {pos} with radius {rad}")
    
    # --- Environment Creation --- 
    venv = None
    try:
        # if args.box_maze:
        print(f"Creating box maze with entity codes: primary={args.entity1_code}, secondary={args.entity2_code}")
        env_args = {"entity1": args.entity1_code, "entity2": args.entity2_code}
        _, venv = create_box_maze(**env_args) # Get the venv object
        
        # Map entity code to description for better output file naming
        entity_desc = ENTITY_CODE_DESCRIPTION.get(args.entity1_code, f"entity_{args.entity1_code}")
        
        # Create entity-specific output subdirectory
        entity_output = os.path.join(args.output, f"entity_{args.entity1_code}_{entity_desc}")
        os.makedirs(entity_output, exist_ok=True)
        output_path = entity_output # Use this path for results/gifs

        # --- Prepare Options for run_maze_experiment ---
        options = {
            # Removed maze_variant as it's handled by env creation
            "venv": venv, # Pass the created environment
            "max_steps": args.max_steps,
            "save_gif": True,
            "output_path": output_path, # Use determined output path
            "save_gif_freq": args.save_gif_freq,
            "max_gif_frames": args.max_gif_frames
            # Removed entity codes and use_box_maze, handled above
        }
        # --- End Options Preparation ---
        
        # Run experiment with the appropriate options
        result = experiment.run_maze_experiment(**options)
        
        # Print results
        print("\nExperiment Results:")
        print(f"Total steps: {result['total_steps']}")
        print(f"Total reward: {result['total_reward']}")
        print(f"Intervention type: {result['intervention_type']}")
        
        # Print entity information if using box maze
        if args.box_maze:
            print(f"Primary entity: {ENTITY_CODE_DESCRIPTION.get(args.entity1_code, args.entity1_code)}")
            if args.entity2_code:
                print(f"Secondary entity: {ENTITY_CODE_DESCRIPTION.get(args.entity2_code, args.entity2_code)}")
        else:
            print("Using default Procgen maze.")
        
        print(f"Output saved to: {os.path.abspath(options['output_path'])}")
        
        return result

    finally:
        # Ensure environment is closed
        if venv is not None:
            print("Closing environment...")
            venv.close()

if __name__ == "__main__":
    main() 