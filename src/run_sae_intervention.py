#!/usr/bin/env python
import argparse
import os
from sae_spatial_intervention import SAEInterventionExperiment, create_trajectory_based_intervention
from sae_cnn import ordered_layer_names

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SAE spatial intervention experiments")
    
    # Basic configuration
    parser.add_argument("--model_path", type=str, default="../model_interpretable.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--sae_path", type=str, 
                        default="checkpoints/layer_6_conv3a/sae_checkpoint_step_1000000.pt",
                        help="Path to the SAE checkpoint")
    parser.add_argument("--layer", type=int, default=8,
                        help="Layer number to target (default: 6 for conv3a)")
    parser.add_argument("--layer_name", type=str, default=None,
                        help="Direct layer name (e.g., 'conv3a', 'conv4a') - overrides --layer if provided")
    parser.add_argument("--sae_step", type=int, default=1000000,
                        help="Step number for the SAE checkpoint (default: 1000000)")
    parser.add_argument("--output", type=str, default="intervention_results",
                        help="Directory to save results")
    parser.add_argument("--analyze", action="store_true",
                        help="Run direction channel analysis before interventions")
    
    # Experiment settings
    parser.add_argument("--maze_variant", type=int, default=0,
                        help="Maze variant to use (0-7, default: 0)")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Maximum number of steps to run")
    parser.add_argument("--save_gif_freq", type=int, default=1,
                        help="Save every Nth timestep in activation GIFs (1=all frames, 2=every other frame, etc.)")
    parser.add_argument("--max_gif_frames", type=int, default=0,
                        help="Maximum number of frames to include in GIFs (0=all frames)")
    
    # Intervention configuration
    intervention_group = parser.add_mutually_exclusive_group(required=True)
    intervention_group.add_argument("--baseline", action="store_true",
                                   help="Run baseline without intervention")
    
    intervention_group.add_argument("--static", action="store_true",
                                   help="Run with static intervention")
    parser.add_argument("--channel", type=str, default="95",
                        help="Channel(s) to intervene on (for static intervention). Use comma-separated values for multiple channels, e.g., '95,96,97'")
    parser.add_argument("--position", type=str, default="4,4",
                        help="Position(s) (y,x) to intervene at. For multiple channels, provide comma-separated pairs: '4,4;5,6;3,2'")
    parser.add_argument("--value", type=str, default="5.0",
                        help="Value(s) to set at intervention points. For multiple channels, provide comma-separated values: '5.0,8.0,3.0'")
    parser.add_argument("--radius", type=str, default="1",
                        help="Radius of intervention(s) (0 for single point). For multiple channels, provide comma-separated values: '1,0,2'")
    
    intervention_group.add_argument("--dynamic", action="store_true",
                                   help="Run with dynamic intervention")
    parser.add_argument("--start_pos", type=str, default="4,2",
                        help="Starting position (y,x) for dynamic intervention")
    parser.add_argument("--target_pos", type=str, default="2,6",
                        help="Target position (y,x) for dynamic intervention")
    parser.add_argument("--traj_steps", type=int, default=20,
                        help="Steps to complete trajectory in dynamic intervention")
    
    intervention_group.add_argument("--direction", type=str, 
                                   choices=["right", "left", "up", "down"],
                                   help="Use direction-specific channel")
    
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
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine layer information
    layer_number = args.layer
    if args.layer_name:
        # If layer_name is directly provided, find the corresponding index
        try:
            # Fix: ordered_layer_names is a dictionary, not a list
            # We need to search through its values to find the matching layer name
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
    
    # Handle custom SAE path based on layer name
    sae_path = args.sae_path
    layer_name = ordered_layer_names.get(layer_number)
    
    # Build the path directly from layer_number if available
    if layer_number in ordered_layer_names:
        layer_name = ordered_layer_names[layer_number]
        auto_path = f"checkpoints/layer_{layer_number}_{layer_name}/sae_checkpoint_step_{args.sae_step}.pt"
        if os.path.exists(auto_path):
            sae_path = auto_path
            print(f"Using SAE path: {sae_path} (for layer {layer_name})")
        else:
            print(f"Warning: Could not find SAE checkpoint at {auto_path}")
            print(f"Falling back to provided path: {sae_path}")
    else:
        print(f"Warning: Layer number {layer_number} not found in ordered_layer_names")
        print(f"Using provided path: {sae_path}")
    
    # Create experiment
    print(f"Creating experiment with layer {layer_number} ({layer_name})")
    experiment = SAEInterventionExperiment(
        model_path=args.model_path,
        sae_checkpoint_path=sae_path,
        layer_number=layer_number
    )
    
    # Analyze channels if requested
    direction_channels = None
    if args.analyze:
        print("Analyzing direction-responsive channels...")
        direction_channels = experiment.analyze_activation_patterns(
            maze_variants=[0, 1, 2, 3, 4, 5, 6, 7],
            output_path=args.output
        )
    
    # Configure intervention based on arguments
    if args.baseline:
        print("Running baseline without intervention")
        experiment.disable_intervention()
    
    elif args.static:
        # Parse multi-channel configurations
        config = parse_multi_channels(args)
        channels_str = ', '.join([str(cfg["channel"]) for cfg in config])
        positions_str = ', '.join([str(cfg["position"]) for cfg in config])
        print(f"Running static intervention on channels {channels_str} at positions {positions_str}")
        experiment.set_intervention(config)
    
    elif args.dynamic:
        # Create trajectory-based intervention
        start_pos = parse_position(args.start_pos)
        target_pos = parse_position(args.target_pos)
        print(f"Running dynamic intervention from {start_pos} to {target_pos}")
        
        # Parse channels (support multiple channels)
        channels = list(map(int, args.channel.split(',')))
        
        # Parse values
        if ',' in args.value:
            values = list(map(float, args.value.split(',')))
        else:
            value = float(args.value)
            values = [value] * len(channels)
            
        # Parse radii
        if ',' in args.radius:
            radii = list(map(int, args.radius.split(',')))
        else:
            radius = int(args.radius)
            radii = [radius] * len(channels)
        
        # Ensure all lists have the same length
        n_channels = len(channels)
        if len(values) < n_channels:
            values.extend([values[-1]] * (n_channels - len(values)))
        if len(radii) < n_channels:
            radii.extend([radii[-1]] * (n_channels - len(radii)))
            
        # Truncate if needed
        values = values[:n_channels]
        radii = radii[:n_channels]
        
        # Create dynamic interventions for each channel
        dynamic_interventions = []
        for i in range(n_channels):
            dynamic_interventions.append(create_trajectory_based_intervention(
                channel=channels[i],
                start_pos=start_pos,
                target_pos=target_pos,
                max_steps=args.traj_steps,
                value=values[i],
                radius=radii[i]
            ))
        
        # Combine interventions
        def combined_intervention(step, original_activations, modified_activations):
            result = modified_activations.clone()
            for intervention in dynamic_interventions:
                result = intervention(step, original_activations, result)
            return result
            
        experiment.set_dynamic_intervention(combined_intervention)
    
    elif args.direction:
        # First make sure we have direction channels
        if direction_channels is None:
            direction_channels = experiment.analyze_activation_patterns(
                maze_variants=[0, 1, 2, 3, 4, 5, 6, 7],
                output_path=args.output
            )
        
        # Get top channel for the specified direction
        if args.direction in direction_channels and direction_channels[args.direction]:
            channel = direction_channels[args.direction][0]["channel"]
        else:
            print(f"No channels found for direction '{args.direction}', falling back to channel {args.channel.split(',')[0]}")
            channel = int(args.channel.split(',')[0])
        
        # Configure intervention based on direction
        position_map = {
            "right": (4, 6),
            "left": (4, 2),
            "up": (2, 4),
            "down": (6, 4)
        }
        position = position_map[args.direction]
        # Use value and radius from args (first value if multiple)
        value = float(args.value.split(',')[0])
        radius = int(args.radius.split(',')[0]) 
        config = [{"channel": channel, "position": position, "value": value, "radius": radius}]
        print(f"Running {args.direction} intervention on channel {channel} at position {position}")
        experiment.set_intervention(config)
    
    # Run experiment
    result = experiment.run_maze_experiment(
        maze_variant=args.maze_variant,
        max_steps=args.max_steps,
        save_gif=True,
        output_path=args.output,
        save_gif_freq=args.save_gif_freq,
        max_gif_frames=args.max_gif_frames
    )
    
    # Print results
    print("\nExperiment Results:")
    print(f"Total steps: {result['total_steps']}")
    print(f"Total reward: {result['total_reward']}")
    print(f"Intervention type: {result['intervention_type']}")
    
    return result

if __name__ == "__main__":
    main() 