# src/parallel_ablation_sweep.py

import argparse
import subprocess
import os
import math
from multiprocessing import Pool, cpu_count
import datetime

# --- Defaults ---
default_model_path = "../model_interpretable.pt"
default_output_dir = "channel_ablation_results_parallel"
default_max_steps = 100
default_num_trials = 5
# Cannot easily default total_channels anymore, must be provided if base layer
# ---

def parse_args():
    """Parse command line arguments for the parallel runner"""
    parser = argparse.ArgumentParser(description="Run channel ablation sweep experiment in parallel.")

    # Arguments mirroring channel_ablation_sweep.py (excluding channel ranges)
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the base interpretable model.")
    parser.add_argument("--sae_checkpoint_path", type=str, default=None, help="Path to the SAE checkpoint (.pt file). If omitted, runs base model ablation.") # Optional
    parser.add_argument("--layer_spec", type=str, required=True, help="SAE layer number (int) OR base model layer name (str).") # Changed
    parser.add_argument("--num_trials", type=int, default=default_num_trials, help="Number of trials per channel.")
    parser.add_argument("--max_steps", type=int, default=default_max_steps, help="Maximum steps per episode.")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Base directory to save results CSVs and GIFs.")
    parser.add_argument("--no_save_gifs", action="store_true", help="Disable saving a GIF for the first trial of each channel (within each process).")

    # Arguments specific to the parallel runner
    parser.add_argument("--num_processes", type=int, default=min(16, cpu_count()), help="Number of parallel processes to launch.")
    parser.add_argument("--total_channels", type=int, required=True, help="Total number of channels in the target layer (SAE or Base). Required for parallel chunking.") # Required now
    parser.add_argument("--worker_script", type=str, default="channel_ablation_sweep.py", help="Path to the worker script.")
    parser.add_argument("--maze_type", type=str, default="fork", choices=["fork", "corners"], help="Type of maze to use for the experiment.")
    parser.add_argument("--bias_direction", type=str, default=None, choices=["up", "down", "left", "right"], help="Bias direction for maze orientation (optional).")


    args = parser.parse_args()
    args.is_sae_run = args.sae_checkpoint_path is not None

    # --- Validation ---
    if not os.path.exists(args.model_path):
         parser.error(f"Model path not found: {args.model_path}")
    if args.is_sae_run and not os.path.exists(args.sae_checkpoint_path):
         parser.error(f"SAE checkpoint path specified but not found: {args.sae_checkpoint_path}")
    if not os.path.exists(args.worker_script):
         parser.error(f"Worker script not found: {args.worker_script}")
    if args.total_channels <= 0:
         parser.error("Total channels must be positive.")
    # Basic validation of layer_spec format based on mode
    if args.is_sae_run:
        try: int(args.layer_spec) # Check if it can be an int
        except ValueError: parser.error("If --sae_checkpoint_path is provided, --layer_spec must be an integer.")
    else:
        if not isinstance(args.layer_spec, str) or args.layer_spec.isdigit():
             parser.error("If --sae_checkpoint_path is NOT provided, --layer_spec must be a non-numeric string layer name.")
    # ---

    # Create a unique subdirectory for this parallel run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_type = "sae" if args.is_sae_run else "base"
    safe_layer_spec = args.layer_spec.replace('.', '_')
    dir_name_parts = [f"run_{args.maze_type}_{run_type}_{safe_layer_spec}"]
    if args.is_sae_run:
        safe_sae_name = os.path.basename(args.sae_checkpoint_path).replace('.pt', '')
        dir_name_parts.append(safe_sae_name) # Add SAE name for clarity
    dir_name_parts.append(timestamp)
    
    args.output_dir = os.path.join(args.output_dir, "_".join(dir_name_parts))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved in: {args.output_dir}")

    # Store run_type on args for later use
    args.run_type = "sae" if args.is_sae_run else "base"

    return args

def save_run_arguments(args_obj, output_dir):
    """Saves the run arguments to a text file in the output directory."""
    args_file_path = os.path.join(output_dir, "parallel_run_arguments.txt")
    try:
        with open(args_file_path, 'w') as f:
            f.write(f"Run Timestamp: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write("Command Line Arguments Used for Parallel Run:\n")
            for arg, value in sorted(vars(args_obj).items()): # Sort for consistent order
                f.write(f"  --{arg}: {value}\n")
        print(f"Saved parallel run arguments to: {args_file_path}")
    except Exception as e:
        print(f"Error saving run arguments: {e}")

def run_worker(worker_args):
    """Function to be executed by each process."""
    process_id, start_channel, end_channel, common_args = worker_args

    command = [
        "python",
        common_args.worker_script,
        "--model_path", common_args.model_path,
        "--layer_spec", common_args.layer_spec, # Pass layer_spec
        "--num_trials", str(common_args.num_trials),
        "--max_steps", str(common_args.max_steps),
        "--output_dir", common_args.output_dir,
        "--start_channel", str(start_channel),
        "--end_channel", str(end_channel),
        "--maze_type", common_args.maze_type,
    ]
    
    # Add bias_direction if provided
    if common_args.bias_direction:
        command.extend(["--bias_direction", common_args.bias_direction])

    # Conditionally add SAE path or total_channels_for_base_layer
    if common_args.is_sae_run:
        if common_args.sae_checkpoint_path: # Should always be true if is_sae_run
            command.extend(["--sae_checkpoint_path", common_args.sae_checkpoint_path])
        else:
            # This case should ideally be prevented by argument parsing in the main script
            print(f"[Process {process_id}] Error: SAE run indicated but no SAE checkpoint path provided to parallel runner.")
            return # Or raise an error
    else: # Base model run
        # common_args.total_channels is required by the parallel script's arg parser
        command.extend(["--total_channels_for_base_layer", str(common_args.total_channels)])

    if common_args.no_save_gifs:
        command.append("--no_save_gifs")

    print(f"[Process {process_id}] Running channels {start_channel}-{end_channel-1}...")
    print(f"[Process {process_id}] Command: {' '.join(command)}")

    try:
        # Run the subprocess, capturing output for error diagnosis
        result = subprocess.run(command, check=True, text=True)
        print(f"[Process {process_id}] Completed successfully.")
        # Optional debug print
        # print(f"[Process {process_id} STDOUT]:\n{result.stdout}") # No longer captured if capture_output is False

    except subprocess.CalledProcessError as e:
        print(f"[Process {process_id}] FAILED!")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return Code: {e.returncode}")
        # Stderr might still be available on the exception object even if not captured by default
        print(f"  Stderr: {e.stderr}") 
    except Exception as e:
         print(f"[Process {process_id}] FAILED with unexpected error: {e}")

def main():
    args = parse_args()

    # Save the arguments after output_dir is finalized in parse_args
    save_run_arguments(args, args.output_dir)

    num_processes = args.num_processes
    total_channels = args.total_channels # Now required arg

    if num_processes <= 0:
        print("Number of processes must be positive.")
        return
    # total_channels check already done in parse_args

    # Calculate channel chunks for each process
    chunk_size = total_channels // num_processes
    remainder = total_channels % num_processes

    tasks = []
    current_channel = 0
    for i in range(num_processes):
        start = current_channel
        size = chunk_size + (1 if i < remainder else 0)
        end = start + size
        if start >= total_channels: break
        end = min(end, total_channels)
        tasks.append((i, start, end, args))
        current_channel = end

    if not tasks:
        print("No tasks generated.")
        return

    print(f"Launching {len(tasks)} processes to cover {total_channels} channels.")

    # Use multiprocessing Pool to run tasks
    with Pool(processes=len(tasks)) as pool:
        pool.map(run_worker, tasks)

    print("\nParallel execution finished.")
    print(f"Results saved in subdirectories/files within: {args.output_dir}")
    # print("You may now need to combine the CSV files from that directory for analysis.") # Plotting now runs automatically

    # --- Automatically run the plotting script --- 
    plot_script_path = "plot_ablation_results.py" 
    if os.path.exists(plot_script_path):
        print(f"\nAttempting to generate plots using {plot_script_path}...")
        # Simplified title prefix
        plot_title_prefix = f"{args.run_type}_{args.layer_spec}_{args.maze_type}"
        plot_command = [
            "python",
            plot_script_path,
            "--results_dir", args.output_dir,
            "--output_dir", args.output_dir, # Save plots in the same run directory
            "--plot_title_prefix", plot_title_prefix
        ]
        print(f"Plotting command: {' '.join(plot_command)}")
        try:
            subprocess.run(plot_command, check=True, text=True)
            print("Plotting script finished.")
        except FileNotFoundError:
            print(f"Error: Plotting script not found at {plot_script_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running plotting script (Return Code: {e.returncode}):")
            # stderr might contain useful info if the script failed
            if e.stderr:
                 print(f"Plotting Script Stderr:\n{e.stderr}")
            if e.stdout:
                 print(f"Plotting Script Stdout:\n{e.stdout}")
        except Exception as e:
            print(f"An unexpected error occurred while running the plotting script: {e}")
    else:
        print(f"\nWarning: Plotting script not found at {plot_script_path}. Skipping automatic plot generation.")
    # ---

if __name__ == "__main__":
    main()