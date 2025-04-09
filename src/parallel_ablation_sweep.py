# src/parallel_ablation_sweep.py

import argparse
import subprocess
import os
import math
from multiprocessing import Pool, cpu_count
import datetime

# --- Defaults (Match channel_ablation_sweep.py where applicable) ---
DEFAULT_MODEL_PATH = "../model_interpretable.pt"
DEFAULT_OUTPUT_DIR = "channel_ablation_results_parallel"
DEFAULT_MAX_STEPS = 300
DEFAULT_NUM_TRIALS = 5
DEFAULT_TOTAL_CHANNELS = 128 # Default assuming the common SAE size
# ---

def parse_args():
    """Parse command line arguments for the parallel runner"""
    parser = argparse.ArgumentParser(description="Run channel ablation sweep experiment in parallel.")

    # Arguments mirroring channel_ablation_sweep.py (excluding channel ranges)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the base interpretable model.")
    parser.add_argument("--sae_checkpoint_path", type=str, required=True, help="Path to the SAE checkpoint (.pt file).")
    parser.add_argument("--layer_number", type=int, required=True, help="SAE layer number (e.g., 8 for conv4a).")
    parser.add_argument("--num_trials", type=int, default=DEFAULT_NUM_TRIALS, help="Number of trials per channel.")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum steps per episode.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base directory to save results CSVs and GIFs.")
    parser.add_argument("--save_gifs", action="store_true", help="Save a GIF for the first trial of each channel (within each process).")

    # Arguments specific to the parallel runner
    parser.add_argument("--num_processes", type=int, default=min(16, cpu_count()), help="Number of parallel processes to launch.")
    parser.add_argument("--total_channels", type=int, default=DEFAULT_TOTAL_CHANNELS, help="Total number of channels in the SAE layer.")
    parser.add_argument("--worker_script", type=str, default="channel_ablation_sweep.py", help="Path to the worker script.")


    args = parser.parse_args()

    if not os.path.exists(args.model_path):
         parser.error(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.sae_checkpoint_path):
         parser.error(f"SAE checkpoint path not found: {args.sae_checkpoint_path}")
    if not os.path.exists(args.worker_script):
         parser.error(f"Worker script not found: {args.worker_script}")

    # Create a unique subdirectory for this parallel run based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_sae_name = os.path.basename(args.sae_checkpoint_path).replace('.pt', '')
    args.output_dir = os.path.join(args.output_dir, f"run_{safe_sae_name}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved in: {args.output_dir}")

    return args

def run_worker(worker_args):
    """Function to be executed by each process."""
    process_id, start_channel, end_channel, common_args = worker_args

    command = [
        "python",
        common_args.worker_script,
        "--model_path", common_args.model_path,
        "--sae_checkpoint_path", common_args.sae_checkpoint_path,
        "--layer_number", str(common_args.layer_number),
        "--num_trials", str(common_args.num_trials),
        "--max_steps", str(common_args.max_steps),
        "--output_dir", common_args.output_dir, # Use the timestamped dir
        "--start_channel", str(start_channel),
        "--end_channel", str(end_channel),
    ]
    if common_args.save_gifs:
        command.append("--save_gifs")

    print(f"[Process {process_id}] Running channels {start_channel}-{end_channel-1}...")
    print(f"[Process {process_id}] Command: {' '.join(command)}")

    try:
        # Run the subprocess, CAPTURING output again for error diagnosis
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[Process {process_id}] Completed successfully.")
        # We won't see regular output here, only if it finishes.
        # For debugging success, uncomment below (but it might be verbose)
        # print(f"[Process {process_id} STDOUT]:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"[Process {process_id}] FAILED!")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return Code: {e.returncode}")
        # Print captured output which might contain the error message
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
    except Exception as e:
         print(f"[Process {process_id}] FAILED with unexpected error: {e}")

def main():
    args = parse_args()

    num_processes = args.num_processes
    total_channels = args.total_channels

    if num_processes <= 0:
        print("Number of processes must be positive.")
        return
    if total_channels <= 0:
        print("Total channels must be positive.")
        return

    # Calculate channel chunks for each process
    chunk_size = total_channels // num_processes
    remainder = total_channels % num_processes

    tasks = []
    current_channel = 0
    for i in range(num_processes):
        start = current_channel
        size = chunk_size + (1 if i < remainder else 0)
        end = start + size
        if start >= total_channels: # Avoid creating tasks if no channels left
            break
        end = min(end, total_channels) # Ensure end doesn't exceed total
        tasks.append((i, start, end, args))
        current_channel = end

    if not tasks:
        print("No tasks generated (total_channels might be 0 or less).")
        return

    print(f"Launching {len(tasks)} processes to cover {total_channels} channels.")

    # Use multiprocessing Pool to run tasks
    with Pool(processes=len(tasks)) as pool:
        pool.map(run_worker, tasks)

    print("\nParallel execution finished.")
    print(f"Results saved in subdirectories/files within: {args.output_dir}")
    print("You may now need to combine the CSV files from that directory for analysis.")

if __name__ == "__main__":
    main()