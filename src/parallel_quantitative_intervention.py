import argparse
import subprocess
import os
import math
from multiprocessing import Pool, cpu_count
import datetime

# --- Defaults ---
DEFAULT_MODEL_PATH = "../model_interpretable.pt"
DEFAULT_SAE_CHECKPOINT_DIR = "checkpoints_of_interest" # Copied from quantitative_intervention_experiment
DEFAULT_OUTPUT_DIR = "quantitative_intervention_results_parallel"
DEFAULT_MAX_STEPS = 20 # From quantitative_intervention_experiment
DEFAULT_NUM_TRIALS = 10 # From quantitative_intervention_experiment
DEFAULT_WORKER_SCRIPT = "quantitative_intervention_experiment.py"


def parse_args():
    """Parse command line arguments for the parallel runner"""
    parser = argparse.ArgumentParser(description="Run quantitative intervention experiments in parallel across channels.")

    # Arguments from quantitative_intervention_experiment.py that are relevant for the parallel runner
    # Model and Layer Configuration
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the base model checkpoint.")
    parser.add_argument("--model_base_dir", type=str, default="models", # from quantitative_intervention_experiment
                        help="Base directory to search for the latest model checkpoint if --model_path is not specified.")
    parser.add_argument("--sae_checkpoint_path", type=str, default=None,
                        help=f"Path to the SAE checkpoint. If --is_sae is set and this is omitted, worker will try to find the latest.")
    parser.add_argument("--layer_spec", type=str, required=True,
                        help="Layer specification: either base model layer name (e.g., 'conv_seqs.2.res_block1.conv1') or SAE layer number (e.g., '18')")
    parser.add_argument("--is_sae", action="store_true",
                        help="Flag indicating the layer_spec refers to an SAE layer number")

    # Experiment Parameters (from quantitative_intervention_experiment.py)
    parser.add_argument("--target_entities", type=str, default="gem,blue_key,green_key,red_key",
                        help="Comma-separated list of entity names to target (e.g., 'gem,blue_key')")
    parser.add_argument("--num_trials", type=int, default=DEFAULT_NUM_TRIALS,
                        help="Number of trials per channel")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Maximum number of steps per trial simulation")
    parser.add_argument("--intervention_position", type=str, default=None,
                        help="Intervention position as 'y,x'. Worker will use its defaults if None.")
    parser.add_argument("--intervention_radius", type=int, default=1,
                        help="Radius for the intervention patch (0 for single point)")
    parser.add_argument("--intervention_value", type=str, default="0,0.7",
                        help="Intervention value. Either a fixed float (e.g., '3.0') or a range 'min,max' (e.g., '0,3').")
    
    # Output Configuration (base output_dir for the parallel run)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Base directory to save results. Each run will create a subdirectory here.")
    # num_top_channels_visualize is handled by the worker script individually.

    # Arguments specific to the parallel runner
    parser.add_argument("--num_processes", type=int, default=min(8, cpu_count()), # Adjusted default
                        help="Number of parallel processes to launch.")
    parser.add_argument("--total_channels", type=int, required=True,
                        help="Total number of channels in the target layer (SAE or Base). Required for parallel chunking.")
    parser.add_argument("--worker_script", type=str, default=DEFAULT_WORKER_SCRIPT,
                        help="Path to the worker script (quantitative_intervention_experiment.py).")

    args = parser.parse_args()

    # --- Validation ---
    if not os.path.exists(args.model_path) and not args.model_base_dir : # Worker will search if model_path is not given but base_dir is
        parser.error(f"Model path {args.model_path} not found and no model_base_dir specified for worker to search.")
    if args.is_sae:
        if args.sae_checkpoint_path and not os.path.exists(args.sae_checkpoint_path):
            parser.error(f"SAE checkpoint path specified but not found: {args.sae_checkpoint_path}")
        try:
            int(args.layer_spec)
        except ValueError:
            parser.error("If --is_sae is set, --layer_spec must be an integer layer number.")
    else: # Base model
        if args.layer_spec.isdigit():
            parser.error("If --is_sae is NOT set, --layer_spec must be a string layer name (not a number).")

    if not os.path.exists(args.worker_script):
         parser.error(f"Worker script not found: {args.worker_script}")
    if args.total_channels <= 0:
         parser.error("Total channels must be positive.")
    if args.num_processes <=0:
        parser.error("Number of processes must be positive.")

    # Create a unique subdirectory for this parallel run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_type_str = "sae" if args.is_sae else "base"
    safe_layer_spec = args.layer_spec.replace('.', '_').replace('/', '_')
    
    value_str_part = args.intervention_value.replace(',', '-').replace('.', 'p')

    dir_name_parts = [
        f"run_{run_type_str}_{safe_layer_spec}",
        f"val_{value_str_part}",
        f"ents_{args.target_entities.replace(',', '_')}",
        timestamp
    ]
    if args.is_sae and args.sae_checkpoint_path:
        safe_sae_name = os.path.basename(args.sae_checkpoint_path).replace('.pt', '')
        dir_name_parts.insert(2, safe_sae_name)

    args.run_specific_output_dir = os.path.join(args.output_dir, "_".join(dir_name_parts))
    os.makedirs(args.run_specific_output_dir, exist_ok=True)
    print(f"Output for this run will be saved in: {args.run_specific_output_dir}")

    return args

def save_run_arguments(args_obj, output_dir):
    """Saves the run arguments to a text file in the output directory."""
    args_file_path = os.path.join(output_dir, "parallel_run_config.txt")
    try:
        with open(args_file_path, 'w') as f:
            f.write(f"Parallel Run Timestamp: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            f.write("Command Line Arguments Used for Parallel Orchestrator:")
            for arg, value in sorted(vars(args_obj).items()):
                f.write(f"  --{arg}: {value}")
        print(f"Saved parallel run arguments to: {args_file_path}")
    except Exception as e:
        print(f"Error saving run arguments: {e}")


def run_worker(worker_args_tuple):
    """Function to be executed by each process."""
    process_id, start_channel, end_channel, common_args = worker_args_tuple

    # Each worker will output to a subdirectory within the run_specific_output_dir
    # This helps keep worker-specific logs and potentially outputs separate if needed.
    # However, quantitative_intervention_experiment.py already creates structured output,
    # so we'll pass the run_specific_output_dir directly.
    worker_output_dir = common_args.run_specific_output_dir

    command = [
        "python", common_args.worker_script,
        "--layer_spec", str(common_args.layer_spec),
        "--num_trials", str(common_args.num_trials),
        "--max_steps", str(common_args.max_steps),
        "--output_dir", worker_output_dir, # Worker saves its results here
        "--target_entities", common_args.target_entities,
        "--intervention_radius", str(common_args.intervention_radius),
        "--intervention_value", common_args.intervention_value,
        # Crucially, pass the channel range for this worker
        "--start_channel", str(start_channel),
        "--end_channel", str(end_channel),
        # Pass through model path related args
        "--model_path", common_args.model_path,
        "--model_base_dir", common_args.model_base_dir,
    ]

    if common_args.is_sae:
        command.append("--is_sae")
        if common_args.sae_checkpoint_path:
            command.extend(["--sae_checkpoint_path", common_args.sae_checkpoint_path])
        # If sae_checkpoint_path is None here, the worker script's logic
        # for finding the latest checkpoint will be triggered.
    
    if common_args.intervention_position:
        command.extend(["--intervention_position", common_args.intervention_position])
    
    # quantitative_intervention_experiment.py handles its own --num_top_channels_visualize

    print(f"[Dispatcher P-{process_id}] Launching worker for channels {start_channel}-{end_channel-1}...")
    print(f"[Dispatcher P-{process_id}] CMD: {' '.join(command)}")

    log_file_path = os.path.join(worker_output_dir, f"worker_p{process_id}_channels_{start_channel}-{end_channel-1}.log")

    try:
        # Try to open the log file first
        with open(log_file_path, 'w') as log_file:
            try:
                # Run the subprocess
                result = subprocess.run(command, text=True, stdout=log_file, stderr=subprocess.STDOUT, check=False) # check=False to handle non-zero exits manually
                
                if result.returncode == 0:
                    print(f"[Dispatcher P-{process_id}] Worker for channels {start_channel}-{end_channel-1} COMPLETED. Log: {log_file_path}")
                else:
                    print(f"[Dispatcher P-{process_id}] Worker for channels {start_channel}-{end_channel-1} FAILED with exit code {result.returncode}. Log: {log_file_path}")

            except subprocess.CalledProcessError as e: # This might not be hit if check=False
                 print(f"[Dispatcher P-{process_id}] Worker for channels {start_channel}-{end_channel-1} SUBPROCESS ERROR: {e}. Log: {log_file_path}")
                 # Log more details if available
                 if e.stdout:
                     log_file.write("\n--- Subprocess Stdout: ---\n" + e.stdout)
                 if e.stderr:
                     log_file.write("\n--- Subprocess Stderr: ---\n" + e.stderr)
            except FileNotFoundError:
                 print(f"[Dispatcher P-{process_id}] Worker for channels {start_channel}-{end_channel-1} FAILED: Worker script '{common_args.worker_script}' not found. Log: {log_file_path}")
                 log_file.write(f"Error: Worker script '{common_args.worker_script}' not found.\n")
            except Exception as e_subproc: # Catch other errors during subprocess execution
                 print(f"[Dispatcher P-{process_id}] Worker for channels {start_channel}-{end_channel-1} THREW UNEXPECTED SUBPROCESS EXCEPTION: {e_subproc}. Log: {log_file_path}")
                 log_file.write(f"Unexpected subprocess error: {e_subproc}\n")

    except IOError as e_io:
        # This catches errors related to opening the log_file_path itself
        print(f"[Dispatcher P-{process_id}] Worker for channels {start_channel}-{end_channel-1} FAILED: Could not open log file {log_file_path}. Error: {e_io}")
    except Exception as e_outer:
         # Catch any other unexpected errors in the worker setup
         print(f"[Dispatcher P-{process_id}] Worker for channels {start_channel}-{end_channel-1} THREW EXCEPTION BEFORE SUBPROCESS: {e_outer}. Log: {log_file_path} (if accessible).")


def main():
    args = parse_args()
    save_run_arguments(args, args.run_specific_output_dir)

    num_processes = args.num_processes
    total_channels = args.total_channels

    # Calculate channel chunks for each process
    if total_channels < num_processes :
        num_processes = total_channels # Don't use more processes than channels
        print(f"Warning: total_channels ({total_channels}) < num_processes ({args.num_processes}). Adjusting to use {num_processes} processes.")
    
    chunk_size = total_channels // num_processes
    remainder = total_channels % num_processes

    tasks = []
    current_channel = 0
    for i in range(num_processes):
        start = current_channel
        size = chunk_size + (1 if i < remainder else 0)
        if size == 0 and total_channels > 0 : # Ensure at least one channel if there are some left
             if i < remainder : # Should not happen if chunk_size >=1 or remainder covers it
                  size = 1 # Ensure progress
        
        end = start + size
        if start >= total_channels: # No more channels to assign
            break 
        end = min(end, total_channels) # Ensure 'end' does not exceed total_channels
        
        if start < end: # Only add task if there's a valid range
            tasks.append((i, start, end, args))
        current_channel = end
    
    if not tasks:
        print("No tasks generated. This might happen if total_channels is 0.")
        return

    print(f"Dispatcher: Launching {len(tasks)} worker processes to cover {total_channels} channels (from 0 to {total_channels-1}).")
    print(f"Each worker will run quantitative_intervention_experiment.py for a sub-range of channels.")
    print(f"Output directory for this parallel run: {args.run_specific_output_dir}")

    # Use multiprocessing Pool to run tasks
    with Pool(processes=len(tasks)) as pool:
        pool.map(run_worker, tasks)

    print("Parallel execution of quantitative interventions finished.")
    print(f"All worker outputs (CSVs, logs, plots) should be in: {args.run_specific_output_dir}")
    print("You can analyze the individual CSV files or combine them for a full overview.")
    print("Each worker would have generated its own plots based on its channel subset.")

if __name__ == "__main__":
    main() 