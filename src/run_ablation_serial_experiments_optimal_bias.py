#!/usr/bin/env python3
import argparse
import subprocess
import os
import re
import datetime
import pandas as pd
import json
import sys

"""
This script automates a series of ablation sweep experiments **for BASE models** (i.e. *not* SAEs).

For each checkpoint in the supplied directory that matches the pattern
    model_<step>.0.pt
where <step> increases in 1000-step intervals (e.g. 1001, 2001, … 60001), we:
 1. Test all 4 bias directions to find the optimal one (lowest collection rate)
 2. Launch `channel_ablation_sweep_bias_corrected.py` **sequentially** once per model
    using the optimal bias direction.

This follows the same checkpoint discovery pattern as run_base_model_serial_experiments.py
but runs ablation sweeps with optimal bias correction.

Example usage (from the src/ directory):
    python run_ablation_serial_experiments_optimal_bias.py \
        --models_dir ../base_models/full_run

The output of each parallel run is placed in a timestamped sub-directory inside
`--output_root_dir` (default: ablation_sweep_runs/ablation_sweep_base_optimal_bias).
"""

# --------------------------------------------------------------------------------------
# Default configuration – tweak via command-line flags if needed
# --------------------------------------------------------------------------------------
DEFAULT_MODELS_DIR = ("../base_models/full_run")

# Only every 1000-step checkpoint (i.e. model_1001.0.pt, model_2001.0.pt, …)
CHECKPOINT_INTERVAL = 1000  # training steps
MAX_STEP_DEFAULT = 60000  # inclusive upper bound on <step> to consider

# ablation sweep parameters ---------------------------------------------------
NUM_TRIALS = 250                # per channel
MAX_STEPS_PER_TRIAL = 100       # default in parallel_ablation_sweep script
LAYER_SPEC = "conv4a"           # layer to ablate
TOTAL_CHANNELS_CONV4A = 32      # conv4a has 32 output channels in the base CNN
MAZE_TYPE = "fork"              # maze type for experiments
BIAS_TEST_TRIALS = 30           # trials for bias direction optimization

# Output -------------------------------------------------------------------------------
OUTPUT_ROOT_DIR = "ablation_sweep_runs/ablation_sweep_base_optimal_bias"  # created inside src/

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = "python"  # adjust if your system needs "python3"

# --------------------------------------------------------------------------------------

# Accept an explicit `interval` parameter (defaults to CHECKPOINT_INTERVAL)
def discover_checkpoints(models_dir: str, max_step: int, interval: int = CHECKPOINT_INTERVAL):
    """Return a list of (step, path) tuples sorted ascending for 1000-interval ckpts."""
    # Match both variants:
    #   1) model_<step>.pt          (e.g. model_60000.pt)
    #   2) model_<step>.0.pt        (e.g. model_60001.0.pt)
    # The training script we used originally saved checkpoints with the off-by-one
    # "+1" and a trailing ".0".  Some later runs dropped the ".0" suffix and
    # use the exact training step instead.  We support both here.
    pattern = re.compile(r"model_(\d+)(?:\.0)?\.pt$")
    candidates = []
    for fname in os.listdir(models_dir):
        m = pattern.match(fname)
        if not m:
            continue
        # When the filename ends with ".0.pt" we continue to assume the historical
        # off-by-one convention (e.g. model_1001.0.pt == training step 1000).
        # For the newer "model_<step>.pt" files we take the number verbatim.
        num = int(m.group(1))
        if fname.endswith(".0.pt"):
            step = num - 1
        else:
            step = num
        # Keep only multiples of the desired interval and within range
        if step % interval == 0 and 0 < step <= max_step:
            candidates.append((step, os.path.join(models_dir, fname)))
    candidates.sort(key=lambda t: t[0])
    return candidates

def test_bias_direction_optimization(model_path: str, output_dir: str, num_trials: int = 30):
    """Test all 4 bias directions to find the one with lowest collection rate."""
    
    directions = ["up", "down", "left", "right"]
    results = []
    
    print(f"[Bias Test] Testing optimal bias direction for {model_path}")
    
    for direction in directions:
        print(f"[Bias Test] Testing {direction} direction...")
        
        try:
            # Run the base collection rate test
            cmd = [
                PYTHON, "test_base_collection_rates.py",
                "--single_model", model_path,
                "--bias_direction", direction,
                "--num_trials", str(num_trials)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse results from output
            gem_rate = 0.0
            timeout_rate = 1.0
            avg_reward = 0.0
            
            lines = result.stdout.split('\\n')
            for line in lines:
                if "Gem collection rate:" in line:
                    gem_rate = float(line.split(":")[1].strip().replace("%", "")) / 100
                elif "Timeout rate:" in line:
                    timeout_rate = float(line.split(":")[1].strip().replace("%", "")) / 100
                elif "Average reward:" in line:
                    avg_reward = float(line.split(":")[1].strip())
            
            results.append({
                'direction': direction,
                'gem_rate': gem_rate,
                'timeout_rate': timeout_rate,
                'avg_reward': avg_reward
            })
            
            print(f"[Bias Test]   {direction}: {gem_rate:.1%} gem rate, {avg_reward:.2f} avg reward")
            
        except Exception as e:
            print(f"[Bias Test]   Error testing {direction}: {e}")
            results.append({
                'direction': direction,
                'gem_rate': 1.0,  # High rate = bad for optimization
                'timeout_rate': 0.0,
                'avg_reward': 0.0,
                'error': str(e)
            })
    
    # Save bias testing results
    bias_df = pd.DataFrame(results)
    bias_file = os.path.join(output_dir, "bias_direction_optimization.csv")
    bias_df.to_csv(bias_file, index=False)
    
    # Find optimal direction (lowest gem collection rate)
    optimal = min(results, key=lambda x: x['gem_rate'])
    print(f"[Bias Test] Optimal bias direction: {optimal['direction']} ({optimal['gem_rate']:.1%} gem rate)")
    
    return optimal['direction'], results

def run_ablation_sweep(model_path: str, output_root: str, num_trials: int, bias_direction: str):
    """Launch channel_ablation_sweep_bias_corrected.py for the given model with optimal bias direction."""
    
    cmd = [
        PYTHON, "channel_ablation_sweep_bias_corrected.py",
        "--model_path", model_path,
        "--layer_spec", LAYER_SPEC,
        "--num_trials", str(num_trials),
        "--max_steps", str(MAX_STEPS_PER_TRIAL),
        "--output_dir", output_root,
        "--total_channels_for_base_layer", str(TOTAL_CHANNELS_CONV4A),
        "--start_channel", "0",
        "--end_channel", str(TOTAL_CHANNELS_CONV4A),
        "--bias_direction", bias_direction,
        "--maze_type", MAZE_TYPE,
        "--no_save_gifs"
    ]
    
    bias_info = f" (bias-corrected for {bias_direction})"
    print(f"[Ablation] Running{bias_info}: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[Ablation] ✓ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Ablation] ✗ Failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Sequential ablation sweep runs over base-model checkpoints with optimal bias correction.")
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR,
                        help="Directory containing model_*.pt checkpoints.")
    parser.add_argument("--max_step", type=int, default=MAX_STEP_DEFAULT,
                        help="Highest training-step (inclusive) to consider.")
    parser.add_argument("--output_root_dir", type=str, default=OUTPUT_ROOT_DIR,
                        help="Base directory for results (created inside src/).")
    parser.add_argument("--interval", type=int, default=CHECKPOINT_INTERVAL,
                        help="Select checkpoints spaced every N training steps (default: %(default)s).")
    parser.add_argument("--num_trials", type=int, default=NUM_TRIALS,
                        help="Number of trials per channel (default: %(default)s).")
    parser.add_argument("--layer_spec", type=str, default=LAYER_SPEC,
                        help="Layer specification (default: %(default)s).")
    parser.add_argument("--total_channels", type=int, default=TOTAL_CHANNELS_CONV4A,
                        help="Total number of channels in the layer (default: %(default)s).")
    parser.add_argument("--maze_type", type=str, default=MAZE_TYPE, choices=["fork", "corners"],
                        help="Type of maze to use (default: %(default)s).")
    parser.add_argument("--bias_test_trials", type=int, default=BIAS_TEST_TRIALS,
                        help="Number of trials for bias direction testing (default: %(default)s)")
    parser.add_argument("--skip_bias_test", action="store_true",
                        help="Skip bias testing and use known optimal directions")
    
    args = parser.parse_args()

    src_cwd = os.getcwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create serial sweep directory with configuration details
    sweep_config_str = f"{args.layer_spec}_{args.num_trials}trials_{args.interval}step"
    full_output_root = os.path.join(src_cwd, args.output_root_dir, f"serial_sweep_{sweep_config_str}_{timestamp}")
    os.makedirs(full_output_root, exist_ok=True)

    checkpoints = discover_checkpoints(args.models_dir, args.max_step, args.interval)
    if not checkpoints:
        print("No checkpoints found – check the directory or max_step value.")
        return

    print(f"Discovered {len(checkpoints)} checkpoints to process ({args.interval}-step intervals).")
    print(f"Configuration:")
    print(f"  Layer: {args.layer_spec}")
    print(f"  Channels: {args.total_channels}")
    print(f"  Trials per channel: {args.num_trials}")
    print(f"  Maze type: {args.maze_type}")
    print(f"  Bias test trials: {args.bias_test_trials}")
    
    # Save configuration file
    config_file = os.path.join(full_output_root, "serial_sweep_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"Serial Ablation Sweep Configuration (Optimal Bias)\\n")
        f.write(f"Timestamp: {timestamp}\\n")
        f.write(f"Models directory: {args.models_dir}\\n")
        f.write(f"Max step: {args.max_step}\\n")
        f.write(f"Checkpoint interval: {args.interval}\\n")
        f.write(f"Layer: {args.layer_spec}\\n")
        f.write(f"Total channels: {args.total_channels}\\n")
        f.write(f"Trials per channel: {args.num_trials}\\n")
        f.write(f"Maze type: {args.maze_type}\\n")
        f.write(f"Bias test trials: {args.bias_test_trials}\\n")
        f.write(f"Skip bias test: {args.skip_bias_test}\\n")
        f.write(f"Number of checkpoints: {len(checkpoints)}\\n\\n")
        f.write("Checkpoints:\\n")
        for step, path in checkpoints:
            f.write(f"  Step {step}: {path}\\n")

    # Known optimal directions for major checkpoints (can skip bias testing)
    known_optimal_directions = {
        30000: "down",  # Has upward bias
        40000: "down",  # Has upward bias
        50000: "down",  # Has upward bias
        60000: "down",  # Has upward bias
    }

    # Track results
    results_summary = []
    
    for idx, (step, ckpt_path) in enumerate(checkpoints, 1):
        print("\\n" + "="*80)
        print(f"Processing checkpoint {idx}/{len(checkpoints)} – step {step} – {ckpt_path}")

        # Create output directory for this checkpoint
        checkpoint_output = os.path.join(full_output_root, f"step_{step}")
        os.makedirs(checkpoint_output, exist_ok=True)

        # Test for optimal bias direction
        if args.skip_bias_test and step in known_optimal_directions:
            optimal_bias_direction = known_optimal_directions[step]
            bias_results = []
            print(f"[Bias Test] Using known optimal direction: {optimal_bias_direction}")
        else:
            optimal_bias_direction, bias_results = test_bias_direction_optimization(
                ckpt_path, checkpoint_output, args.bias_test_trials
            )

        # Run ablation sweep with optimal bias direction
        ablation_success = run_ablation_sweep(
            ckpt_path, checkpoint_output, args.num_trials, optimal_bias_direction
        )

        # Record results
        results_summary.append({
            'step': step,
            'checkpoint_path': ckpt_path,
            'optimal_bias_direction': optimal_bias_direction,
            'bias_test_skipped': args.skip_bias_test and step in known_optimal_directions,
            'ablation_success': ablation_success,
            'output_dir': checkpoint_output
        })

        print(f"Completed ablation sweep for step {step}.")

    # Save results summary
    summary_df = pd.DataFrame(results_summary)
    summary_file = os.path.join(full_output_root, "processing_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    print("\\n" + "="*80)
    print("ALL ablation sweeps finished. Results in:")
    print(f"  {full_output_root}")
    print(f"Processing summary: {summary_file}")
    
    # Print summary table
    print(f"\\nProcessing Summary:")
    print(summary_df[['step', 'optimal_bias_direction', 'ablation_success']].to_string(index=False))

if __name__ == "__main__":
    main()