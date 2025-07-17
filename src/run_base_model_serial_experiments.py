import argparse
import subprocess
import os
import re
import datetime

"""
This script automates a series of quantitative-intervention experiments **for BASE models** (i.e. *not* SAEs).

For each checkpoint in the supplied directory that matches the pattern
    model_<step>.0.pt
where <step> increases in 1000-step intervals (e.g. 1001, 2001, … 36001), we:
 1. Invoke `analyze_activations.py` in *fast* + *no_sae* mode to obtain the
    observed max activation of the conv4a layer for that specific model.
 2. Set the intervention range to 0->1.25×(max activation) and pass this as
    `--intervention_value 0,<upper>` to `parallel_quantitative_intervention.py`.
 3. Launch `parallel_quantitative_intervention.py` **sequentially** once per
    model.

Example usage (from the src/ directory):
    python run_base_model_serial_experiments.py \
        --models_dir ../train-procgen-pfrl/log/heist/nlev_0_easy/vanilla/envpool_64x256_bs8_ep11

The output of each parallel run is placed in a timestamped sub-directory inside
`--output_root_dir` (default: quantitative_intervention_runs_base).
"""

# --------------------------------------------------------------------------------------
# Default configuration – tweak via command-line flags if needed
# --------------------------------------------------------------------------------------
DEFAULT_MODELS_DIR = ("../base_models/full_run")

# Only every 1000-step checkpoint (i.e. model_1001.0.pt, model_2001.0.pt, …)
CHECKPOINT_INTERVAL = 1000  # training steps
MAX_STEP_DEFAULT = 60000  # inclusive upper bound on <step> to consider

# quantitative-intervention parameters ---------------------------------------------------
NUM_TRIALS = 500                # per channel, matches default in parallel script
MAX_STEPS_PER_TRIAL = 20        # default in parallel script
INTERVENTION_RADIUS = 1
TARGET_ENTITIES = "gem,blue_key,green_key,red_key,blue_lock,green_lock,red_lock"
TOTAL_CHANNELS_CONV4A = 32      # conv4a has 32 output channels in the base CNN
NUM_PARALLEL_PROCESSES = 8      # workers launched *inside* parallel script

# Output -------------------------------------------------------------------------------
OUTPUT_ROOT_DIR = "quantitative_intervention_runs/quantitative_interventions_base"  # created inside src/

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = "python"  # adjust if your system needs "python3"
ANALYZE_SCRIPT = os.path.join(SCRIPTS_DIR, "analyze_activations.py")
PARALLEL_SCRIPT = os.path.join(SCRIPTS_DIR, "parallel_quantitative_intervention.py")

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


def get_conv4a_max_activation(model_path: str, tmp_report: str):
    """Run analyze_activations.py (fast / no_sae) and parse conv4a max value."""
    cmd = [
        PYTHON, ANALYZE_SCRIPT,
        "--no_sae",
        "--fast",
        "--output_report_file", tmp_report,
        "--base_model_path", model_path,
    ]
    print(f"[Activation] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[Activation] FAILED for model {model_path}.\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")
        raise

    # Parse the report
    max_val = None
    with open(tmp_report, "r") as f:
        for line in f:
            if line.strip().startswith("Layer conv4a:"):
                # The next line should be "  Min: ...", then "  Max: ..."
                _min_line = next(f, None)
                max_line = next(f, None)
                if max_line is None:
                    break
                m = re.search(r"Max:\s+([0-9.]+)", max_line)
                if m:
                    max_val = float(m.group(1))
                break
    if max_val is None:
        raise RuntimeError(f"Could not parse conv4a max activation from {tmp_report}")
    return max_val


def run_parallel_intervention(model_path: str, intervention_upper: float, output_root: str):
    """Launch parallel_quantitative_intervention.py for the given model."""
    intervention_value = f"0,{intervention_upper:.6f}"

    cmd = [
        PYTHON, PARALLEL_SCRIPT,
        "--model_path", model_path,
        "--layer_spec", "conv4a",
        "--target_entities", TARGET_ENTITIES,
        "--num_trials", str(NUM_TRIALS),
        "--max_steps", str(MAX_STEPS_PER_TRIAL),
        "--intervention_radius", str(INTERVENTION_RADIUS),
        "--intervention_value", intervention_value,
        "--output_dir", output_root,
        "--total_channels", str(TOTAL_CHANNELS_CONV4A),
        "--num_processes", str(NUM_PARALLEL_PROCESSES),
        "--worker_script", os.path.join(SCRIPTS_DIR, "quantitative_intervention_experiment.py")
    ]

    print(f"[Parallel] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Sequential quantitative-intervention runs over base-model checkpoints.")
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR,
                        help="Directory containing model_*.pt checkpoints.")
    parser.add_argument("--max_step", type=int, default=MAX_STEP_DEFAULT,
                        help="Highest training-step (inclusive) to consider.")
    parser.add_argument("--output_root_dir", type=str, default=OUTPUT_ROOT_DIR,
                        help="Base directory for results (created inside src/).")
    parser.add_argument("--interval", type=int, default=CHECKPOINT_INTERVAL,
                        help="Select checkpoints spaced every N training steps (default: %(default)s).")
    args = parser.parse_args()

    src_cwd = os.getcwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_root = os.path.join(src_cwd, f"{args.output_root_dir}_{timestamp}")
    os.makedirs(full_output_root, exist_ok=True)

    checkpoints = discover_checkpoints(args.models_dir, args.max_step, args.interval)
    if not checkpoints:
        print("No checkpoints found – check the directory or max_step value.")
        return

    print(f"Discovered {len(checkpoints)} checkpoints to process ({args.interval}-step intervals).")

    for idx, (step, ckpt_path) in enumerate(checkpoints, 1):
        print("\n" + "="*80)
        print(f"Processing checkpoint {idx}/{len(checkpoints)} – step {step} – {ckpt_path}")

        tmp_report = os.path.join(full_output_root, f"activation_report_step_{step}.txt")
        max_act = get_conv4a_max_activation(ckpt_path, tmp_report)
        upper = max_act * 1.25
        print(f"[Activation] conv4a max = {max_act:.6f} → using upper intervention bound {upper:.6f}")

        # Each parallel run places *its own* timestamped subdir inside full_output_root
        run_parallel_intervention(ckpt_path, upper, full_output_root)
        print(f"Completed experiment for step {step}.")

    print("\n" + "="*80)
    print("ALL experiments finished. Results in:")
    print(f"  {full_output_root}")


if __name__ == "__main__":
    main() 