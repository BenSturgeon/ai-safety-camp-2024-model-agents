import torch
import numpy as np
import os
from tqdm import tqdm
import argparse
import re
import glob

# Procgen environment and project utilities
from utils import heist as heist_utils
from utils import helpers 

# SAE loading
from sae_cnn import load_sae_from_checkpoint

# --- Configuration ---
NUM_RUNS_DEFAULT = 50
NUM_STEPS_PER_RUN_DEFAULT = 300
BATCH_SIZE_DEFAULT = 16  # Process SAEs in batches for efficiency
SAMPLE_RATIO_DEFAULT = 1.0  # Sample ratio for faster analysis (1.0 = all steps)
BASE_MODEL_PATH_DEFAULT = "../model_interpretable.pt"

# Auto-discover SAE files in checkpoints_of_interest
def discover_sae_checkpoints():
    """Automatically discover all SAE checkpoint files for conv3a and conv4a layers."""
    sae_info = {}
    
    # Define patterns for conv3a and conv4a layers
    patterns = [
        ("layer_6_conv3a", "conv3a"),
        ("layer_8_conv4a", "conv4a")
    ]
    
    for layer_dir, base_layer_name in patterns:
        checkpoint_dir = f"checkpoints_of_interest/{layer_dir}"
        if os.path.exists(checkpoint_dir):
            # Look for .pt files that are not .backup files
            pt_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
            pt_files = [f for f in pt_files if not f.endswith(".backup")]
            
            for pt_file in pt_files:
                filename = os.path.basename(pt_file)
                # Create a descriptive SAE ID
                if "alpha" in filename:
                    sae_id = f"sae_{base_layer_name}_alpha_{layer_dir.split('_')[1]}"
                elif "beta" in filename:
                    sae_id = f"sae_{base_layer_name}_beta_{layer_dir.split('_')[1]}"
                elif "gamma" in filename:
                    sae_id = f"sae_{base_layer_name}_gamma_{layer_dir.split('_')[1]}"
                elif "delta" in filename:
                    sae_id = f"sae_{base_layer_name}_delta_{layer_dir.split('_')[1]}"
                elif "epsilon" in filename:
                    sae_id = f"sae_{base_layer_name}_epsilon_{layer_dir.split('_')[1]}"
                else:
                    # For files without specific names, use a generic identifier
                    sae_id = f"sae_{base_layer_name}_generic_{layer_dir.split('_')[1]}"
                
                sae_info[sae_id] = {
                    "path": pt_file,
                    "base_layer_name": base_layer_name
                }
    
    return sae_info

# SAE_INFO_DEFAULT will be populated by discover_sae_checkpoints()
SAE_INFO_DEFAULT = discover_sae_checkpoints()

# Print discovered SAEs for user confirmation
print("Discovered SAE checkpoints:")
for sae_id, info in SAE_INFO_DEFAULT.items():
    print(f"  {sae_id}: {info['path']} -> {info['base_layer_name']}")
print()

# Define module paths for direct hooking
BASE_LAYER_MODULE_PATHS = {
    "conv1a": "conv1a",
    "conv2a": "conv2a",
    "conv2b": "conv2b",
    "conv3a": "conv3a",
    "conv4a": "conv4a",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global storage for activations ---
collected_base_activations = {}
collected_sae_features = {}

# --- Helper function to get a module by its path string ---
def get_module_by_path(model, path_str):
    module = model
    for element in path_str.split('.'):
        if '[' in element and ']' in element:
            base, idx_str = element.split('[')
            idx = int(idx_str[:-1])
            module = getattr(module, base)[idx]
        else:
            module = getattr(module, element)
    return module

# --- Hook Function for Base Model Layers ---
def get_base_hook(layer_name_key):
    def hook_fn(module, input_act, output_act):
        collected_base_activations[layer_name_key].append(output_act.detach().cpu())
    return hook_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze activation ranges in a base model and all discovered SAEs.")
    parser.add_argument("--num_runs", type=int, default=NUM_RUNS_DEFAULT, help="Number of simulation runs.")
    parser.add_argument("--num_steps_per_run", type=int, default=NUM_STEPS_PER_RUN_DEFAULT, help="Number of steps per simulation run.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size for SAE processing (higher = faster but more memory).")
    parser.add_argument("--sample_ratio", type=float, default=SAMPLE_RATIO_DEFAULT, help="Ratio of steps to sample for SAE analysis (0.1 = 10%% of steps, faster).")
    parser.add_argument("--fast", action="store_true", help="Fast mode: use fewer runs/steps for quick testing (10 runs, 50 steps each).")
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH_DEFAULT, help="Path to the base model checkpoint (relative to src/).")
    parser.add_argument("--output_report_file", type=str, default="comprehensive_sae_activation_report.txt", help="File to save the comprehensive activation report (will be saved in src/).")
    return parser.parse_args()

def main():
    args = parse_args()

    # Apply fast mode settings
    if args.fast:
        args.num_runs = 10
        args.num_steps_per_run = 50
        args.sample_ratio = 0.5
        print("Fast mode enabled: 10 runs, 50 steps each, 50% sampling")

    global collected_base_activations, collected_sae_features
    collected_base_activations = {layer: [] for layer in SAE_INFO_DEFAULT}
    collected_sae_features = {sae_id: [] for sae_id in SAE_INFO_DEFAULT}

    print(f"Using device: {DEVICE}")
    print(f"Number of runs: {args.num_runs}, Steps per run: {args.num_steps_per_run}")
    print(f"Batch size: {args.batch_size}, Sample ratio: {args.sample_ratio}")
    print(f"Base model path: {args.base_model_path}")
    print(f"Analyzing {len(SAE_INFO_DEFAULT)} SAE checkpoints")
    print()

    # Load the base model
    model = helpers.load_interpretable_model(model_path=args.base_model_path)
    model.to(DEVICE)
    model.eval()
    print("Base model loaded successfully using helpers.load_interpretable_model.")

    # Load all SAEs
    print("\nLoading SAEs...")
    loaded_saes = {}
    sae_input_layer_map = {}
    
    for sae_id, sae_info in SAE_INFO_DEFAULT.items():
        try:
            sae_path = sae_info["path"]
            print(f"Loading SAE from checkpoint: {sae_path}")
            sae_module = load_sae_from_checkpoint(sae_path, DEVICE)
            loaded_saes[sae_id] = sae_module
            sae_input_layer_map[sae_id] = sae_info["base_layer_name"]
            print(f"Loaded SAE module: {sae_id} (from {sae_path}), takes input from base layer '{sae_info['base_layer_name']}'.")
        except Exception as e:
            print(f"Failed to load SAE {sae_id} from {sae_path}: {e}")
            continue

    # Initialize collections for SAE feature activations
    collected_sae_features = {sae_id: [] for sae_id in loaded_saes.keys()}
    unique_base_layers = set(info["base_layer_name"] for info in SAE_INFO_DEFAULT.values())
    collected_base_activations = {layer_name: [] for layer_name in unique_base_layers}

    # Set up hooks for base layer activations
    base_layer_hooks = {}
    
    def create_base_hook(layer_name):
        def hook_fn(module, input, output):
            collected_base_activations[layer_name].append(output.detach())  # Keep on GPU for efficiency
        return hook_fn

    # Register hooks for each unique base layer
    for base_layer_name in unique_base_layers:
        if hasattr(model, base_layer_name):
            base_layer_module = getattr(model, base_layer_name)
            hook = base_layer_module.register_forward_hook(create_base_hook(base_layer_name))
            base_layer_hooks[base_layer_name] = hook
            print(f"Hooked base layer: {base_layer_name} (module: {base_layer_name})")
        else:
            print(f"Warning: Base layer {base_layer_name} not found in model")

    print(f"\nRunning {args.num_runs} simulation runs...")
    
    # Calculate sampling parameters
    total_steps = args.num_runs * args.num_steps_per_run
    sample_interval = max(1, int(1.0 / args.sample_ratio)) if args.sample_ratio < 1.0 else 1
    expected_samples = int(total_steps * args.sample_ratio)
    
    print(f"Collecting ~{expected_samples:,} samples (sampling every {sample_interval} steps)")
    
    # Batch collection of activations
    activation_batch = {layer_name: [] for layer_name in unique_base_layers}
    global_step = 0
    
    # Run simulations
    with torch.no_grad():
        for run_idx in tqdm(range(args.num_runs), desc="Simulation Runs"):
                # Generate a random environment and run for num_steps_per_run
                env = heist_utils.create_venv(num=1, start_level=0, num_levels=0)
                obs = env.reset()
                
                for step_idx in range(args.num_steps_per_run):
                    global_step += 1
                    
                    # Only collect activations based on sampling ratio
                    if global_step % sample_interval == 0:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                        _ = model(obs_tensor)  # Forward pass to collect base activations
                        
                        # Collect base activations (hooks fill collected_base_activations)
                        for layer_name in unique_base_layers:
                            if collected_base_activations.get(layer_name) and collected_base_activations[layer_name]:
                                # Take the latest activation and add to batch
                                base_act = collected_base_activations[layer_name][-1]
                                activation_batch[layer_name].append(base_act)
                                
                        # Process SAE batches when batch is full
                        for layer_name in activation_batch:
                            if len(activation_batch[layer_name]) >= args.batch_size:
                                batch_tensor = torch.cat(activation_batch[layer_name], dim=0)
                                
                                # Process all SAEs for this layer in the batch
                                for sae_id, sae_module in loaded_saes.items():
                                    if sae_input_layer_map[sae_id] == layer_name:
                                        try:
                                            sae_outputs_tuple = sae_module(batch_tensor)
                                            if isinstance(sae_outputs_tuple, tuple) and len(sae_outputs_tuple) > 2:
                                                sae_feature_acts = sae_outputs_tuple[2].detach().cpu()
                                                collected_sae_features[sae_id].extend([act for act in sae_feature_acts])
                                        except Exception as e:
                                            tqdm.write(f"Error processing SAE {sae_id}: {e}")
                                
                                # Clear the batch
                                activation_batch[layer_name] = []

                    # Step in environment
                    action = np.random.randint(0, env.action_space.n)
                    obs, _, done, _ = env.step(np.array([action]))
                    if done[0]:
                        obs = env.reset()
                
                # Close environment after each run
                env.close()

            # Process any remaining activations in batches
                print("\nProcessing remaining activation batches...")
        for layer_name in activation_batch:
            if activation_batch[layer_name]:
                batch_tensor = torch.cat(activation_batch[layer_name], dim=0)
                
                for sae_id, sae_module in loaded_saes.items():
                    if sae_input_layer_map[sae_id] == layer_name:
                        try:
                            sae_outputs_tuple = sae_module(batch_tensor)
                            if isinstance(sae_outputs_tuple, tuple) and len(sae_outputs_tuple) > 2:
                                sae_feature_acts = sae_outputs_tuple[2].detach().cpu()
                                collected_sae_features[sae_id].extend([act for act in sae_feature_acts])
                        except Exception as e:
                            print(f"Error processing remaining SAE {sae_id}: {e}")

    print(f"\nCompleted {args.num_runs} runs with {global_step:,} total steps.")

    # Remove hooks
    for hook in base_layer_hooks.values():
        hook.remove()
    print("Removed base layer hooks.")

    # Collect all base activations for analysis
    print("Collecting base activations for final analysis...")
    for base_layer_name in unique_base_layers:
        if collected_base_activations.get(base_layer_name):
            # Keep a small sample of base activations for the report
            sample_size = min(1000, len(collected_base_activations[base_layer_name]))
            if sample_size > 0:
                sampled_acts = collected_base_activations[base_layer_name][:sample_size]
                collected_base_activations[base_layer_name] = [act.cpu() for act in sampled_acts]

    # Generate comprehensive report
    print("\nGenerating comprehensive activation report...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE SAE ACTIVATION ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Number of simulation runs: {args.num_runs}")
    report_lines.append(f"Steps per run: {args.num_steps_per_run}")
    report_lines.append(f"Total SAEs analyzed: {len(loaded_saes)}")
    report_lines.append("")

    # Base layer activations summary
    report_lines.append("BASE LAYER ACTIVATIONS SUMMARY:")
    report_lines.append("-" * 40)
    for base_layer_name in unique_base_layers:
        if collected_base_activations.get(base_layer_name):
            all_base_acts = torch.cat(collected_base_activations[base_layer_name], dim=0)
            base_min = all_base_acts.min().item()
            base_max = all_base_acts.max().item()
            base_mean = all_base_acts.mean().item()
            base_std = all_base_acts.std().item()
            
            report_lines.append(f"Layer {base_layer_name}:")
            report_lines.append(f"  Min: {base_min:.6f}")
            report_lines.append(f"  Max: {base_max:.6f}")
            report_lines.append(f"  Mean: {base_mean:.6f}")
            report_lines.append(f"  Std: {base_std:.6f}")
            report_lines.append(f"  Total samples: {all_base_acts.shape[0]}")
            report_lines.append("")

    # SAE feature activations summary
    report_lines.append("SAE FEATURE ACTIVATIONS SUMMARY:")
    report_lines.append("-" * 40)
    
    # Group SAEs by layer for better organization
    sae_by_layer = {}
    for sae_id, sae_info in SAE_INFO_DEFAULT.items():
        if sae_id in loaded_saes:  # Only include successfully loaded SAEs
            base_layer = sae_info["base_layer_name"]
            if base_layer not in sae_by_layer:
                sae_by_layer[base_layer] = []
            sae_by_layer[base_layer].append(sae_id)
    
    for base_layer_name, sae_ids in sae_by_layer.items():
        report_lines.append(f"\n{base_layer_name.upper()} LAYER SAEs:")
        report_lines.append("=" * 50)
        
        for sae_id in sae_ids:
            if collected_sae_features.get(sae_id):
                all_sae_features = torch.cat(collected_sae_features[sae_id], dim=0)
                sae_min = all_sae_features.min().item()
                sae_max = all_sae_features.max().item()
                sae_mean = all_sae_features.mean().item()
                sae_std = all_sae_features.std().item()
                
                # Count non-zero activations for sparsity analysis
                non_zero_count = (all_sae_features != 0).sum().item()
                total_count = all_sae_features.numel()
                sparsity_ratio = non_zero_count / total_count
                
                report_lines.append(f"\n{sae_id}:")
                report_lines.append(f"  Min: {sae_min:.6f}")
                report_lines.append(f"  Max: {sae_max:.6f}")
                report_lines.append(f"  Mean: {sae_mean:.6f}")
                report_lines.append(f"  Std: {sae_std:.6f}")
                report_lines.append(f"  Non-zero ratio: {sparsity_ratio:.4f} ({non_zero_count}/{total_count})")
                report_lines.append(f"  Shape: {all_sae_features.shape}")
            else:
                report_lines.append(f"\n{sae_id}: No data collected")

    # Summary table
    report_lines.append("\n" + "="*80)
    report_lines.append("QUICK REFERENCE TABLE - MIN/MAX VALUES:")
    report_lines.append("="*80)
    report_lines.append(f"{'SAE ID':<35} {'Layer':<10} {'Min':<12} {'Max':<12} {'Non-zero %':<12}")
    report_lines.append("-" * 80)
    
    for base_layer_name, sae_ids in sae_by_layer.items():
        for sae_id in sae_ids:
            if collected_sae_features.get(sae_id):
                all_sae_features = torch.cat(collected_sae_features[sae_id], dim=0)
                sae_min = all_sae_features.min().item()
                sae_max = all_sae_features.max().item()
                non_zero_count = (all_sae_features != 0).sum().item()
                total_count = all_sae_features.numel()
                sparsity_ratio = non_zero_count / total_count * 100
                
                report_lines.append(f"{sae_id:<35} {base_layer_name:<10} {sae_min:<12.6f} {sae_max:<12.6f} {sparsity_ratio:<12.2f}")

    report_lines.append("="*80)

    # Write report to file
    report_content = "\n".join(report_lines)
    with open(args.output_report_file, 'w') as f:
        f.write(report_content)
    
    print(f"Comprehensive activation report saved to: {args.output_report_file}")
    print("\nREPORT PREVIEW:")
    print("-" * 50)
    # Print the summary table for quick viewing
    summary_start = None
    summary_end = None
    for i, line in enumerate(report_lines):
        if "QUICK REFERENCE TABLE" in line:
            summary_start = i
        elif summary_start and line.startswith("="*80) and i > summary_start + 5:
            summary_end = i + 1
            break
    
    if summary_start and summary_end:
        for line in report_lines[summary_start:summary_end]:
            print(line)

if __name__ == "__main__":
    main() 