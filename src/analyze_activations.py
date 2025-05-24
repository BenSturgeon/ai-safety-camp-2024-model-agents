import torch
import numpy as np
import os
from tqdm import tqdm
import argparse
import re

# Procgen environment and project utilities
from utils import heist as heist_utils
from utils import helpers

# SAE loading
from sae_cnn import load_sae_from_checkpoint

# --- Configuration ---
NUM_RUNS_DEFAULT = 100
NUM_STEPS_PER_RUN_DEFAULT = 200
BASE_MODEL_PATH_DEFAULT = "../model_interpretable.pt"

SAE_INFO_DEFAULT = {
    "sae_conv4a_l8": {
        "path": "checkpoints_of_interest/layer_8_conv4a/sae_checkpoint_step_20000000_alpha_l8.pt",
        "base_layer_name": "conv4a"
    },
    "sae_conv3a_l6": {
        "path": "checkpoints_of_interest/layer_6_conv3a/sae_checkpoint_step_20000000_alpha_l6.pt",
        "base_layer_name": "conv3a"
    },
}
BASE_LAYERS_TO_ANALYZE_DEFAULT = ["conv4a", "conv3a"]

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
    parser = argparse.ArgumentParser(description="Analyze activation ranges in a base model and SAEs.")
    parser.add_argument("--num_runs", type=int, default=NUM_RUNS_DEFAULT, help="Number of simulation runs.")
    parser.add_argument("--num_steps_per_run", type=int, default=NUM_STEPS_PER_RUN_DEFAULT, help="Number of steps per simulation run.")
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH_DEFAULT, help="Path to the base model checkpoint (relative to src/).")
    parser.add_argument("--output_report_file", type=str, default="activation_report.txt", help="File to save the activation report (will be saved in src/).")
    return parser.parse_args()

def main():
    args = parse_args()

    global collected_base_activations, collected_sae_features
    collected_base_activations = {layer: [] for layer in BASE_LAYERS_TO_ANALYZE_DEFAULT}
    collected_sae_features = {sae_id: [] for sae_id in SAE_INFO_DEFAULT}

    print(f"Using device: {DEVICE}")
    print(f"Number of runs: {args.num_runs}, Steps per run: {args.num_steps_per_run}")
    print(f"Base model path: {args.base_model_path}")

    # 1. Load the main model using helpers
    try:
        model = helpers.load_interpretable_model(model_path=args.base_model_path)
        model.to(DEVICE)
        model.eval()
        print("Base model loaded successfully using helpers.load_interpretable_model.")
    except Exception as e:
        print(f"Error loading base model from {args.base_model_path} using helpers: {e}")
        return

    # 2. Register hooks for specified base layers
    base_handles = {}
    for layer_name_key in BASE_LAYERS_TO_ANALYZE_DEFAULT:
        if layer_name_key not in BASE_LAYER_MODULE_PATHS:
            print(f"Error: Module path for layer '{layer_name_key}' not defined in BASE_LAYER_MODULE_PATHS. Skipping hook.")
            continue
        module_path = BASE_LAYER_MODULE_PATHS[layer_name_key]
        try:
            module_to_hook = get_module_by_path(model, module_path)
            if module_to_hook:
                handle = module_to_hook.register_forward_hook(get_base_hook(layer_name_key))
                base_handles[layer_name_key] = handle
                print(f"Hooked base layer: {layer_name_key} (module: {module_path})")
            else:
                print(f"Error: Could not find module at path '{module_path}' for layer {layer_name_key}.")
        except Exception as e:
            print(f"Error setting up hook for base layer {layer_name_key} (path {module_path}): {e}")

    if not base_handles:
        print("Error: No base layer hooks were successfully attached. Aborting.")
        return
            
    # 3. Load SAEs using sae_cnn.load_sae_from_checkpoint
    loaded_saes = {}
    sae_input_layer_map = {} 
    
    for sae_id, info in SAE_INFO_DEFAULT.items():
        try:
            sae_module = load_sae_from_checkpoint(info["path"], device=DEVICE)
            loaded_saes[sae_id] = sae_module 
            sae_input_layer_map[sae_id] = info["base_layer_name"]
            print(f"Loaded SAE module: {sae_id} (from {info['path']}), takes input from base layer '{info['base_layer_name']}'.")
        except Exception as e:
            print(f"Error loading SAE {sae_id} from {info['path']} using load_sae_from_checkpoint: {e}")
            for h in base_handles.values(): h.remove()
            return
    
    if not loaded_saes:
        print("Warning: No SAEs were loaded. Proceeding with base model analysis only.")

    # 4. Simulation Loop
    venv = heist_utils.create_venv(num=1, start_level=0, num_levels=args.num_runs)

    for run_idx in tqdm(range(args.num_runs), desc="Simulation Runs"):
        obs = venv.reset()
        for step_idx in range(args.num_steps_per_run):
            if isinstance(obs, tuple) and len(obs) > 0 and isinstance(obs[0], dict) and 'rgb' in obs[0]:
                 obs_np = obs[0]['rgb'] 
            elif isinstance(obs, np.ndarray) and obs.shape == (1, 64, 64, 3):
                 obs_np = obs[0]
            elif isinstance(obs, np.ndarray) and obs.shape == (64, 64, 3):
                 obs_np = obs
            else: 
                 obs_np = obs 
            
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
            if obs_tensor.ndim == 3: 
                obs_tensor = obs_tensor.unsqueeze(0)
            elif obs_tensor.ndim == 4 and obs_tensor.shape[0] != 1:
                obs_tensor = obs_tensor[0].unsqueeze(0)
            
            if obs_tensor.shape[1:] != (3, 64, 64) and obs_tensor.shape[1:] != (64, 64, 3):
                 if obs_tensor.shape == (1, 64, 64, 3) :
                     obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                 else:
                    tqdm.write(f"Warning: Unexpected observation tensor shape: {obs_tensor.shape} in run {run_idx}, step {step_idx}. Skipping step.")
                    continue

            with torch.no_grad():
                model_output = model(obs_tensor)

            with torch.no_grad():
                for sae_id, sae_module in loaded_saes.items():
                    input_base_layer_name = sae_input_layer_map[sae_id]
                    if collected_base_activations.get(input_base_layer_name) and collected_base_activations[input_base_layer_name]:
                        base_act_for_sae = collected_base_activations[input_base_layer_name][-1].to(DEVICE)
                        
                        sae_outputs_tuple = sae_module(base_act_for_sae) 
                        
                        if isinstance(sae_outputs_tuple, tuple) and len(sae_outputs_tuple) > 2:
                             sae_feature_act = sae_outputs_tuple[2].detach().cpu()
                             collected_sae_features[sae_id].append(sae_feature_act)
                        else:
                            tqdm.write(f"Warning: Unexpected output structure from SAE {sae_id}. Type: {type(sae_outputs_tuple)}, Len: {len(sae_outputs_tuple) if isinstance(sae_outputs_tuple, tuple) else 'N/A'}. Cannot extract feature activations.")

            action_logits = None
            if hasattr(model_output, 'logits'):
                action_logits = model_output.logits
            elif isinstance(model_output, tuple) and hasattr(model_output[0], 'logits'):
                 action_logits = model_output[0].logits
            elif isinstance(model_output, torch.Tensor) and model_output.ndim == 2 :
                 action_logits = model_output
            else: 
                try:
                    action_logits = getattr(model_output, 'pi_logits', None) 
                    if action_logits is None: 
                        ac_output = getattr(model_output, 'ac_output', None)
                        if ac_output is not None: action_logits = getattr(ac_output, 'pi_logits', None)
                    if action_logits is None: raise ValueError("Could not find 'pi_logits'.")
                except Exception:
                    tqdm.write(f"Warning: Could not extract logits for action selection. Type: {type(model_output)}. Sampling random action.")
                    action_np = np.array([venv.action_space.sample()])
            
            if action_logits is not None:
                probabilities = torch.nn.functional.softmax(action_logits, dim=-1)
                action = torch.multinomial(probabilities, num_samples=1)
                action_np = action.squeeze(-1).cpu().numpy()
            
            obs, reward, done, info = venv.step(action_np)
            
            if isinstance(done, (bool, np.bool_)): is_done = done
            elif isinstance(done, (list, np.ndarray)): is_done = done[0]
            else: is_done = False 

            if is_done:
                break 

    venv.close()

    for handle in base_handles.values():
        handle.remove()
    print("\nRemoved base layer hooks.")

    # 5. Process Activations and Report
    report_lines = ["Activation Value Report"]
    report_lines.append(f"Number of Runs: {args.num_runs}, Steps per Run: {args.num_steps_per_run} (or until episode done)")
    report_lines.append(f"Base Model: {args.base_model_path}")
    report_lines.append("="*40)

    for layer_name, activations_list in collected_base_activations.items():
        if not activations_list:
            report_lines.append(f"\n--- Base Layer: {layer_name} ---")
            report_lines.append("  No activations collected.")
            continue
        
        all_acts_tensor = torch.cat(activations_list, dim=0) 
        num_channels = all_acts_tensor.shape[1]
        
        report_lines.append(f"\n--- Base Layer: {layer_name} ({num_channels} channels) ---")
        report_lines.append(f"  Shape of all collected activations: {all_acts_tensor.shape}")
        
        overall_min = all_acts_tensor.min().item()
        overall_max = all_acts_tensor.max().item()
        report_lines.append(f"  Overall Min Activation: {overall_min:.6f}")
        report_lines.append(f"  Overall Max Activation: {overall_max:.6f}")
        
        report_lines.append("  Min/Max per channel:")
        for ch_idx in range(num_channels):
            channel_data = all_acts_tensor[:, ch_idx, :, :]
            report_lines.append(f"    Channel {ch_idx:03d}: Min={channel_data.min().item():.6f}, Max={channel_data.max().item():.6f}")

    for sae_id, features_list in collected_sae_features.items():
        sae_checkpoint_info = SAE_INFO_DEFAULT[sae_id]
        if not features_list:
            report_lines.append(f"\n--- SAE: {sae_id} (Input: {sae_checkpoint_info['base_layer_name']}) ---")
            report_lines.append("  No SAE feature activations collected.")
            continue

        all_features_tensor = torch.cat(features_list, dim=0)
        num_sae_features = all_features_tensor.shape[1] 

        report_lines.append(f"\n--- SAE: {sae_id} (Input: {sae_checkpoint_info['base_layer_name']}, {num_sae_features} features) ---")
        report_lines.append(f"  SAE Path: {sae_checkpoint_info['path']}")
        report_lines.append(f"  Shape of all collected SAE features: {all_features_tensor.shape}")

        overall_min = all_features_tensor.min().item()
        overall_max = all_features_tensor.max().item()
        report_lines.append(f"  Overall Min Feature Activation: {overall_min:.6f}")
        report_lines.append(f"  Overall Max Feature Activation: {overall_max:.6f}")

        report_lines.append("  Min/Max per feature:")
        for feat_idx in range(num_sae_features):
            feature_data = all_features_tensor[:, feat_idx, :, :] 
            report_lines.append(f"    Feature {feat_idx:04d}: Min={feature_data.min().item():.6f}, Max={feature_data.max().item():.6f}")
            
    print("\n" + "\n".join(report_lines))

    try:
        with open(args.output_report_file, "w") as f:
            f.write("\n".join(report_lines))
        print(f"\nReport saved to {args.output_report_file}")
    except Exception as e:
        print(f"\nError saving report to {args.output_report_file}: {e}")

if __name__ == "__main__":
    main() 