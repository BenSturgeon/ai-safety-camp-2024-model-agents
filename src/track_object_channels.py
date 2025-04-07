#!/usr/bin/env python
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import argparse
from PIL import Image
import imageio
from tqdm import tqdm

# Import local modules
from utils.environment_modification_experiments import create_example_maze_sequence, ENTITY_TYPES, ENTITY_COLORS
from utils import helpers
from sae_cnn import load_sae_from_checkpoint, ordered_layer_names

class EntityTrackingExperiment:
    def __init__(self, model_path, device=None):
        """
        Initialize the experiment to track SAE channels that correlate with entity positions.
        
        Args:
            model_path (str): Path to the model checkpoint
            device (torch.device): Device to run on (defaults to CUDA if available, else CPU)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading model from {model_path}")
        self.model = helpers.load_interpretable_model(model_path=model_path).to(self.device)
        
        self.target_layers = {
            1: "conv1a",
            3: "conv2a",
            4: "conv2b",
            6: "conv3a",
            8: "conv4a",
        }
        
        self.saes = {}
        self.handles = []
        self.all_activations = {}
        
    def load_sae(self, layer_number, sae_checkpoint_path=None, sae_step=15000000):
        """
        Load an SAE for a specific layer. Prioritizes sae_checkpoint_path if provided.

        Args:
            layer_number (int): The layer number to load the SAE for.
            sae_checkpoint_path (str, optional): Direct path to the SAE checkpoint file.
            sae_step (int, optional): Step number to use if path is not provided.
        """
        layer_name = self.target_layers.get(layer_number)
        if not layer_name:
            print(f"Error: Invalid layer number {layer_number} provided to load_sae.")
            return False

        checkpoint_path_to_load = None

        # --- Prioritize the provided path ---
        if sae_checkpoint_path:
            if os.path.exists(sae_checkpoint_path):
                checkpoint_path_to_load = sae_checkpoint_path
                print(f"Using provided SAE checkpoint path: {checkpoint_path_to_load}")
            else:
                print(f"Error: Provided SAE checkpoint path not found: {sae_checkpoint_path}")
                return False
        # --- Fallback to step-based path construction ---
        else:
            constructed_path = f"checkpoints/layer_{layer_number}_{layer_name}/sae_checkpoint_step_{sae_step}.pt"
            if os.path.exists(constructed_path):
                checkpoint_path_to_load = constructed_path
                print(f"Using constructed SAE checkpoint path (step {sae_step}): {checkpoint_path_to_load}")
            else:
                # Don't print warning here, let the calling function handle missing files if needed
                pass # Path doesn't exist, will return False later

        # --- Load if a valid path was determined ---
        if checkpoint_path_to_load:
            print(f"Loading SAE for layer {layer_name} from {checkpoint_path_to_load}")
            try:
                sae = load_sae_from_checkpoint(checkpoint_path_to_load).to(self.device)
                self.saes[layer_number] = sae
                return True
            except Exception as e:
                print(f"Error loading SAE from {checkpoint_path_to_load}: {e}")
                return False
        else:
            # If neither provided path worked nor constructed path exists
            print(f"Warning: No valid SAE checkpoint found for layer {layer_name} (path: {sae_checkpoint_path}, step: {sae_step})")
            return False
    
    def _get_module(self, model, layer_name):
        """Get a module from the model using its layer name"""
        module = model
        for element in layer_name.split("."):
            if "[" in element:
                base, idx = element.rstrip("]").split("[")
                module = getattr(module, base)[int(idx)]
            else:
                module = getattr(module, element)
        return module
    
    def _hook_sae_activations(self, layer_number):
        """Create a hook function for the specified layer"""
        def hook(module, input, output):
            with torch.no_grad():
                output = output.to(self.device)
                sae = self.saes[layer_number].to(self.device)
                _, _, acts, _ = sae(output)
                
                layer_name = self.target_layers[layer_number]
                if layer_name not in self.all_activations:
                    self.all_activations[layer_name] = []
                
                self.all_activations[layer_name].append(acts.squeeze().cpu())
            
            return output
        return hook
    
    def register_hooks(self):
        """Register hooks for all loaded SAEs"""
        self.remove_hooks()
        
        for layer_number in self.saes:
            layer_name = ordered_layer_names[layer_number]
            module = self._get_module(self.model, layer_name)
            
            hook_fn = self._hook_sae_activations(layer_number)
            handle = module.register_forward_hook(hook_fn)
            self.handles.append(handle)
            
            print(f"Registered hook for layer {self.target_layers[layer_number]}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("Removed all hooks")
    
    def reset_activations(self):
        """Clear stored activations"""
        self.all_activations = {}
    
    def run_entity_tracking_experiment(self, entity1_code=4, entity2_code=None, output_path="entity_tracking_results"):
        """
        Run an experiment to track how SAE activations correlate with entity positions.
        
        Args:
            entity1_code (int): Code of primary entity to track (default: 4, blue key)
            entity2_code (int): Code of secondary entity (optional)
            output_path (str): Directory to save results
        
        Returns:
            dict: Results showing which channels correlate with entity positions
        """
        # Map entity codes to their descriptions (for reporting purposes)
        entity_code_description = {
            3: "gem",
            4: "blue_key",
            5: "green_key",
            6: "red_key",
            7: "blue_lock",
            8: "green_lock",
            9: "red_lock"
        }
        
        entity_desc_filename = entity_code_description.get(entity1_code, f"entity_{entity1_code}")
        output_path = f"{output_path}_{entity_desc_filename}"
        os.makedirs(output_path, exist_ok=True)
        
        self.reset_activations()
        
        if not self.handles:
            self.register_hooks()
        
        print(f"Creating maze sequence with entity codes: primary={entity1_code}, secondary={entity2_code}")
        observations, venv = create_example_maze_sequence(entity1=entity1_code, entity2=entity2_code)
        
        # Format: (y, x) - y is row, x is column (0-indexed)
        known_positions = [
            (1, 3),  # Starting position - middle of top row
            (1, 4),  # Move right
            (1, 5),  # Far right of top row
            (2, 5),  # Down to right edge
            (3, 5),  # Continue down right edge
            (4, 5),  # Bottom right corner
            (4, 4),  # Move left
            (4, 3),  # Continue left
            (4, 2),  # Continue more left
            (4, 1),  # Far left of bottom row
            (3, 1)   # Move up on left side
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
                y_pos, x_pos = 3.5, 3.5
                
            entity_x_positions.append(x_pos)
            entity_y_positions.append(y_pos)
            
            converted_obs = helpers.observation_to_rgb(obs)
            obs_tensor = torch.tensor(converted_obs, dtype=torch.float32).to(self.device)
            
            if obs_tensor.ndim == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            with torch.no_grad():
                self.model(obs_tensor)
        
        venv.close()
        
        position_file = os.path.join(output_path, f"{entity_desc_filename}_positions.csv")
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
        
        results = self.analyze_correlations_with_positions(
            entity_positions, entity_x_positions, entity_y_positions, output_path, entity_info
        )
        
        self.visualize_results(observations, entity_positions, results, output_path, entity_info)
        
        return results
    
    def analyze_correlations_with_positions(self, frame_positions, x_positions, y_positions, output_path, entity_info=None):
        """
        Analyze correlations between SAE activations and entity positions.
        Also calculates IOU scores, equivariance metrics, TopK activation 
        ratio and signal-to-noise ratio (SNR).
        
        Args:
            frame_positions (list): List of frame indices
            x_positions (list): List of x-coordinates
            y_positions (list): List of y-coordinates
            output_path (str): Directory to save results
            entity_info (dict): Information about the tracked entity
        
        Returns:
            dict: Results showing which channels correlate with entity positions
        """
        results = {}
        
        if entity_info is None:
            entity_info = {
                "entity_code": 4,
                "secondary_code": None,
                "description": "blue key",
                "filename_desc": "blue_key"
            }
        
        entity_desc = entity_info['description']
        entity_filename = entity_info['filename_desc']
        
        frame_pos = np.array(frame_positions)
        x_pos = np.array(x_positions)
        y_pos = np.array(y_positions)
        
        print("Analyzing correlations with entity position...")
        
        results_txt = os.path.join(output_path, f"{entity_filename}_detailed_results.txt")
        with open(results_txt, "w") as txt_file:
            txt_file.write("=" * 80 + "\n")
            txt_file.write(f"ENTITY TRACKING EXPERIMENT REPORT - {entity_desc.upper()}\n")
            txt_file.write("=" * 80 + "\n\n")
            
            txt_file.write("CONFIGURATION:\n")
            txt_file.write(f"Entity tracked: {entity_desc.title()} (entity code {entity_info['entity_code']})\n")
            if entity_info['secondary_code']:
                txt_file.write(f"Secondary entity: code {entity_info['secondary_code']}\n")
            txt_file.write("Mode: SAE activations\n\n")
            
            txt_file.write("METRICS USED:\n")
            txt_file.write("- Spatial overlap (IOU): How much activation overlaps with entity position\n")
            txt_file.write("- Equivariance: How well activation movement matches entity movement\n")
            txt_file.write("- TopK Activation Ratio: Ratio of activation in entity region vs. elsewhere\n")
            txt_file.write("- Signal-to-Noise Ratio (SNR): Mean activation in entity region vs. background noise\n\n")
            
            txt_file.write("RESULTS BY LAYER:\n\n")
        
        for layer_name, activations_list in self.all_activations.items():
            print(f"Analyzing layer {layer_name}...")
            
            all_acts = torch.stack(activations_list)
            num_channels = all_acts.shape[1]
            
            layer_results = {
                "frame_correlations": [],
                "iou_scores": [],
                "equivariance_scores": [],
                "topk_ratios": [],
                "snr_scores": []
            }
            
            for channel in range(num_channels):
                channel_acts = all_acts[:, channel].cpu()
                
                channel_mean_acts = channel_acts.mean(dim=(1, 2)).numpy()
                
                if len(channel_mean_acts) > 1:
                    try:
                        frame_corr, frame_p_value = pearsonr(frame_pos, channel_mean_acts)
                        layer_results["frame_correlations"].append((channel, frame_corr, frame_p_value))
                    except:
                        layer_results["frame_correlations"].append((channel, 0, 1.0))
                
                iou_scores = []
                topk_ratios = []
                snr_scores = []
                
                for i in range(len(x_pos)):
                    entity_mask = torch.zeros_like(channel_acts[i])
                    h, w = entity_mask.shape
                    y_idx, x_idx = int(y_pos[i] * h / 7), int(x_pos[i] * w / 7)
                    
                    radius = 1
                    for y in range(max(0, y_idx-radius), min(h, y_idx+radius+1)):
                        for x in range(max(0, x_idx-radius), min(w, x_idx+radius+1)):
                            entity_mask[y, x] = 1
                    
                    act = channel_acts[i]
                    norm_act = act - act.min()
                    if norm_act.max() > 0:
                        norm_act = norm_act / norm_act.max()
                    
                    threshold = 0.5
                    act_mask = (norm_act > threshold).float()
                    
                    intersection = (entity_mask * act_mask).sum().item()
                    union = (entity_mask + act_mask).clamp(0, 1).sum().item()
                    iou = intersection / union if union > 0 else 0
                    iou_scores.append(iou)
                    
                    k = max(1, int(0.1 * h * w))
                    _, topk_indices = torch.topk(act.view(-1), k)
                    topk_mask = torch.zeros_like(act.view(-1))
                    topk_mask[topk_indices] = 1
                    topk_mask = topk_mask.view(h, w)
                    
                    topk_in_entity = (topk_mask * entity_mask).sum().item()
                    topk_ratio = topk_in_entity / k if k > 0 else 0
                    topk_ratios.append(topk_ratio)
                    
                    entity_region_mean = (act * entity_mask).sum().item() 
                    entity_region_count = entity_mask.sum().item()
                    entity_mean = entity_region_mean / entity_region_count if entity_region_count > 0 else 0
                    
                    background_mask = 1.0 - entity_mask
                    background_mean = (act * background_mask).sum().item()
                    background_count = background_mask.sum().item()
                    background_mean = background_mean / background_count if background_count > 0 else 0
                    
                    snr = entity_mean / (background_mean + 1e-6) if background_mean > 0 else 0
                    snr_scores.append(snr)
                
                avg_iou = np.mean(iou_scores) if iou_scores else 0
                avg_topk_ratio = np.mean(topk_ratios) if topk_ratios else 0
                avg_snr = np.mean(snr_scores) if snr_scores else 0
                
                layer_results["iou_scores"].append((channel, avg_iou))
                layer_results["topk_ratios"].append((channel, avg_topk_ratio))
                layer_results["snr_scores"].append((channel, avg_snr))
                
                centroids = []
                for i in range(len(x_pos)):
                    act = channel_acts[i]
                    total_weight = act.sum().item()
                    if total_weight > 0:
                        h, w = act.shape
                        y_indices = torch.arange(h).float()
                        x_indices = torch.arange(w).float()
                        
                        y_centroid = (y_indices.view(-1, 1) * act).sum().item() / total_weight
                        x_centroid = (x_indices.view(1, -1) * act).sum().item() / total_weight
                        
                        y_centroid_scaled = y_centroid * 7 / h
                        x_centroid_scaled = x_centroid * 7 / w
                        
                        centroids.append((y_centroid_scaled, x_centroid_scaled))
                    else:
                        centroids.append((3.5, 3.5))
                
                if len(centroids) > 2:
                    entity_movements_x = np.diff(x_pos)
                    entity_movements_y = np.diff(y_pos)
                    
                    centroid_movements_x = np.diff([c[1] for c in centroids])
                    centroid_movements_y = np.diff([c[0] for c in centroids])
                    
                    try:
                        eq_x_corr, _ = pearsonr(entity_movements_x, centroid_movements_x)
                        eq_y_corr, _ = pearsonr(entity_movements_y, centroid_movements_y)
                        
                        equivariance = (eq_x_corr + eq_y_corr) / 2
                    except:
                        equivariance = 0
                else:
                    equivariance = 0
                
                layer_results["equivariance_scores"].append((channel, equivariance))
            
            layer_results["frame_correlations"].sort(key=lambda f: abs(f[1]), reverse=True)
            layer_results["iou_scores"].sort(key=lambda x: x[1], reverse=True)
            layer_results["equivariance_scores"].sort(key=lambda x: abs(x[1]), reverse=True)
            layer_results["topk_ratios"].sort(key=lambda x: x[1], reverse=True)
            layer_results["snr_scores"].sort(key=lambda x: x[1], reverse=True)
            
            results[layer_name] = {
                "frame_correlations": layer_results["frame_correlations"],
                "iou_scores": layer_results["iou_scores"],
                "equivariance_scores": layer_results["equivariance_scores"],
                "topk_ratios": layer_results["topk_ratios"],
                "snr_scores": layer_results["snr_scores"],
                "top_frame_channels": [c[0] for c in layer_results["frame_correlations"][:10]],
                "top_iou_channels": [c[0] for c in layer_results["iou_scores"][:10]],
                "top_equivariance_channels": [c[0] for c in layer_results["equivariance_scores"][:10]],
                "top_topk_channels": [c[0] for c in layer_results["topk_ratios"][:10]],
                "top_snr_channels": [c[0] for c in layer_results["snr_scores"][:10]]
            }
            
            with open(f"{output_path}/{entity_filename}_{layer_name}_frame_correlations.csv", "w") as f:
                f.write("Channel,Correlation,P-Value\n")
                for channel, corr, p_value in layer_results["frame_correlations"]:
                    f.write(f"{channel},{corr},{p_value}\n")
            
            with open(f"{output_path}/{entity_filename}_{layer_name}_iou_scores.csv", "w") as f:
                f.write("Channel,IOU_Score\n")
                for channel, score in layer_results["iou_scores"]:
                    f.write(f"{channel},{score}\n")
            
            with open(f"{output_path}/{entity_filename}_{layer_name}_equivariance_scores.csv", "w") as f:
                f.write("Channel,Equivariance_Score\n")
                for channel, score in layer_results["equivariance_scores"]:
                    f.write(f"{channel},{score}\n")
            
            with open(f"{output_path}/{entity_filename}_{layer_name}_topk_ratios.csv", "w") as f:
                f.write("Channel,TopK_Ratio\n")
                for channel, score in layer_results["topk_ratios"]:
                    f.write(f"{channel},{score}\n")
                    
            with open(f"{output_path}/{entity_filename}_{layer_name}_snr_scores.csv", "w") as f:
                f.write("Channel,SNR_Score\n")
                for channel, score in layer_results["snr_scores"]:
                    f.write(f"{channel},{score}\n")
            
            with open(results_txt, "a") as txt_file:
                txt_file.write(f"[Layer {layer_name}]\n")
                txt_file.write("=" * 60 + "\n\n")
                
                txt_file.write("Top spatial overlap channels (activations match entity position):\n")
                for i, (channel, score) in enumerate(layer_results["iou_scores"][:10]):
                    txt_file.write(f"  {i+1}. Channel {channel}: IOU score = {score:.4f}\n")
                txt_file.write("\n")
                
                txt_file.write("Top channels by TopK activation ratio:\n")
                for i, (channel, score) in enumerate(layer_results["topk_ratios"][:10]):
                    txt_file.write(f"  {i+1}. Channel {channel}: TopK ratio = {score:.4f}\n")
                txt_file.write("\n")
                
                txt_file.write("Top channels by Signal-to-Noise Ratio (SNR):\n")
                for i, (channel, score) in enumerate(layer_results["snr_scores"][:10]):
                    txt_file.write(f"  {i+1}. Channel {channel}: SNR = {score:.4f}\n")
                txt_file.write("\n")
                
                txt_file.write("Top equivariant channels (activations move with entity):\n")
                for i, (channel, score) in enumerate(layer_results["equivariance_scores"][:10]):
                    sign = "+" if score > 0 else "-"
                    txt_file.write(f"  {i+1}. Channel {channel}: equivariance = {abs(score):.4f} ({sign})\n")
                txt_file.write("\n")
                
                txt_file.write("-" * 60 + "\n\n")
        
        with open(results_txt, "a") as txt_file:
            txt_file.write("\nRECOMMENDED INTERVENTIONS:\n\n")
            
            best_layer = None
            best_score = -1
            
            for layer_name, layer_results in results.items():
                top_iou = layer_results["iou_scores"][0][1] if layer_results["iou_scores"] else 0
                top_eq = abs(layer_results["equivariance_scores"][0][1]) if layer_results["equivariance_scores"] else 0
                top_topk = layer_results["topk_ratios"][0][1] if layer_results["topk_ratios"] else 0
                top_snr = layer_results["snr_scores"][0][1] if layer_results["snr_scores"] else 0
                
                combined_score = top_iou + top_eq + top_topk + top_snr
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_layer = layer_name
            
            if best_layer:
                txt_file.write(f"Layer {best_layer} overall has the strongest channels tracking the {entity_desc}:\n\n")
                
                txt_file.write("  Top spatial overlap channels:\n")
                for i, (channel, score) in enumerate(results[best_layer]["iou_scores"][:3]):
                    txt_file.write(f"    Channel {channel}: IOU score = {score:.4f}\n")
                txt_file.write("\n")
                
                txt_file.write("  Top TopK ratio channels:\n")
                for i, (channel, score) in enumerate(results[best_layer]["topk_ratios"][:3]):
                    txt_file.write(f"    Channel {channel}: TopK ratio = {score:.4f}\n")
                txt_file.write("\n")
                
                txt_file.write("  Top SNR channels:\n")
                for i, (channel, score) in enumerate(results[best_layer]["snr_scores"][:3]):
                    txt_file.write(f"    Channel {channel}: SNR = {score:.4f}\n")
                txt_file.write("\n")
                
                txt_file.write("  Top equivariance channels:\n")
                for i, (channel, score) in enumerate(results[best_layer]["equivariance_scores"][:3]):
                    sign = "+" if score > 0 else "-"
                    txt_file.write(f"    Channel {channel}: equivariance = {abs(score):.4f} ({sign})\n")
                txt_file.write("\n")
                
                txt_file.write("RECOMMENDED INTERVENTION CHANNELS:\n")
                
                channel_scores = {}
                all_channels = set()
                for metric in ["iou_scores", "topk_ratios", "snr_scores", "equivariance_scores"]:
                    for channel, score in results[best_layer][metric][:10]:
                        all_channels.add(channel)
                
                for channel in all_channels:
                    scores = []
                    metrics = []
                    
                    try:
                        iou_score = next(score for ch, score in results[best_layer]["iou_scores"] if ch == channel)
                        relative_rank = next(i for i, (ch, _) in enumerate(results[best_layer]["iou_scores"][:10]) if ch == channel)
                        norm_score = 1.0 - (relative_rank / 10.0)
                        scores.append(norm_score)
                        metrics.append(f"IOU: {iou_score:.3f}")
                    except (StopIteration, ValueError):
                        pass
                        
                    try:
                        topk_score = next(score for ch, score in results[best_layer]["topk_ratios"] if ch == channel)
                        relative_rank = next(i for i, (ch, _) in enumerate(results[best_layer]["topk_ratios"][:10]) if ch == channel)
                        norm_score = 1.0 - (relative_rank / 10.0)
                        scores.append(norm_score)
                        metrics.append(f"TopK: {topk_score:.3f}")
                    except (StopIteration, ValueError):
                        pass
                        
                    try:
                        snr_score = next(score for ch, score in results[best_layer]["snr_scores"] if ch == channel)
                        relative_rank = next(i for i, (ch, _) in enumerate(results[best_layer]["snr_scores"][:10]) if ch == channel)
                        norm_score = 1.0 - (relative_rank / 10.0)
                        scores.append(norm_score)
                        metrics.append(f"SNR: {snr_score:.3f}")
                    except (StopIteration, ValueError):
                        pass
                    
                    try:
                        eq_score = next(score for ch, score in results[best_layer]["equivariance_scores"] if ch == channel)
                        relative_rank = next(i for i, (ch, _) in enumerate(results[best_layer]["equivariance_scores"][:10]) if ch == channel)
                        norm_score = 1.0 - (relative_rank / 10.0)
                        scores.append(norm_score)
                        metrics.append(f"Eq: {abs(eq_score):.3f}")
                    except (StopIteration, ValueError):
                        pass
                    
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        metrics_str = ", ".join(metrics)
                        channel_scores[channel] = (avg_score, metrics_str, len(scores))
                
                top_channels = sorted(channel_scores.items(), key=lambda x: (x[1][2], x[1][0]), reverse=True)[:5]
                
                for i, (channel, (score, metrics_str, num_metrics)) in enumerate(top_channels):
                    txt_file.write(f"  {i+1}. Channel {channel}: Appears in {num_metrics} metrics, average score = {score:.3f}\n")
                    txt_file.write(f"     {metrics_str}\n")
                
                txt_file.write("\n")
                
                if top_channels:
                    best_channel = top_channels[0][0]
                    txt_file.write("Example intervention command for the best overall channel:\n")
                    txt_file.write(f"python run_sae_intervention.py --static --channel {best_channel} --position 4,4 --value 8.0 --layer_name {best_layer}\n\n")
                    
                    channels_str = ",".join(str(ch) for ch, _ in top_channels[:3])
                    positions_str = ";".join("4,4" for _ in range(min(3, len(top_channels))))
                    txt_file.write("Example intervention with multiple top channels:\n")
                    txt_file.write(f"python run_sae_intervention.py --static --channel {channels_str} --position {positions_str} --value 8.0 --layer_name {best_layer}\n\n")
        
        return results
    
    def visualize_results(self, observations, positions, results, output_path, entity_info):
        """
        Create visualizations of the results.
        
        Args:
            observations (list): List of observations
            positions (list): List of frame indices (entity_positions)
            results (dict): Results from analyze_correlations_with_positions
            output_path (str): Directory to save visualizations
            entity_info (dict): Information about the tracked entity
        """
        print("Creating visualizations...")
        
        # Create a directory for visualizations
        vis_dir = f"{output_path}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load position data from the CSV file we saved
        positions_file = os.path.join(output_path, f"{entity_info['filename_desc']}_positions.csv")
        x_positions = []
        y_positions = []
        
        with open(positions_file, "r") as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    x_positions.append(float(parts[1]))
                    y_positions.append(float(parts[2]))
        
        # Get entity description
        entity_desc = entity_info['description']
        entity_code = entity_info['entity_code']
        entity_filename = entity_info['filename_desc']
        
        # Plot entity trajectory
        plt.figure(figsize=(10, 10))
        plt.plot(x_positions, y_positions, 'b-o', label=f"Entity {entity_desc} Path")
        plt.xlim(0, 7)
        plt.ylim(0, 7)
        plt.grid(True)
        plt.title(f"Entity {entity_desc} (Code: {entity_code}) Path")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.savefig(f"{vis_dir}/{entity_filename}_path.png", dpi=150)
        plt.close()
        
        # Create charts for each layer
        for layer_name, layer_results in results.items():
            # Create frame correlation chart
            plt.figure(figsize=(12, 8))
            top_channels = layer_results["frame_correlations"][:20]
            channels = [c[0] for c in top_channels]
            correlations = [c[1] for c in top_channels]
            plt.barh(range(len(channels)), [abs(c) for c in correlations], color=['r' if c < 0 else 'g' for c in correlations])
            plt.yticks(range(len(channels)), channels)
            plt.xlabel("Frame Index Correlation (absolute)")
            plt.ylabel("Channel")
            plt.title(f"{entity_desc.title()} - Top Frame-Correlating Channels for Layer {layer_name}")
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/{entity_filename}_{layer_name}_frame_correlations.png", dpi=150)
            plt.close()
            
            # Create IOU chart
            plt.figure(figsize=(12, 8))
            top_channels = layer_results["iou_scores"][:20]
            channels = [c[0] for c in top_channels]
            scores = [c[1] for c in top_channels]
            plt.barh(range(len(channels)), scores, color='blue')
            plt.yticks(range(len(channels)), channels)
            plt.xlabel("IOU Score")
            plt.ylabel("Channel")
            plt.title(f"{entity_desc.title()} - Top IOU Scoring Channels for Layer {layer_name}")
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/{entity_filename}_{layer_name}_iou_scores.png", dpi=150)
            plt.close()
            
            # Create equivariance chart
            plt.figure(figsize=(12, 8))
            top_channels = layer_results["equivariance_scores"][:20]
            channels = [c[0] for c in top_channels]
            scores = [c[1] for c in top_channels]
            plt.barh(range(len(channels)), [abs(s) for s in scores], color=['r' if s < 0 else 'g' for s in scores])
            plt.yticks(range(len(channels)), channels)
            plt.xlabel("Equivariance Score (absolute)")
            plt.ylabel("Channel")
            plt.title(f"{entity_desc.title()} - Top Equivariant Channels for Layer {layer_name}")
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/{entity_filename}_{layer_name}_equivariance_scores.png", dpi=150)
            plt.close()
            
            # Create TopK Activation Ratio chart
            plt.figure(figsize=(12, 8))
            top_channels = layer_results["topk_ratios"][:20]
            channels = [c[0] for c in top_channels]
            scores = [c[1] for c in top_channels]
            plt.barh(range(len(channels)), scores, color='purple')
            plt.yticks(range(len(channels)), channels)
            plt.xlabel("TopK Activation Ratio")
            plt.ylabel("Channel")
            plt.title(f"{entity_desc.title()} - Top TopK Ratio Channels for Layer {layer_name}")
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/{entity_filename}_{layer_name}_topk_ratios.png", dpi=150)
            plt.close()
            
            # Create SNR chart
            plt.figure(figsize=(12, 8))
            top_channels = layer_results["snr_scores"][:20]
            channels = [c[0] for c in top_channels]
            scores = [c[1] for c in top_channels]
            plt.barh(range(len(channels)), scores, color='orange')
            plt.yticks(range(len(channels)), channels)
            plt.xlabel("Signal-to-Noise Ratio (SNR)")
            plt.ylabel("Channel")
            plt.title(f"{entity_desc.title()} - Top SNR Channels for Layer {layer_name}")
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/{entity_filename}_{layer_name}_snr_scores.png", dpi=150)
            plt.close()
        
        # Create activation visualizations for the top channels
        for layer_name, layer_results in results.items():
            activations_list = self.all_activations[layer_name]
            
            # Get top channels from different metrics (avoiding duplicates)
            top_iou_channels = layer_results["top_iou_channels"][:1]
            top_topk_channels = [c for c in layer_results["top_topk_channels"][:1] if c not in top_iou_channels]
            top_snr_channels = [c for c in layer_results["top_snr_channels"][:1] if c not in top_iou_channels and c not in top_topk_channels]
            top_eq_channels = [c for c in layer_results["top_equivariance_channels"][:1] if c not in top_iou_channels and c not in top_topk_channels and c not in top_snr_channels]
            
            # Add top frame-correlated channels if we have space
            top_frame_channels = [c for c in layer_results["top_frame_channels"][:1] if c not in top_iou_channels and c not in top_topk_channels and c not in top_snr_channels and c not in top_eq_channels]
            
            # Combined unique list of top channels
            top_channels = top_iou_channels + top_topk_channels + top_snr_channels + top_eq_channels + top_frame_channels
            if len(top_channels) > 6:
                top_channels = top_channels[:6]
            
            # Create a visualization showing how these channels change over time
            frames = []
            
            for i, (obs, acts) in enumerate(zip(observations, activations_list)):
                # Determine how many rows and columns we need based on number of channels
                # We want to display the original observation + all top channels
                n_channels = len(top_channels)
                n_cols = min(3, n_channels + 1)  # Max 3 columns
                n_rows = (n_channels + n_cols) // n_cols  # Ceiling division to get number of rows needed
                
                # Create a figure with the right dimensions
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
                
                # Convert to numpy array if we have a single subplot
                if n_rows == 1 and n_cols == 1:
                    axs = np.array([[axs]])
                elif n_rows == 1:
                    axs = axs.reshape(1, -1)
                elif n_cols == 1:
                    axs = axs.reshape(-1, 1)
                
                # Show the original observation with the entity's position in the first subplot
                axs[0, 0].imshow(obs.squeeze().transpose(1, 2, 0))
                if i < len(x_positions) and i < len(y_positions):
                    axs[0, 0].plot(x_positions[i], y_positions[i], 'ro', markersize=10)
                axs[0, 0].set_title(f"Frame {i}")
                axs[0, 0].axis("off")
                
                # Show activations for top channels in the remaining subplots
                for j, channel in enumerate(top_channels):
                    # Calculate row and column for this channel's subplot
                    # We add 1 to j because we've used (0,0) for the original observation
                    row, col = (j + 1) // n_cols, (j + 1) % n_cols
                    
                    # Get activation for this channel
                    act = acts[channel].numpy()
                    
                    # Plot activation
                    im = axs[row, col].imshow(act, cmap='viridis')
                    # Add entity position on the activation map
                    if i < len(x_positions) and i < len(y_positions):
                        # Scale the position to the activation map dimensions
                        h, w = act.shape
                        scaled_x = x_positions[i] * w / 7  # Assuming 7x7 maze
                        scaled_y = y_positions[i] * h / 7
                        axs[row, col].plot(scaled_x, scaled_y, 'ro', markersize=5)
                    
                    # Determine which dimension(s) this channel correlates with
                    metric_info = []
                    if channel in layer_results["top_iou_channels"][:5]:
                        iou_score = next(score for ch, score in layer_results["iou_scores"] if ch == channel)
                        metric_info.append(f"IOU={iou_score:.3f}")
                    if channel in layer_results["top_topk_channels"][:5]:
                        topk_score = next(score for ch, score in layer_results["topk_ratios"] if ch == channel)
                        metric_info.append(f"TopK={topk_score:.3f}")
                    if channel in layer_results["top_snr_channels"][:5]:
                        snr_score = next(score for ch, score in layer_results["snr_scores"] if ch == channel)
                        metric_info.append(f"SNR={snr_score:.3f}")
                    if channel in layer_results["top_equivariance_channels"][:5]:
                        eq_score = next(score for ch, score in layer_results["equivariance_scores"] if ch == channel)
                        metric_info.append(f"Eq={eq_score:.3f}")
                    if channel in layer_results["top_frame_channels"][:5]:
                        metric_info.append("Frame")
                    
                    metrics_label = ", ".join(metric_info)
                    
                    axs[row, col].set_title(f"Channel {channel}\n{metrics_label}")
                    fig.colorbar(im, ax=axs[row, col])
                
                # Hide any unused subplots
                for j in range(len(top_channels) + 1, n_rows * n_cols):
                    row, col = j // n_cols, j % n_cols
                    axs[row, col].axis('off')
                
                # Save the figure to a buffer
                fig.tight_layout()
                fig.canvas.draw()
                
                # Convert figure to numpy array
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                frames.append(img)
                plt.close(fig)
            
            # Create a GIF of the visualizations
            if frames:
                imageio.mimsave(f"{vis_dir}/{entity_filename}_{layer_name}_top_channels.gif", frames, fps=2)
        
        # Create one aggregated visualization showing the top channel from each layer
        if len(results) > 0:
            print("Creating aggregated visualization across layers...")
            frames = []
            
            # Determine how many layers we have
            n_layers = len(results)
            
            for i, obs in enumerate(observations):
                # Create a figure with the observation and top channel from each layer
                fig, axs = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))
                
                # Make axs a 2D array for consistent indexing
                if n_layers == 0:  # Just in case
                    axs = np.array([[axs]])
                else:
                    axs = np.array([axs])
                
                # Show the original observation with the entity position
                axs[0, 0].imshow(obs.squeeze().transpose(1, 2, 0))
                if i < len(x_positions) and i < len(y_positions):
                    axs[0, 0].plot(x_positions[i], y_positions[i], 'ro', markersize=10)
                axs[0, 0].set_title(f"Frame {i}")
                axs[0, 0].axis("off")
                
                # Show top channel from each layer
                for j, (layer_name, layer_results) in enumerate(results.items()):
                    if layer_name in self.all_activations and i < len(self.all_activations[layer_name]):
                        # Get activation for the top channel - prefer IOU channels
                        if "top_iou_channels" in layer_results and layer_results["top_iou_channels"]:
                            top_channel = layer_results["top_iou_channels"][0]
                            metric = "IOU"
                        elif "top_equivariance_channels" in layer_results and layer_results["top_equivariance_channels"]:
                            top_channel = layer_results["top_equivariance_channels"][0]
                            metric = "Eq"
                        elif "top_frame_channels" in layer_results and layer_results["top_frame_channels"]:
                            top_channel = layer_results["top_frame_channels"][0]
                            metric = "Frame"
                        else:
                            top_channel = layer_results["top_iou_channels"][0]
                            metric = "IOU"
                            
                        act = self.all_activations[layer_name][i][top_channel].numpy()
                        
                        # Plot activation
                        im = axs[0, j+1].imshow(act, cmap='viridis')
                        axs[0, j+1].set_title(f"{layer_name}\nCh {top_channel} ({metric})")
                        fig.colorbar(im, ax=axs[0, j+1])
                        
                        # Mark the entity position
                        if i < len(x_positions) and i < len(y_positions):
                            # Scale to activation dimensions
                            h, w = act.shape
                            scaled_x = x_positions[i] * w / 7
                            scaled_y = y_positions[i] * h / 7
                            axs[0, j+1].plot(scaled_x, scaled_y, 'ro', markersize=5)
                
                # Save the figure to a buffer
                fig.tight_layout()
                fig.canvas.draw()
                
                # Convert figure to numpy array
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                frames.append(img)
                plt.close(fig)
            
            # Create a GIF of the aggregated visualization
            if frames:
                imageio.mimsave(f"{vis_dir}/{entity_filename}_all_layers_top_channels.gif", frames, fps=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Track which SAE channels correlate with entity positions")
    
    parser.add_argument("--model_path", type=str, default="../model_interpretable.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--entity1_code", type=int, default=4, choices=[3,4,5,6,7,8,9],
                        help="Primary entity code to track (default: 4, blue key)")
    parser.add_argument("--entity2_code", type=int, default=None, choices=[None,3,4,5,6,7,8,9],
                        help="Secondary entity code (optional)")
    parser.add_argument("--output", type=str, default="entity_tracking_results",
                        help="Directory to save results")
    parser.add_argument("--sae_checkpoint_path", type=str, default=None,
                        help="Optional: Full path to a specific SAE checkpoint file to use. Overrides --sae_step.")
    parser.add_argument("--layer_number_for_sae", type=int, default=None,
                        help="Layer number corresponding to the --sae_checkpoint_path (required if path is set).")
    parser.add_argument("--sae_step", type=int, default=15000000,
                        help="Step number for SAE checkpoints (default: 15000000). Ignored if --sae_checkpoint_path is set.")
    
    args = parser.parse_args()

    # --- Add validation ---
    if args.sae_checkpoint_path and args.layer_number_for_sae is None:
        parser.error("--layer_number_for_sae is required when using --sae_checkpoint_path")
    if not args.sae_checkpoint_path and args.layer_number_for_sae is not None:
        print("Warning: --layer_number_for_sae is ignored when --sae_checkpoint_path is not set.")
    # --- End validation ---

    return args

def main():
    args = parse_args()
    
    entity_code_description = {
        3: "gem",
        4: "blue_key",
        5: "green_key",
        6: "red_key",
        7: "blue_lock",
        8: "green_lock",
        9: "red_lock"
    }
    
    entity_desc_filename = entity_code_description.get(args.entity1_code, f"entity_{args.entity1_code}")
    entity_desc_display = entity_desc_filename.replace("_", " ")
    output_path = f"{args.output}_{entity_desc_filename}"
    
    print("=" * 80)
    print("ENTITY TRACKING EXPERIMENT")
    print("=" * 80)
    print(f"This experiment will analyze which SAE channels correlate with entity movement.")
    print(f"We'll track the {entity_desc_display} (code {args.entity1_code}) as it moves in a predefined path around the maze.")
    
    if args.entity2_code is not None:
        entity2_desc_filename = entity_code_description.get(args.entity2_code, f"entity_{args.entity2_code}")
        entity2_desc_display = entity2_desc_filename.replace("_", " ")
        print(f"A secondary entity ({entity2_desc_display}, code {args.entity2_code}) will also be in the maze.")
    
    print(f"For each layer's SAE, we'll analyze which channels' activations correlate with:")
    print(f"  - Frame sequence (time)")
    print(f"  - Spatial overlap (IOU) with the {entity_desc_display}")
    print(f"  - Equivariance (whether activations move with the entity)")
    print("-" * 80)
    
    experiment = EntityTrackingExperiment(model_path=args.model_path)

    print("Loading SAE models:")
    loaded_layers_nums = [] # Store layer numbers that were loaded

    # --- Adjust SAE loading logic ---
    if args.sae_checkpoint_path:
        # If a specific path is given, load only that layer
        print(f"Attempting to load specified SAE checkpoint for layer {args.layer_number_for_sae}: {args.sae_checkpoint_path}")

        # Check if the specified layer number is valid according to the experiment's initial setup
        if args.layer_number_for_sae not in experiment.target_layers:
             print(f"Error: Provided layer number {args.layer_number_for_sae} is not in the initially defined target layers: {list(experiment.target_layers.keys())}")
        else:
             # Attempt to load the specified SAE
             if experiment.load_sae(args.layer_number_for_sae, sae_checkpoint_path=args.sae_checkpoint_path):
                  loaded_layers_nums.append(args.layer_number_for_sae)
                  # --- Crucially, modify the experiment to only target this loaded layer ---
                  target_layer_name = experiment.target_layers[args.layer_number_for_sae]
                  experiment.target_layers = {args.layer_number_for_sae: target_layer_name}
                  print(f"Successfully loaded. Experiment will now ONLY target layer {args.layer_number_for_sae} ({target_layer_name}).")
                  # ---
             else:
                  # load_sae already prints errors/warnings
                  pass # Loading failed

    else:
        # Original logic: Try multiple steps for all target layers if a specific path isn't given
        sae_steps_to_try = [args.sae_step, 1000000, 5000000, 500000] # Example steps
        print(f"No specific SAE path provided. Trying steps {sae_steps_to_try} for layers: {list(experiment.target_layers.keys())}")

        initial_target_layers = list(experiment.target_layers.keys()) # Copy keys before potential modification
        for layer_num in initial_target_layers:
            layer_name = experiment.target_layers[layer_num]
            print(f"  - Attempting to load layer {layer_num} ({layer_name})...")
            loaded_successfully = False
            for step in sae_steps_to_try:
                # Pass None for path, so it uses the step
                if experiment.load_sae(layer_num, sae_checkpoint_path=None, sae_step=step):
                    loaded_layers_nums.append(layer_num)
                    loaded_successfully = True
                    print(f"    Successfully loaded SAE for {layer_name} with step {step}")
                    break # Stop trying steps for this layer once loaded
            if not loaded_successfully:
                 print(f"    Could not load SAE for layer {layer_num} ({layer_name}) using any of the specified steps.")
                 # Optionally remove the layer from target_layers if it couldn't be loaded
                 # del experiment.target_layers[layer_num]
    # --- End Adjust SAE loading logic ---


    if not loaded_layers_nums: # Check if the list is empty
        print("\nError: No SAE models could be loaded. Exiting.")
        return

    # Use the potentially modified experiment.target_layers to report loaded layers
    loaded_layer_names = [experiment.target_layers[num] for num in loaded_layers_nums]
    print(f"\nSuccessfully loaded SAEs for layers: {loaded_layer_names} (Numbers: {loaded_layers_nums})")
    print(f"Experiment will analyze these layers: {list(experiment.target_layers.values())}")
    print("-" * 80)

    # --- The rest of the script now uses the potentially filtered experiment.target_layers ---
    print(f"Registering hooks for loaded layers...")
    experiment.register_hooks() # Will only register for layers remaining in experiment.target_layers and experiment.saes

    print(f"\nRunning simulation tracking the {entity_desc_display} (code {args.entity1_code})...")
    results = experiment.run_entity_tracking_experiment(
        entity1_code=args.entity1_code,
        entity2_code=args.entity2_code,
        output_path=args.output
    )

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY FOR {entity_desc_display.upper()}")
    print("=" * 80)
    
    for layer_name, layer_results in results.items():
        print(f"\n[Layer {layer_name}]")
        
        print("  Top spatial overlap channels:")
        for i, (channel, score) in enumerate(layer_results["iou_scores"][:5]):
            print(f"    {i+1}. Channel {channel}: IOU score = {score:.4f}")
        
        print("\n  Top equivariant channels (activations move with entity):")
        for i, (channel, score) in enumerate(layer_results["equivariance_scores"][:5]):
            sign = "+" if score > 0 else "-"
            print(f"    {i+1}. Channel {channel}: equivariance = {abs(score):.4f} ({sign})")
        
        print("\n  Top frame-sequence correlating channels:")
        for i, (channel, corr, p_value) in enumerate(layer_results["frame_correlations"][:5]):
            corr_type = "+" if corr > 0 else "-"
            print(f"    {i+1}. Channel {channel}: correlation = {corr:.4f} ({corr_type}), p-value = {p_value:.4f}")
    
    experiment.remove_hooks()
    
    print("\n" + "-" * 80)
    print(f"Results saved to {output_path}")
    print(f"Detailed analysis saved to {os.path.join(output_path, f'{entity_desc_filename}_detailed_results.txt')}")
    print(f"Visualizations saved to {os.path.join(output_path, 'visualizations')}")
    print("\nRECOMMENDED NEXT STEPS:")
    print("1. Review the GIFs in the visualizations directory to see how channels activate")
    print("2. Read the detailed_results.txt file for comprehensive analysis")
    print(f"3. Try running the spatial intervention experiment with the top channels for {entity_desc_display}:")
    
    best_layer = None
    best_channel = None
    best_score = -1
    
    for layer_name, layer_results in results.items():
        if "iou_scores" in layer_results and layer_results["iou_scores"]:
            channel, score = layer_results["iou_scores"][0]
            if score > best_score:
                best_score = score
                best_layer = layer_name
                best_channel = channel
    
    if best_layer and best_channel is not None:
        print(f"   python run_sae_intervention.py --static --channel {best_channel} --position 4,4 --value 8.0 --layer_name {best_layer}")
    else:
        print("   python run_sae_intervention.py --static --channel <channel_number> --position 4,4 --value 8.0 --layer_name <layer_name>")
    
    print("=" * 80)

if __name__ == "__main__":
    main() 