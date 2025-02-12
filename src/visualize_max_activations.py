# %%
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import gym
from PIL import Image

# Import necessary functions and variables from sae_cnn and utils
from sae_cnn import generate_batch_activations_parallel, ordered_layer_names, get_device, load_interpretable_model
from utils import helpers


def extract_activation_patches(activation_map, observation, patch_size=16, stride=8):
    """
    Extract patches around high-activation regions in the activation map.
    """
    patches = []
    act_h, act_w = activation_map.shape
    
    # Ensure observation is in (64,64,3) format
    if observation.shape[0] == 64 and observation.shape[2] == 3:
        # Already in (64,64,3) format
        obs_np = observation.detach().cpu().numpy()
    elif observation.shape[0] == 3:
        # In (3,64,64) format, transpose to (64,64,3)
        obs_np = observation.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        # Try to reshape to (64,64,3)
        obs_np = observation.detach().cpu().numpy()
        if obs_np.shape[0] == 64:
            obs_np = obs_np.transpose(0, 2, 1)  # (64,3,64) -> (64,64,3)
    
    print(f"Observation shape after format conversion: {obs_np.shape}")
    if obs_np.shape != (64, 64, 3):
        raise ValueError(f"Could not convert observation to (64,64,3) format. Current shape: {obs_np.shape}")
    
    obs_h, obs_w = obs_np.shape[:2]
    
    print(f"Activation map shape: {activation_map.shape}")
    
    # Scale factors between activation map and original image
    scale_h = obs_h / act_h
    scale_w = obs_w / act_w
    
    print(f"Scale factors: h={scale_h}, w={scale_w}")
    
    # Ensure patch size isn't larger than the image
    patch_size = min(patch_size, obs_h, obs_w)
    
    # Convert patch_size and stride to activation map space
    act_patch_size = max(1, int(patch_size / scale_h))
    act_stride = max(1, int(stride / scale_h))
    
    print(f"Activation space: patch_size={act_patch_size}, stride={act_stride}")
    
    # Find regions of high activation
    flat_activations = activation_map.flatten()
    top_k = 10  # Number of high activation points to consider
    top_indices = np.argsort(flat_activations)[-top_k:]
    
    for idx in top_indices:
        # Convert flat index back to 2D coordinates
        i, j = idx // act_w, idx % act_w
        
        # Convert to image space coordinates
        img_i = int(i * scale_h)
        img_j = int(j * scale_w)
        
        # Calculate patch boundaries
        half_size = patch_size // 2
        start_i = max(0, img_i - half_size)
        start_j = max(0, img_j - half_size)
        end_i = min(obs_h, img_i + half_size)
        end_j = min(obs_w, img_j + half_size)
        
        # Extract patch from numpy array
        patch = obs_np[start_i:end_i, start_j:end_j, :]
        
        # Get activation score for this region
        act_start_i = max(0, i - act_patch_size//2)
        act_start_j = max(0, j - act_patch_size//2)
        act_end_i = min(act_h, i + act_patch_size//2)
        act_end_j = min(act_w, j + act_patch_size//2)
        region_score = activation_map[act_start_i:act_end_i, act_start_j:act_end_j].mean().item()
        
        # Resize patch to standard size if needed
        if patch.shape[:2] != (patch_size, patch_size):
            try:
                patch_pil = Image.fromarray((patch * 255).astype(np.uint8))
                patch_pil = patch_pil.resize((patch_size, patch_size), Image.BILINEAR)
                patch = np.array(patch_pil).astype(np.float32) / 255.0
            except Exception as e:
                print(f"Failed to resize patch: {e}, patch shape: {patch.shape}")
                continue
        
        # Convert back to tensor in the original format
        patch_tensor = t.from_numpy(patch).permute(2, 0, 1)  # HWC -> CHW
        patches.append((region_score, patch_tensor, (img_i, img_j)))
    
    # Sort by activation score
    patches.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Found {len(patches)} valid patches")
    return patches


def gather_max_activating_samples(model, model_activations, layer_number, iterations=20, batch_size=32, num_envs=8, 
                                episode_length=150, top_k=4, diversity_weight=2.0, patch_size=16):
    """
    Gather patches from the environment that maximally activate each channel of the target layer.
    """
    best_samples = {}
    total_samples = 0

    def cosine_similarity(patch1, patch2):
        """Compute cosine similarity between two patches"""
        vec1 = patch1.flatten()
        vec2 = patch2.flatten()
        norm1 = np.linalg.norm(vec1) + 1e-8
        norm2 = np.linalg.norm(vec2) + 1e-8
        sim = np.dot(vec1, vec2) / (norm1 * norm2)
        return sim

    for it in range(iterations):
        batch_acts, batch_obs, _ = generate_batch_activations_parallel(
            model, model_activations, layer_number,
            batch_size=batch_size, num_envs=num_envs, episode_length=episode_length
        )
        if batch_acts is None or batch_obs is None:
            continue

        _, num_channels, act_h, act_w = batch_acts.shape
        print(f"\nBatch activations shape: {batch_acts.shape}")
        print(f"Batch observations shape: {batch_obs.shape}")
        
        if not best_samples:
            for c in range(num_channels):
                best_samples[c] = []

        # Process each sample in the batch
        for b in range(batch_acts.shape[0]):
            total_samples += 1
            for c in range(num_channels):
                # Get activation map for this channel
                activation_map = batch_acts[b, c].detach().cpu().numpy()
                observation = batch_obs[b]
                
                print(f"\nProcessing channel {c}, sample {b}")
                # Extract patches around high activation regions
                patches = extract_activation_patches(activation_map, observation, 
                                                  patch_size=patch_size)
                
                if not patches:
                    continue
                
                # Process top patches
                for raw_score, patch, center in patches[:top_k*2]:  # Get more candidates for diversity
                    # For the first sample, just use raw score
                    if len(best_samples[c]) == 0:
                        effective_score = raw_score
                    else:
                        # Compute average similarity to existing samples
                        candidate_img = patch.detach().cpu().numpy()
                        similarities = []
                        for _, _, stored_patch, _ in best_samples[c]:
                            stored_img = stored_patch.detach().cpu().numpy()
                            sim = cosine_similarity(candidate_img, stored_img)
                            similarities.append(sim)
                        
                        avg_similarity = np.mean(similarities)
                        # Reward dissimilarity
                        diversity_bonus = diversity_weight * (1 - avg_similarity)
                        effective_score = raw_score + diversity_bonus
                    
                    # Update best samples
                    if len(best_samples[c]) < top_k:
                        best_samples[c].append((effective_score, raw_score, patch.clone(), center))
                    else:
                        min_effective_score = min(best_samples[c], key=lambda x: x[0])[0]
                        if effective_score > min_effective_score:
                            min_idx = np.argmin([s[0] for s in best_samples[c]])
                            best_samples[c][min_idx] = (effective_score, raw_score, patch.clone(), center)

        print(f"Iteration {it+1}/{iterations} completed. Total samples processed: {total_samples}")

    # Sort by effective score and keep top_k
    for c in best_samples:
        best_samples[c] = sorted(best_samples[c], key=lambda x: x[0], reverse=True)
        best_samples[c] = best_samples[c][:top_k]
    
    return best_samples


def visualize_max_activations(best_samples, target_layer_name):
    """Visualize the patches that maximally activate each channel."""
    print(f"Visualizing samples for {len(best_samples)} channels")
    
    for c, samples in best_samples.items():
        num_samples = len(samples)
        if num_samples == 0:
            print(f"Skipping channel {c} - no samples found")
            continue
            
        print(f"Channel {c}: Found {num_samples} samples")
        fig, axs = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
        if num_samples == 1:
            axs = [axs]
        for i, (effective_score, raw_score, patch, center) in enumerate(samples):
            img = patch.detach().cpu().numpy()  # Already in CHW format
            img = img.transpose(1, 2, 0)  # Convert to HWC for plotting
            print(f"  Sample {i}: shape {img.shape}, range [{img.min():.2f}, {img.max():.2f}]")
            axs[i].imshow(img)
            axs[i].set_title(f"Raw: {raw_score:.2f}\nEff: {effective_score:.2f}\nPos: {center}")
            axs[i].axis('off')
        fig.suptitle(f"Channel {c}: Maximally Activating Patches for {target_layer_name}")
        plt.tight_layout()
        plt.show()


def main():
    device = get_device()
    print(f"Using device: {device}")
    
    model = load_interpretable_model()
    model.to(device)
    model.eval()
    
    model_activations = helpers.ModelActivations(model)
    
    target_layer_number = 8  # conv4a
    target_layer_name = ordered_layer_names[target_layer_number]
    print(f"Gathering samples for layer {target_layer_name} (Layer number {target_layer_number})")
    
    # Now gathering patches around high activation regions
    best_samples = gather_max_activating_samples(
        model, model_activations, target_layer_number,
        iterations=20, batch_size=32, num_envs=8, episode_length=150, 
        top_k=4, diversity_weight=2.0, patch_size=32  
    )
    
    # Debug print
    print("\nSample collection summary:")
    for c in best_samples:
        print(f"Channel {c}: {len(best_samples[c])} samples")
    
    visualize_max_activations(best_samples, target_layer_name)
    model_activations.clear_hooks()

if __name__ == '__main__':
    main()
# %%
