# %%
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

# ------------------------------
# 1. Regularization: Total Variation
# ------------------------------
def total_variation(x):
    """Encourages spatial smoothness."""
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ------------------------------
# 2. Transformation Pipeline (Lucent-inspired)
# ------------------------------
def apply_transforms_batch(x, device):
    """
    Applies a sequence of transforms to each image in the batch:
      1. Pad with 12px (constant value 0.5)
      2. Coarse jitter: shift randomly by up to ±8 pixels
      3. Random scale: choose a factor between 0.9 and 1.1
      4. Random rotation: rotate by an angle between -10° and +10°
      5. Fine jitter: shift randomly by up to ±4 pixels
    Finally, each image is resized to 224×224.
    
    x: Tensor of shape [B, 3, H, W]
    Returns: Tensor of shape [B, 3, 224, 224]
    """
    B = x.shape[0]
    transformed_list = []
    for i in range(B):
        xi = x[i:i+1]  # [1, 3, H, W]
        # 1. Padding
        padded = F.pad(xi, (12, 12, 12, 12), mode='constant', value=0.5)
        # 2. Coarse jitter (±8 px)
        ox, oy = torch.randint(-8, 9, (2,), device=device)
        jittered = torch.roll(padded, shifts=(int(ox.item()), int(oy.item())), dims=(2, 3))
        # 3. Random scale: factor between 0.9 and 1.1
        scale = 1 + (torch.randint(0, 11, (1,), device=device).item() - 5) / 50.0
        scaled = F.interpolate(jittered, scale_factor=scale, mode='bilinear', align_corners=False)
        # 4. Random rotation: angle between -10° and +10°
        angle = torch.randint(-10, 11, (1,), device=device).item()
        theta = torch.tensor([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                              [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0]],
                             device=device, dtype=torch.float).unsqueeze(0)
        grid = F.affine_grid(theta, scaled.size(), align_corners=False)
        rotated = F.grid_sample(scaled, grid, align_corners=False)
        # 5. Fine jitter (±4 px)
        ox, oy = torch.randint(-4, 5, (2,), device=device)
        final = torch.roll(rotated, shifts=(int(ox.item()), int(oy.item())), dims=(2, 3))
        # Resize to 224×224
        final = F.interpolate(final, size=(224, 224), mode='bilinear', align_corners=False)
        transformed_list.append(final)
    return torch.cat(transformed_list, dim=0)

# ------------------------------
# 3. Diversity Loss (via Gram Matrix Cosine Similarity)
# ------------------------------
def diversity_loss(activations):
    """
    Computes the average pairwise cosine similarity between the flattened
    Gram matrices of the activations. Lower similarity (more diversity) is better.
    
    activations: Tensor of shape [B, C, H, W]
    """
    B, C, H, W = activations.shape
    A = activations.view(B, C, -1)  # [B, C, H*W]
    G = torch.bmm(A, A.transpose(1, 2))  # [B, C, C]
    G_flat = G.view(B, -1)  # [B, C*C]
    
    loss_div = 0.0
    count = 0
    for i in range(B):
        for j in range(i+1, B):
            sim = torch.dot(G_flat[i], G_flat[j]) / (torch.norm(G_flat[i]) * torch.norm(G_flat[j]) + 1e-8)
            loss_div += sim
            count += 1
    if count > 0:
        loss_div = loss_div / count
    return loss_div

# ------------------------------
# 4. Advanced Visualization Function
# ------------------------------
def visualize_inception_channel_diverse(model, target_channel, num_steps=1024, batch_size=4,
                                         lr=0.05, tv_weight=1e-2, l2_weight=5e-2,
                                         diversity_weight=1e-2, device='cpu'):
    """
    Optimizes a batch of images to maximize the activation of the specified channel
    in the 'inception4a' layer of GoogLeNet (Inception V1). Regularizes with TV,
    L2, and diversity losses. After optimization, applies Lucent's color correlation
    matrix (SVD square-root of the natural image color correlation) as a post‐processing step.
    
    Returns: Optimized images (Tensor of shape [batch_size, 3, 224, 224])
    """
    model.eval()
    model.to(device)
    
    # Initialize a batch of random images (224×224)
    input_img = torch.rand((batch_size, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    # Use a forward hook to capture activations from inception4a
    activations = {}
    def hook_fn(module, inp, out):
        activations['inception4a'] = out
    hook_handle = model.inception4a.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        # Apply the transform pipeline to the current batch
        transformed_img = apply_transforms_batch(input_img, device)
        _ = model(transformed_img)  # Hook saves activations in 'activations'
        act = torch.relu(activations['inception4a'])
        main_loss = -torch.mean(act[:, target_channel, :, :])
        tv_loss = total_variation(input_img)
        l2_loss = torch.norm(input_img)
        div_loss = diversity_loss(act)
        
        total_loss = main_loss + tv_weight * tv_loss + l2_weight * l2_loss + diversity_weight * div_loss
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        if step % 50 == 0:
            print(f"Step {step:04d} - Total Loss: {total_loss.item():.4f}, "
                  f"Main: {main_loss.item():.4f}, TV: {tv_loss.item():.4f}, "
                  f"L2: {l2_loss.item():.4f}, Div: {div_loss.item():.4f}")
    
    hook_handle.remove()
    
    # ------------------------------
    # 5. Post-Processing: Apply Color Correlation Transform
    # ------------------------------
    # This matrix is the SVD square-root of the color correlation matrix derived from natural images.
    # color_correlation = torch.tensor([
    #     [0.26, 0.09, 0.02],
    #     [0.27, 0.00, -0.05],
    #     [0.27, -0.09, 0.03]
    # ], device=device, dtype=torch.float)
    
    # Option 1: Apply the given matrix (might produce red/black images)
    # output_img = torch.matmul(input_img.permute(0, 2, 3, 1), color_correlation)
    
    # Option 2: Apply the inverse of the color correlation matrix
    # inv_color_correlation = torch.inverse(color_correlation)
    # output_img = torch.matmul(input_img.permute(0, 2, 3, 1), inv_color_correlation)
    
    output_img = input_img  # Use input directly without color transformation
    # output_img = output_img.permute(0, 3, 1, 2)  # Not needed anymore since we're using input_img directly
    output_img = output_img.detach().clone()  # Create a detached copy
    output_img.clamp_(0, 1)
    # Normalize each image individually to [0,1]
    for i in range(batch_size):
        mi = output_img[i].min()
        ma = output_img[i].max()
        output_img[i] = (output_img[i] - mi) / (ma - mi + 1e-8)

    
    return output_img # [B, 3, 224, 224]

# ------------------------------
# 6. (Optional) Hyperparameter Sweep Function
# ------------------------------
def hyperparameter_sweep(model, target_channel, num_steps=512, batch_size=4, lr=0.05,
                         tv_weights=[1e-2], l2_weights=[1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2],
                         diversity_weight=1e-2, device='cpu'):
    """
    Runs a hyperparameter sweep over TV and L2 regularization weights.
    Displays the first optimized image from each run.
    """
    num_tv = len(tv_weights)
    num_l2 = len(l2_weights)
    fig, axs = plt.subplots(num_tv, num_l2, figsize=(4*num_l2, 4*num_tv))
    fig.suptitle(f"Hyperparameter Sweep: Inception4a Channel {target_channel}", fontsize=16)
    
    for i, tv_w in enumerate(tv_weights):
        for j, l2_w in enumerate(l2_weights):
            print(f"Optimizing with TV={tv_w}, L2={l2_w}")
            optimized_batch = visualize_inception_channel_diverse(model, target_channel,
                                                                   num_steps=num_steps,
                                                                   batch_size=batch_size,
                                                                   lr=lr,
                                                                   tv_weight=tv_w,
                                                                   l2_weight=l2_w,
                                                                   diversity_weight=diversity_weight,
                                                                   device=device)
            # For display, take the first image from the batch.
            img = optimized_batch[0].permute(1, 2, 0).cpu().numpy()
            if num_tv == 1 and num_l2 == 1:
                axs.imshow(img)
                axs.set_title(f"TV={tv_w}, L2={l2_w}")
                axs.axis('off')
            elif num_tv == 1:
                axs[j].imshow(img)
                axs[j].set_title(f"TV={tv_w}, L2={l2_w}")
                axs[j].axis('off')
            elif num_l2 == 1:
                axs[i].imshow(img)
                axs[i].set_title(f"TV={tv_w}, L2={l2_w}")
                axs[i].axis('off')
            else:
                axs[i, j].imshow(img)
                axs[i, j].set_title(f"TV={tv_w}, L2={l2_w}")
                axs[i, j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
# %%
# ------------------------------
# 7. Main Usage Example
# ------------------------------

# Load pretrained GoogLeNet (Inception V1) and set the device
model = models.googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# Choose the target channel (ensure this index exists in inception4a)
target_channel = 476

# Run the advanced visualization
optimized_batch = visualize_inception_channel_diverse(model, target_channel,
                                                        num_steps=512, batch_size=4,
                                                        lr=0.05, tv_weight=1e-2,
                                                        l2_weight=5e-2, diversity_weight=1e-2,
                                                        device=device)
# Display the optimized images (from the batch)
batch_size = optimized_batch.shape[0]
fig, axs = plt.subplots(1, batch_size, figsize=(4*batch_size, 4))
for i in range(batch_size):
    img = optimized_batch[i].permute(1, 2, 0).cpu().numpy()
    axs[i].imshow(img)
    axs[i].axis('off')
plt.suptitle(f"Inception4a Channel {target_channel}\nDiverse, Color-Correlated Optimizations", fontsize=16)
plt.show()

# (Optional) Run a hyperparameter sweep:
tv_weights = [1e-1]
l2_weights = [ 2e-2]
hyperparameter_sweep(model, target_channel, num_steps=256, batch_size=4, lr=0.05,
                        tv_weights=tv_weights, l2_weights=l2_weights,
                        diversity_weight=1e-2, device=device)
