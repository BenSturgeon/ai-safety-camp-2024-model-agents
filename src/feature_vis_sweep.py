# %%
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim

# ------------------------------
# Load Pretrained GoogLeNet (Inception V1)
# ------------------------------
model = models.googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# ------------------------------
# Regularization Functions
# ------------------------------
def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

def jitter(img, ox, oy):
    """
    Randomly shift the image by (ox, oy) pixels.
    """
    return torch.roll(img, shifts=(ox, oy), dims=(2, 3))

# ------------------------------
# Helper: Apply Color Transformation
# ------------------------------
def apply_color_correlation(x, color_matrix):
    """
    Apply a 3x3 color correlation matrix to an image tensor.
    x: tensor of shape (batch, 3, H, W)
    color_matrix: tensor of shape (3, 3)
    Returns: transformed image (batch, 3, H, W)
    """
    # Here we use einsum to apply the matrix to the channel dimension.
    # For each pixel: new_pixel[i] = sum_j color_matrix[i,j] * old_pixel[j]
    return torch.einsum('ij,bjhw->bihw', color_matrix, x)

# ------------------------------
# Feature Visualization Function with Color Matrix Integration
# ------------------------------
def visualize_inception_channel(model, target_module_name, target_channel, 
                                num_steps=300, lr=0.05, tv_weight=1e-3, l2_weight=1e-3, 
                                jitter_amount=8, device='cpu', color_matrix=None):
    """
    Optimize an input image to maximize the activation of a specific channel
    in a target module of the model. After each update step, if a color_matrix is
    provided, the image is transformed by it.
    """
    model.eval()
    model.to(device)

    # GoogLeNet expects an image of shape (1, 3, 224, 224)
    input_img = torch.rand((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = optim.Adam([input_img], lr=lr)
    
    activations = {}
    
    def hook_fn(module, input, output):
        activations[target_module_name] = output

    # Register a forward hook on the target module
    target_module = getattr(model, target_module_name)
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Apply random jitter each step
        ox, oy = np.random.randint(-jitter_amount, jitter_amount+1, 2)
        jittered_img = jitter(input_img, ox, oy)
        
        _ = model(jittered_img)
        act = activations[target_module_name]
        if isinstance(act, tuple):
            act = act[0]
        channel_activation = act[0, target_channel, :, :]
        
        # Loss: negative activation (to maximize activation) plus regularizers
        loss = -channel_activation.mean()
        loss = loss + tv_weight * total_variation(input_img)
        loss = loss + l2_weight * torch.norm(input_img)
        
        loss.backward()
        optimizer.step()
        
        # After the optimizer step, apply the color decorrelation if provided.
        with torch.no_grad():
            if color_matrix is not None:
                # Apply the matrix to each pixel's RGB values.
                input_img.data = apply_color_correlation(input_img.data, color_matrix)
            input_img.data.clamp_(0, 1)
        
        if step % 20 == 0:
            print(f"Step {step:03d} - Loss: {loss.item():.4f}")
    
    hook_handle.remove()
    
    result = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return result

# ------------------------------
# Define Hyperparameter and Color Matrix Sweeps
# ------------------------------

# Hyperparameters for regularization
tv_weights = [1e-2]  # Fixed TV weight for this sweep
l2_weights = [1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]  # Sweep over L2 weights
# We'll keep jitter constant here (or you can add an extra loop if desired)
jitter_amount = 4

# Define color matrices to sweep over.
# Each tuple is (name, matrix). If the matrix is None, no color transform is applied.
color_matrices = [
    ("No Color Transform", None),
    ("Lucent", torch.tensor([
            [ 0.26,  0.09,  0.02],
            [ 0.27,  0.00, -0.05],
            [ 0.27, -0.09,  0.03]
        ], device=device)),
    ("Identity", torch.eye(3, device=device))
]

# The target module and channel we want to visualize
target_module_name = 'inception4a'
target_channel = 475  # Ensure this index exists; adjust if needed

# For clarity, we will create one subplot grid per color matrix.
num_color = len(color_matrices)
num_l2 = len(l2_weights)
fig, axs = plt.subplots(num_color, num_l2, figsize=(15, 5 * num_color))
fig.suptitle(
    f"Hyperparameter Sweep for {target_module_name} Channel {target_channel}\n"
    f"TV weight fixed at: {tv_weights[0]:.0e}\nL2 weights: {l2_weights}\nJitter amount: {jitter_amount}",
    fontsize=14
)

print("TV weight:", tv_weights[0])
print("L2 weights:", l2_weights)
print("Color matrices:", [name for name, _ in color_matrices])

# Sweep over the color matrices and L2 weight values.
for i, (color_name, color_matrix) in enumerate(color_matrices):
    for j, l2_w in enumerate(l2_weights):
        print(f"\nOptimizing with Color: {color_name}, TV weight={tv_weights[0]:.0e}, L2 weight={l2_w:.0e}")
        optimized_img = visualize_inception_channel(
            model,
            target_module_name=target_module_name,
            target_channel=target_channel,
            num_steps=500,
            lr=0.05,
            tv_weight=tv_weights[0],
            l2_weight=l2_w,
            jitter_amount=jitter_amount,
            device=device,
            color_matrix=color_matrix
        )
        axs[i, j].imshow(optimized_img)
        axs[i, j].set_title(f'{color_name}\nL2 = {l2_w:.0e}')
        axs[i, j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
