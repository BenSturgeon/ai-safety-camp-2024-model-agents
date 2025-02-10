# %%
import torch
from torchvision.models import googlenet
from lucent.optvis import render, objectives

# Load pretrained GoogLeNet/InceptionV1
model = googlenet(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# Example feature visualization for channel 476 in inception4a layer
objective = objectives.channel('inception4a', 476)  # Visualize channel 476 in inception4a layer
list_of_images = render.render_vis(model, objective)  # Generate visualization using Lucent's render_vis
# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.imshow(list_of_images[0][0])  # Show first image from the list
plt.axis('off')
plt.title('Channel 476 in inception4a layer')
plt.show()
# %%
