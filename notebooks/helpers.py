import torch
import imageio

import sys
sys.path.append('../') #This is added so we can import from the source folder
@torch.no_grad()
def generate_action(model, observation):
    observation = torch.tensor(observation, dtype=torch.float32)
    
    if len(observation.shape) == 3:
        observation = observation.unsqueeze(0)
    
    model_output = model(observation)
    
    logits = model_output[0].logits  # discard the output of the critic in our actor critic network
    
    probabilities = torch.softmax(logits, dim=-1)
    
    # if args.argmax:
    action = probabilities.argmax(dim=-1).numpy()
    # else:
    #     action = torch.multinomial(probabilities, 1).squeeze().numpy()
    
    return action

