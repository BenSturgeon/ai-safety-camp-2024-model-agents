import torch
import torch.nn as nn


class CustomCNNNoDropout(nn.Module):
    def __init__(
        self, obs_space, num_outputs
    ):
        super(CustomCNNNoDropout, self).__init__()

        # Handle HWC (height, width, channel) and CHW (channel, height, width) data formats.
        # procgen returns (64, 64, 3) -> HWC
        # envpool returns (3, 64, 64) -> CHW
        shape = obs_space.shape
        if len(shape) == 3 and shape[2] == 3:  # HWC
            h, w, c = shape
        elif len(shape) == 3 and shape[0] == 3:  # CHW
            c, h, w = shape
        else:
            raise ValueError(f"Unsupported observation space shape: {shape}")

        self.num_outputs = num_outputs

        self.conv1a = nn.Conv2d(
            in_channels=c, out_channels=16, kernel_size=7, padding=3
        )
        self.pool1 = nn.LPPool2d(2, kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding=2
        )
        self.conv2b = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, padding=2
        )
        self.pool2 = nn.LPPool2d(2, kernel_size=2, stride=2)

        self.conv3a = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, padding=2
        )
        self.pool3 = nn.LPPool2d(2, kernel_size=2, stride=2)

        self.conv4a = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, padding=2
        )
        self.pool4 = nn.LPPool2d(2, kernel_size=2, stride=2)

        # Compute the flattened dimension after convolutions and pooling
        self.flattened_dim = self._get_flattened_dim(h, w)

        self.fc1 = nn.Linear(in_features=self.flattened_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_outputs)

        self.value_fc = nn.Linear(in_features=512, out_features=1)

    def _get_flattened_dim(self, h, w):
        x = torch.zeros(1, 3, h, w)  # Dummy input to compute the shape
        x = self.pool1(self.conv1a(x))
        x = self.pool2(self.conv2b(self.conv2a(x)))
        x = self.pool3(self.conv3a(x))
        x = self.pool4(self.conv4a(x))
        return x.numel()

    def forward(self, obs):
        assert obs.ndim == 4, f"Expected 4D input, got {obs.ndim}D"
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs)

        # Check for different input formats and convert if necessary
        if obs.shape[1:] != (3, 64, 64):
            if obs.shape[1:] == (64, 3, 64):  # NHWC format
                obs = obs.permute(0, 2, 1, 3)
            elif obs.shape[1:] == (64, 64, 3):  # NHWC format
                obs = obs.permute(0, 3, 1, 2)

        # Input range check and normalization
        obs_min, obs_max = obs.min(), obs.max()
        if obs_min < 0 or obs_max > 255:
            raise ValueError(
                f"Input values out of expected range. Min: {obs_min}, Max: {obs_max}"
            )
        elif obs_max <= 1:
            x = obs
        else:
            x = obs.float() / 255.0  # scale to 0-1

        x = x.to(
            self.conv1a.weight.device
        )  

        x = torch.relu(self.conv1a(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2a(x))
        x = torch.relu(self.conv2b(x))
        x = self.pool2(x)

        x = torch.relu(self.conv3a(x))
        x = self.pool3(x)

        x = torch.relu(self.conv4a(x))
        x = self.pool4(x)

        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        logits = self.fc3(x)
        dist = torch.distributions.Categorical(logits=logits)

        value = self.value_fc(x)

        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))

    def get_state_dict(self):
        return self.state_dict() 