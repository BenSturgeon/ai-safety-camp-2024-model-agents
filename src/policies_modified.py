import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, obs_space, num_outputs, conv_dropout_prob=0.1, fc_dropout_prob=0.5):
        super(CustomCNN, self).__init__()

        h, w, c = obs_space.shape
        self.num_outputs = num_outputs

        self.conv1a = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.LPPool2d(2, kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.LPPool2d(2, kernel_size=2, stride=2)

        self.conv3a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool3 = nn.LPPool2d(2, kernel_size=2, stride=2)

        self.conv4a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.pool4 = nn.LPPool2d(2, kernel_size=2, stride=2)

        # Compute the flattened dimension after convolutions and pooling
        self.flattened_dim = self._get_flattened_dim(h, w)

        self.fc1 = nn.Linear(in_features=self.flattened_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_outputs)

        self.value_fc = nn.Linear(in_features=512, out_features=1)
        
        self.dropout_conv = nn.Dropout2d(p=conv_dropout_prob)
        self.dropout_fc = nn.Dropout(p=fc_dropout_prob)

    def _get_flattened_dim(self, h, w):
        x = torch.zeros(1, 3, h, w)  # Dummy input to compute the shape
        x = self.pool1(self.conv1a(x))
        x = self.pool2(self.conv2b(self.conv2a(x)))
        x = self.pool3(self.conv3a(x))
        x = self.pool4(self.conv4a(x))
        return x.numel()

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW

        x = torch.relu(self.conv1a(x))
        x = self.pool1(x)
        x = self.dropout_conv(x)

        x = torch.relu(self.conv2a(x))
        x = torch.relu(self.conv2b(x))
        x = self.pool2(x)
        x = self.dropout_conv(x)

        x = torch.relu(self.conv3a(x))
        x = self.pool3(x)
        x = self.dropout_conv(x)

        x = torch.relu(self.conv4a(x))
        x = self.pool4(x)
        x = self.dropout_conv(x)

        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout_fc(x)
        
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
