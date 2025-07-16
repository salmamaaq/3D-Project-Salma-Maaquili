
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Assuming PointNextEncoder is defined in pointnext_model.py
from pointnext_model import PointNextEncoder

class BYOLProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.net(x)

class BYOLPredictionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, encoder: PointNextEncoder, projection_hidden_size: int, projection_out_size: int):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)

        # Freeze target encoder parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Get the output feature size of the encoder
        # Assuming the encoder outputs a global feature vector
        encoder_output_size = 1024 # Based on PointNextEncoder global_mlp output

        self.online_projection_head = BYOLProjectionHead(encoder_output_size, projection_hidden_size, projection_out_size)
        self.online_prediction_head = BYOLPredictionHead(projection_out_size, projection_hidden_size, projection_out_size)

        self.target_projection_head = BYOLProjectionHead(encoder_output_size, projection_hidden_size, projection_out_size)

        # Freeze target projection head parameters
        for param in self.target_projection_head.parameters():
            param.requires_grad = False

    def _get_representation(self, x):
        return self.online_encoder(x)

    def _get_target_representation(self, x):
        return self.target_encoder(x)

    def forward(self, x1, x2):
        # Online network forward pass
        online_proj_one = self.online_prediction_head(self.online_projection_head(self.online_encoder(x1)))
        online_proj_two = self.online_prediction_head(self.online_projection_head(self.online_encoder(x2)))

        # Target network forward pass
        with torch.no_grad():
            target_proj_one = self.target_projection_head(self.target_encoder(x1))
            target_proj_two = self.target_projection_head(self.target_encoder(x2))

        # Calculate loss (symmetric)
        loss = self.loss_fn(online_proj_one, target_proj_two.detach()) + \
               self.loss_fn(online_proj_two, target_proj_one.detach())

        return loss

    def loss_fn(self, x, y):
        # Normalized MSE loss
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1).mean()

    def update_target_network(self, tau):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
        for online_param, target_param in zip(self.online_projection_head.parameters(), self.target_projection_head.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data

if __name__ == '__main__':
    # Example usage
    batch_size = 2
    num_points = 1024
    input_dim = 3

    # Create a dummy PointNextEncoder
    encoder = PointNextEncoder(in_channels=input_dim)

    # Instantiate BYOL model
    byol_model = BYOL(encoder, projection_hidden_size=256, projection_out_size=128)

    # Create dummy augmented input point clouds
    x1 = torch.randn(batch_size, num_points, input_dim)
    x2 = torch.randn(batch_size, num_points, input_dim)

    # Forward pass
    loss = byol_model(x1, x2)
    print(f"BYOL Loss: {loss.item()}")

    # Simulate an optimization step and target network update
    optimizer = torch.optim.Adam(byol_model.online_encoder.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    byol_model.update_target_network(tau=0.996)
    print("Target network updated.")



