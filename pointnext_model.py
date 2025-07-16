
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper functions for PointNext components

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        # Set track_running_stats to False and affine to False for BatchNorm1d to handle batch_size=1 during training
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=False, affine=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channels, track_running_stats=False, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.mlp(x)

class LocalGrouper(nn.Module):
    def __init__(self, radius, num_samples, min_group_size=4):
        super().__init__()
        self.radius = radius
        self.num_samples = num_samples
        self.min_group_size = min_group_size

    def forward(self, xyz, points):
        # xyz: (B, 3, N) - coordinates
        # points: (B, C, N) - features

        B, C, N = points.shape

        # For demonstration, we will simplify the grouping significantly.
        # In a real PointNext, this would involve Farthest Point Sampling (FPS)
        # and then ball queries or k-NN to find neighbors.

        # Number of new points (centroids) to sample. Ensure it\"s at least 1.
        num_new_points = max(1, N // self.num_samples)

        # Randomly sample indices for the new_xyz (centroids)
        # This is a basic downsampling, not FPS.
        new_xyz_indices = torch.randint(0, N, (B, num_new_points), device=xyz.device)
        new_xyz = torch.gather(xyz, 2, new_xyz_indices.unsqueeze(1).expand(-1, 3, -1))

        # Create dummy grouped_xyz and grouped_points with correct dimensions
        # (B, 3, num_new_points, num_samples)
        grouped_xyz = new_xyz.unsqueeze(-1).expand(-1, -1, -1, self.num_samples)

        # Randomly sample features for each \"group\". This is not spatially coherent.
        # We need to ensure that the sampled indices are within the bounds of N.
        feature_indices = torch.randint(0, N, (B, num_new_points, self.num_samples), device=points.device)
        
        # Reshape points to (B, N, C) for easier gathering
        points_transposed = points.transpose(1, 2) # (B, N, C)
        
        # To use torch.gather, the index tensor must have the same number of dimensions as the input tensor.
        # Here, points_transposed is (B, N, C). We want to gather along dim=1 (N).
        # So, feature_indices (B, num_new_points, num_samples) needs to be expanded to (B, num_new_points, num_samples, 1)
        # and then broadcasted along the last dimension (C).
        
        # First, flatten the num_new_points and num_samples dimensions of feature_indices
        flat_feature_indices = feature_indices.view(B, -1) # (B, num_new_points * num_samples)
        
        # Expand flat_feature_indices to (B, num_new_points * num_samples, C) to match points_transposed dimensions
        expanded_feature_indices = flat_feature_indices.unsqueeze(-1).expand(-1, -1, C)
        
        # Gather the features. The output will be (B, num_new_points * num_samples, C)
        gathered_features_flat = torch.gather(points_transposed, 1, expanded_feature_indices)
        
        # Reshape back to (B, num_new_points, num_samples, C) and then permute to (B, C, num_new_points, num_samples)
        grouped_points = gathered_features_flat.view(B, num_new_points, self.num_samples, C).permute(0, 3, 1, 2)

        return new_xyz, grouped_xyz, grouped_points

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, mid_channels)
        self.depthwise_conv = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.bn = nn.BatchNorm1d(mid_channels, track_running_stats=False, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels, track_running_stats=False, affine=False)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, track_running_stats=False, affine=False)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.relu(self.bn(self.depthwise_conv(x)))
        x = self.bn2(self.conv2(x))
        return x + shortcut

class PointNextEncoder(nn.Module):
    def __init__(self, in_channels=3, num_classes=40, width_multiplier=1.0, depth_multiplier=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial feature extraction
        self.embedding = ConvBNReLU(in_channels, int(64 * width_multiplier))

        # Set Abstraction layers (simplified for demonstration)
        self.sa1_grouper = LocalGrouper(radius=0.1, num_samples=32)
        self.sa1_block = InvertedResidualBlock(int(64 * width_multiplier) + 3, int(128 * width_multiplier), int(128 * width_multiplier))

        self.sa2_grouper = LocalGrouper(radius=0.2, num_samples=32)
        self.sa2_block = InvertedResidualBlock(int(128 * width_multiplier) + 3, int(256 * width_multiplier), int(256 * width_multiplier))

        self.sa3_grouper = LocalGrouper(radius=0.4, num_samples=32)
        self.sa3_block = InvertedResidualBlock(int(256 * width_multiplier) + 3, int(512 * width_multiplier), int(512 * width_multiplier))

        # Global feature aggregation (simplified)
        self.global_mlp = SharedMLP(int(512 * width_multiplier), int(1024 * width_multiplier))

    def forward(self, xyz):
        # xyz: (B, N, 3) - input point cloud coordinates
        # Transpose to (B, 3, N) for Conv1d
        xyz = xyz.transpose(1, 2)

        # Initial embedding
        features = self.embedding(xyz)

        # SA layers
        new_xyz1, grouped_xyz1, grouped_features1 = self.sa1_grouper(xyz, features)
        features1 = self.sa1_block(torch.cat([grouped_xyz1, grouped_features1], dim=1).max(dim=-1)[0])

        new_xyz2, grouped_xyz2, grouped_features2 = self.sa2_grouper(new_xyz1, features1)
        features2 = self.sa2_block(torch.cat([grouped_xyz2, grouped_features2], dim=1).max(dim=-1)[0])

        new_xyz3, grouped_xyz3, grouped_features3 = self.sa3_grouper(new_xyz2, features2)
        features3 = self.sa3_block(torch.cat([grouped_xyz3, grouped_features3], dim=1).max(dim=-1)[0])

        # Global feature
        global_feature = self.global_mlp(features3).max(dim=-1)[0] # (B, 1024)

        return global_feature

if __name__ == '__main__':
    # Example usage
    batch_size = 2
    num_points = 1024
    input_dim = 3 # x, y, z coordinates

    # Create a dummy input point cloud (B, N, 3)
    dummy_input = torch.randn(batch_size, num_points, input_dim)

    # Instantiate the PointNext encoder
    model = PointNextEncoder(in_channels=input_dim)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test with different width and depth multipliers (conceptual)
    model_small = PointNextEncoder(in_channels=input_dim, width_multiplier=0.5)
    output_small = model_small(dummy_input)
    print(f"Output shape (small model): {output_small.shape}")

    model_large = PointNextEncoder(in_channels=input_dim, width_multiplier=2.0)
    output_large = model_large(dummy_input)
    print(f"Output shape (large model): {output_large.shape}")


