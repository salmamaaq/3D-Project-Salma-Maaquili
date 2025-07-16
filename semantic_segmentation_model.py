
import torch
import torch.nn as nn
from pointnext_model import PointNextEncoder

class SemanticSegmentationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class PointNextSemanticSegmentation(nn.Module):
    def __init__(self, encoder: PointNextEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get the output feature size of the encoder
        encoder_output_size = 1024 # Based on PointNextEncoder global_mlp output

        self.segmentation_head = SemanticSegmentationHead(encoder_output_size, num_classes)

    def forward(self, x):
        # Get the global feature vector from the encoder
        global_features = self.encoder(x)

        output = self.segmentation_head(global_features)
        return output

if __name__ == '__main__':
    # Example usage
    batch_size = 2
    num_points = 1024
    input_dim = 3
    num_classes = 40 # Example number of classes for ScanNet

    # Create a dummy PointNextEncoder
    encoder = PointNextEncoder(in_channels=input_dim)

    # Instantiate the semantic segmentation model
    model = PointNextSemanticSegmentation(encoder, num_classes)

    # Create a dummy input point cloud
    dummy_input = torch.randn(batch_size, num_points, input_dim)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape for classification: (batch_size, num_classes)")

    # Verify that encoder parameters are frozen
    for name, param in model.encoder.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")



