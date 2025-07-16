
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pointnext_model import PointNextEncoder
from semantic_segmentation_model import PointNextSemanticSegmentation
from semantic_segmentation_data import load_scannet_ply_with_labels, load_scannet_labels_from_json
from evaluation_metrics import calculate_classification_metrics
import numpy as np
import os
import open3d as o3d

# Custom Dataset for ScanNet semantic segmentation
class ScanNetSegmentationDataset(Dataset):
    def __init__(self, mesh_file, label_ply_file, agg_json_file, segs_json_file, num_points=1024):
        self.num_points = num_points
        self.mesh_file = mesh_file
        self.label_ply_file = label_ply_file
        self.agg_json_file = agg_json_file
        self.segs_json_file = segs_json_file

        # Load the point cloud and labels
        self.point_cloud_o3d = o3d.io.read_point_cloud(self.mesh_file)
        if not self.point_cloud_o3d.has_points():
            raise ValueError(f"No points found in mesh file: {self.mesh_file}")
        self.points = np.asarray(self.point_cloud_o3d.points)
        self.colors = np.asarray(self.point_cloud_o3d.colors)

        # Load labels from JSON (more reliable for semantic segmentation)
        self.labels, self.label_map = load_scannet_labels_from_json(
            self.agg_json_file, self.segs_json_file, len(self.points)
        )
        if self.labels is None:
            raise ValueError("Failed to load labels from JSON files.")

        # Filter out points with -1 label (unassigned)
        valid_indices = self.labels != -1
        self.points = self.points[valid_indices]
        self.colors = self.colors[valid_indices]
        self.labels = self.labels[valid_indices]

        if len(self.points) == 0:
            raise ValueError("No valid points after filtering unassigned labels.")

    def __len__(self):
        # For a single scene, we can simulate multiple samples by returning a larger length
        # and sampling points within __getitem__.
        # This is a hack to allow a batch size > 1 for BatchNorm.
        return 100 # Simulate 100 samples from this single scene

    def __getitem__(self, idx):
        # Randomly sample num_points from the scene
        if len(self.points) > self.num_points:
            indices = np.random.choice(len(self.points), self.num_points, replace=False)
        else:
            # If not enough points, sample with replacement
            indices = np.random.choice(len(self.points), self.num_points, replace=True)

        sampled_points = self.points[indices]
        sampled_colors = self.colors[indices]
        sampled_labels = self.labels[indices]

        point_cloud_data = torch.tensor(sampled_points, dtype=torch.float32)
        label_data = torch.tensor(sampled_labels, dtype=torch.long)

        return point_cloud_data, label_data

def train_segmentation_model(model, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        all_predictions = []
        all_true_labels = []

        for batch_idx, (point_clouds, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(point_clouds)
            
            # For classification, we expect (B, num_classes) output
            # And labels are (B, N) or (B,) if per-point or per-cloud
            # Since our model outputs (B, num_classes), we need a single label per cloud.
            # For now, we will use the most frequent label in the sampled points as the scene label.
            # This is still a simplification for true semantic segmentation.
            
            # Determine the most frequent label in the batch for scene-level classification
            # This is a temporary solution for the current model output (scene-level classification)
            # For proper semantic segmentation, the model output and loss calculation would be per-point.
            mode_labels = torch.mode(labels, dim=1).values # Get the most frequent label for each sample in the batch
            target_labels = mode_labels

            loss = criterion(outputs, target_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Collect predictions and true labels for metrics
            all_predictions.append(torch.argmax(outputs, dim=1).cpu())
            all_true_labels.append(target_labels.cpu())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Calculate and print metrics
        all_predictions = torch.cat(all_predictions)
        all_true_labels = torch.cat(all_true_labels)
        metrics = calculate_classification_metrics(all_predictions, all_true_labels)
        print(f"Metrics: {metrics}")


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 4 # Increased batch size for BatchNorm compatibility
    num_epochs = 10
    learning_rate = 1e-3
    num_classes = 40 # Based on ScanNet dataset

    # Paths to the downloaded files for scene0000_00
    mesh_file_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean_2.ply"
    label_ply_file_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply"
    aggregation_json_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean.aggregation.json"
    segs_json_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean_2.0.010000.segs.json"

    # Initialize PointNext Encoder (load pre-trained weights if available)
    encoder = PointNextEncoder(in_channels=3)
    pretrained_encoder_path = "pretrained_pointnext_encoder.pth"
    if os.path.exists(pretrained_encoder_path):
        print(f"Loading pre-trained encoder from {pretrained_encoder_path}")
        # Load state_dict with strict=False to ignore missing running_mean/var if they were not saved
        encoder.load_state_dict(torch.load(pretrained_encoder_path), strict=False)
    else:
        print("Pre-trained encoder not found. Training from scratch.")

    # Initialize Semantic Segmentation Model
    segmentation_model = PointNextSemanticSegmentation(encoder, num_classes)

    # Dataset and DataLoader
    try:
        scannet_dataset = ScanNetSegmentationDataset(
            mesh_file_path, label_ply_file_path, aggregation_json_path, segs_json_path
        )
        scannet_dataloader = DataLoader(scannet_dataset, batch_size=batch_size, shuffle=True)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        print("Skipping semantic segmentation training.")
        exit()

    # Optimizer and Loss Function
    optimizer = optim.Adam(segmentation_model.segmentation_head.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Starting semantic segmentation training...")
    train_segmentation_model(segmentation_model, scannet_dataloader, optimizer, criterion, num_epochs)
    print("Semantic segmentation training finished.")



