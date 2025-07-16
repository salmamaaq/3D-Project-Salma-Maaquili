
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pointnext_model import PointNextEncoder
from byol_framework import BYOL

# Dummy Dataset for testing the training loop
class DummyPointDataset(Dataset):
    def __init__(self, num_samples=100, num_points=1024, input_dim=3):
        self.num_samples = num_samples
        self.num_points = num_points
        self.input_dim = input_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate two augmented views of a point cloud
        # In a real scenario, these would come from actual data augmentation
        point_cloud_1 = torch.randn(self.num_points, self.input_dim)
        point_cloud_2 = torch.randn(self.num_points, self.input_dim)
        return point_cloud_1, point_cloud_2

def train_byol(model, dataloader, optimizer, scheduler, num_epochs, tau_base=0.996, tau_end=1.0):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x1, x2) in enumerate(dataloader):
            optimizer.zero_grad()
            
            loss = model(x1, x2)
            loss.backward()
            optimizer.step()

            # Update target network momentum
            # Linear decay of tau from tau_base to tau_end over epochs
            tau = tau_base + (tau_end - tau_base) * (epoch * len(dataloader) + batch_idx) / (num_epochs * len(dataloader))
            model.update_target_network(tau)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        if scheduler: # For learning rate scheduling
            scheduler.step()

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 4
    num_epochs = 5
    learning_rate = 1e-3
    projection_hidden_size = 256
    projection_out_size = 128

    # Initialize PointNext Encoder
    encoder = PointNextEncoder(in_channels=3)

    # Initialize BYOL model
    byol_model = BYOL(encoder, projection_hidden_size, projection_out_size)

    # Dummy Dataset and DataLoader
    dummy_dataset = DummyPointDataset(num_samples=100, num_points=1024, input_dim=3)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and Scheduler
    optimizer = optim.Adam(byol_model.online_encoder.parameters(), lr=learning_rate)
    # A simple scheduler, e.g., StepLR or CosineAnnealingLR could be used here
    scheduler = None # For now, no scheduler

    print("Starting BYOL pre-training...")
    train_byol(byol_model, dummy_dataloader, optimizer, scheduler, num_epochs)
    print("BYOL pre-training finished.")

    # Save the pre-trained encoder
    torch.save(byol_model.online_encoder.state_dict(), "pretrained_pointnext_encoder.pth")
    print("Pre-trained encoder saved to pretrained_pointnext_encoder.pth")


