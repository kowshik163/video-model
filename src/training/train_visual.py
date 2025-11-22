import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import sys
import os

# Allow imports from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experts.visual import VisualExpert
from src.utils.losses import MultiTaskLoss

class MockVisualPhysicsDataset(Dataset):
    """
    Mock dataset for System Identification (Visual Adaptation).
    Generates synthetic images of objects where visual properties correlate with physical properties.
    
    Example:
    - Size -> Mass (Larger = Heavier)
    - Color (Red channel) -> Friction (Redder = Higher Friction)
    - Color (Blue channel) -> Restitution (Bluer = Bouncier)
    """
    def __init__(self, num_samples=1000, img_size=(224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 1. Generate Physical Params
        mass = np.random.uniform(0.1, 1.0) # Normalized mass
        friction = np.random.uniform(0.0, 1.0)
        restitution = np.random.uniform(0.0, 1.0)
        
        # 2. Generate Image based on params
        img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        # Size correlates with mass
        radius = int(mass * 50) + 10
        
        # Color correlates with friction/restitution
        # R = Friction * 255
        # B = Restitution * 255
        # G = Random
        color = (int(restitution * 255), np.random.randint(0, 100), int(friction * 255)) # BGR for OpenCV
        
        # Draw circle
        center = (np.random.randint(radius, self.img_size[1]-radius), 
                  np.random.randint(radius, self.img_size[0]-radius))
        cv2.circle(img, center, radius, color, -1)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Target tensor: [mass, friction, restitution]
        target = torch.tensor([mass, friction, restitution], dtype=torch.float32)
        
        return img_tensor, target

def train_visual_sys_id(
    visual_expert: VisualExpert,
    num_samples=1000,
    batch_size=32,
    epochs=5,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_path='visual_sys_id.pth'
):
    print(f"Starting Visual Expert System ID Training on {device}...")
    
    dataset = MockVisualPhysicsDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    visual_expert.to(device)
    visual_expert.train()
    
    # We only want to train the system_id_head and maybe fine-tune the encoder
    optimizer = optim.Adam(visual_expert.parameters(), lr=lr)
    loss_fn = MultiTaskLoss().l_sys_id
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            # We need to extract features first. 
            # VisualExpert.process_frame does too much (detection etc).
            # We'll access the encoder directly or add a helper method.
            # Assuming visual_expert.encoder exists and returns flattened features.
            
            features = visual_expert.encoder(imgs)
            preds = visual_expert.system_id_head(features)
            
            loss = loss_fn(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1} Batch {batch_idx}: Loss {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} Complete. Avg Loss: {avg_loss:.4f}")
        
    torch.save(visual_expert.state_dict(), save_path)
    print(f"Saved Visual Expert weights to {save_path}")

if __name__ == "__main__":
    ve = VisualExpert(device='cpu', use_raft=False) # Use CPU for test
    train_visual_sys_id(ve, num_samples=100, epochs=2, device='cpu')
