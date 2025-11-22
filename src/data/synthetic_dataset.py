"""
Synthetic Video Dataset for Training

Generates simple synthetic video frames with moving objects for training.
Replace with real video datasets (Kinetics, Physion) for production.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class SyntheticVideoDataset(Dataset):
    """
    Generates synthetic video clips with moving objects.
    """
    def __init__(self, num_samples=1000, frames_per_sample=10, img_size=(640, 640)):
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample
        self.img_size = img_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            frames: Tensor [T, C, H, W]
            targets: Dict with ground truth data
        """
        # Generate a simple moving box
        frames = []
        h, w = self.img_size
        
        # Random starting position and velocity
        x = np.random.randint(100, w-100)
        y = np.random.randint(100, h-100)
        vx = np.random.uniform(-5, 5)
        vy = np.random.uniform(-5, 5)
        
        for t in range(self.frames_per_sample):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Update position
            x += vx
            y += vy
            
            # Bounce off walls
            if x < 50 or x > w - 50:
                vx = -vx
            if y < 50 or y > h - 50:
                vy = -vy
            
            # Draw box
            box_size = 50
            x1, y1 = int(x - box_size//2), int(y - box_size//2)
            x2, y2 = int(x + box_size//2), int(y + box_size//2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
            
            # Convert to tensor [C, H, W]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)
        
        frames = torch.stack(frames)  # [T, C, H, W]
        
        # Create dummy targets (avoid None for collation)
        targets = {
            'bbox_gt': torch.tensor([x1, y1, x2, y2]).float(),
            'visual_state': torch.tensor([x, y, 0, vx, vy, 0]).float(),
            'rewards': torch.tensor([1.0])
        }
        
        return frames, targets
