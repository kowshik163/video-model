"""
End-to-End Training Script for SGW Video Analysis System

This script implements the full training pipeline:
1. Frozen Physics Expert (pretrained on sim)
2. Trainable Visual Expert (learns to map pixels -> physics params)
3. Trainable Supervisor (learns when to route to 3D)
4. Multi-task loss optimization
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.experts.visual import VisualExpert
from src.experts.physics import PhysicsExpert
from src.experts.memory import ObjectMemory
from src.experts.bus import AttentionBus
from src.supervisor.router import Supervisor
from src.sgw.workspace import SharedGlobalWorkspace
from src.utils.losses import MultiTaskLoss
from src.data.synthetic_dataset import SyntheticVideoDataset


def train_epoch(visual, physics, supervisor, sgw, memory, bus, dataloader, loss_fn, optimizer, device):
    """Train for one epoch."""
    visual.train()
    supervisor.train()
    # Physics is frozen
    
    epoch_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        frames, targets = batch
        frames = frames.to(device)
        # Move targets to device
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
        
        optimizer.zero_grad()
        
        # Process batch
        outputs = {}
        # frames shape: [B, T, C, H, W]
        batch_size = frames.shape[0]
        
        for i in range(batch_size):
            # Clear bus for new sample
            bus.clear()

            # Take first frame from sequence
            frame = frames[i, 0].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            frame = (frame * 255).astype(np.uint8)
            
            # Visual perception
            nodes, cam_pose, flow = visual.process_frame(0.0, frame)
            
            if not nodes:
                continue

            # Publish Visual KV
            embeddings = torch.stack([n.embedding for n in nodes])
            k_vis, v_vis = visual.produce_kv(embeddings)
            bus.publish('visual', k_vis, v_vis)
                
            # Track
            tracked = memory.update(nodes, 0.0)
            
            # Physics prediction
            current_graph = sgw.current_graph
            for node in tracked:
                current_graph.add_object(node)
            
            physics_pred = physics.predict_next_state(current_graph, dt=1/30.0)

            # Publish Physics KV
            if tracked:
                tracked.sort(key=lambda n: n.id)
                node_states = torch.stack([n.to_tensor() for n in tracked]).to(device)
                k_phys, v_phys = physics.produce_kv(node_states)
                bus.publish('physics', k_phys, v_phys)
            
            # Supervisor routing
            consistency = 0.1  # Simplified
            
            # Attend to bus
            query = torch.zeros(1, 128, device=device)
            bus_context = bus.read_and_attend(query)
            
            needs_3d, routing_probs = supervisor.decide_routing(consistency, bus_context=bus_context)
            
            # Collect outputs for loss
            outputs['physics_pred'] = physics_pred
            outputs['routing_probs'] = routing_probs
            
        # Compute losses
        losses = loss_fn(outputs, targets)
        total_loss = losses['total']
        
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        
        epoch_loss += total_loss.item()
        batch_count += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss {total_loss.item():.4f}")
    
    return epoch_loss / max(batch_count, 1)


def main():
    print("=== End-to-End Training ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize components
    visual = VisualExpert(device=device)
    physics = PhysicsExpert()
    physics = physics.to(device)  # Move to device
    supervisor = Supervisor()
    supervisor.to(device)
    sgw = SharedGlobalWorkspace()
    memory = ObjectMemory()
    bus = AttentionBus(d_model=128, device=device)
    
    # Load pretrained physics and FREEZE
    try:
        physics.load_weights('physics_pretrained.pth')
        physics.freeze()
        print("Loaded and froze Physics Expert.")
    except Exception as e:
        print(f"Warning: Could not load physics weights ({e})")
    
    # Setup training
    loss_fn = MultiTaskLoss()
    
    # Only train visual and supervisor
    trainable_params = list(visual.parameters()) + list(supervisor.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    
    # Create dataset (using synthetic for now)
    print("\nCreating synthetic training dataset...")
    dataset = SyntheticVideoDataset(num_samples=1000, frames_per_sample=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Training loop
    epochs = 5
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        avg_loss = train_epoch(visual, physics, supervisor, sgw, memory, bus,
                               dataloader, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
    
    # Save trained models
    torch.save(visual.state_dict(), 'visual_trained.pth')
    torch.save(supervisor.state_dict(), 'supervisor_trained.pth')
    print("\n=== Training Complete ===")
    print("Saved visual_trained.pth and supervisor_trained.pth")
    print("\nNext: Run evaluation on Physion benchmark")


if __name__ == '__main__':
    main()
