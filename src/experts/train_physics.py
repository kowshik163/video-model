import torch
import sys
import os

"""
Physics pretraining utilities.

Trains the PhysicsExpert on synthetic physics simulation data.
Use this to pretrain the GNN before fine-tuning on real-world video data.
"""

# Allow imports from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experts.physics import PhysicsExpert
from src.data.synthetic_physics import create_synthetic_dataloader


def train_physics(
    physics: PhysicsExpert, 
    num_samples=10000,
    batch_size=32, 
    epochs=10, 
    lr=1e-3, 
    device='cpu',
    save_path='physics_pretrained.pth'
):
    """
    Train the Physics Expert on synthetic data.
    
    Args:
        physics: PhysicsExpert model
        num_samples: Number of synthetic samples to generate
        batch_size: Batch size (note: currently 1 graph per batch)
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
        save_path: Path to save trained weights
    """
    print(f"Starting Physics Expert pretraining on {device}...")
    print(f"Generating {num_samples} synthetic samples...")
    
    dataloader = create_synthetic_dataloader(
        batch_size=batch_size,
        num_samples=num_samples,
        device=device
    )
    
    physics.to(device)
    physics.train()
    
    # Disable Newtonian residual during training (we want pure learning)
    physics.use_newtonian_residual = False

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, physics.parameters()), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            node_states, edge_index, edge_attr, targets = batch
            node_states = node_states.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            targets = targets.to(device)

            preds = physics(node_states, edge_index, edge_attr)
            loss = loss_fn(preds, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 500 == 0:
                print(f"  Batch {num_batches}/{len(dataloader)}: Loss {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}: Avg Loss {avg_loss:.4f}")

    # Save weights
    torch.save(physics.state_dict(), save_path)
    print(f"Saved pretrained weights to {save_path}")
    print("Training complete!")
    
    return physics
