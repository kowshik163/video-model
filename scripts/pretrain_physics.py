"""
Run physics pretraining on synthetic data.

Usage:
    python scripts/pretrain_physics.py

This will:
1. Generate 10,000 synthetic physics samples
2. Train the PhysicsExpert GNN for 10 epochs
3. Save weights to physics_pretrained.pth
"""

import sys
import os
import torch

# Add repo root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experts.physics import PhysicsExpert
from src.experts.train_physics import train_physics


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    physics = PhysicsExpert(input_dim=9, hidden_dim=128, output_dim=6)
    print("Created PhysicsExpert model")
    print(f"Parameters: {sum(p.numel() for p in physics.parameters()):,}")
    
    # Train
    trained_model = train_physics(
        physics=physics,
        num_samples=10000,
        epochs=10,
        lr=1e-3,
        device=device,
        save_path='physics_pretrained.pth'
    )
    
    print("\n=== Pretraining Complete ===")
    print("Next steps:")
    print("1. Load weights: physics.load_weights('physics_pretrained.pth')")
    print("2. Freeze model: physics.freeze()")
    print("3. Use in main pipeline for sim-to-real transfer")


if __name__ == '__main__':
    main()
