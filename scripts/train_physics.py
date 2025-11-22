"""
Training harness for `PhysicsExpert`.
- Loads synthetic clips from `data/synthetic/*/frames.npz` and object params.
- Creates simple batches and trains PhysicsExpert to predict next-state deltas.
- Saves checkpoints to `checkpoints/physics`.

Usage:
    python scripts/train_physics.py --data_dir data/synthetic --epochs 10 --batch_size 8

This is a minimal trainer to get Phase 6 started. Expand dataset loading and loss functions as needed.
"""
import argparse
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from src.experts.physics import PhysicsExpert


class SyntheticPhysicsDataset(Dataset):
    def __init__(self, root_dir, clip_frames=16):
        self.clips = glob.glob(os.path.join(root_dir, "clip_*"))
        self.clip_frames = clip_frames

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_dir = self.clips[idx]
        
        # Load trajectories (Ground Truth State)
        traj_data = np.load(os.path.join(clip_dir, "trajectories.npz"))
        pos = traj_data["pos"] # (T, N, 3)
        vel = traj_data["vel"] # (T, N, 3)
        
        # Load object params
        params_data = np.load(os.path.join(clip_dir, "object_params.npz"), allow_pickle=True)
        params_list = params_data["params"]
        
        # Construct input/target pairs
        # Input: State at t (pos, vel, mass, friction)
        # Target: Delta State (pos_t+1 - pos_t, vel_t+1 - vel_t)
        
        T = min(self.clip_frames, pos.shape[0] - 1)
        
        # Select a random time step t
        t = np.random.randint(0, T)
        
        # For simplicity, assume 1 object (index 0)
        # In full version, iterate all objects
        p_t = pos[t, 0]
        v_t = vel[t, 0]
        p_next = pos[t+1, 0]
        v_next = vel[t+1, 0]
        
        mass = params_list[0]["mass"]
        friction = params_list[0]["friction"]
        restitution = params_list[0]["restitution"]
        
        # Input vector: [px, py, pz, vx, vy, vz, mass, friction, restitution]
        state_input = np.concatenate([p_t, v_t, [mass, friction, restitution]])
        
        # Target delta: [dpx, dpy, dpz, dvx, dvy, dvz]
        delta_pos = p_next - p_t
        delta_vel = v_next - v_t
        target_delta = np.concatenate([delta_pos, delta_vel])
        
        return torch.from_numpy(state_input).float(), torch.from_numpy(target_delta).float()


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = SyntheticPhysicsDataset(args.data_dir, clip_frames=args.clip_frames)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = PhysicsExpert()
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, (state_input, target_delta) in enumerate(dl):
            state_input = state_input.to(device)
            target_delta = target_delta.to(device)
            
            optimizer.zero_grad()
            
            # PhysicsExpert.forward expects a graph, but for this simple training loop
            # we might need to bypass the graph wrapper or construct a dummy graph.
            # Let's assume we added a direct `predict_from_tensor` method or similar.
            # Or we construct a minimal graph representation.
            # For now, let's assume the model has a simple MLP head we can call directly
            # if we modify PhysicsExpert to expose it.
            # Alternatively, we use the existing `predict_next_state` but that takes a SceneGraph.
            
            # HACK: Access the internal GNN/MLP directly for training
            # Assuming model.gnn or model.dynamics_model exists.
            # If PhysicsExpert is just a wrapper, let's check its code.
            # For this scaffold, let's assume model(state_input) works if we implement it.
            
            # Let's use a dummy forward for now that we will implement in PhysicsExpert
            pred_delta = model.forward_dummy(state_input) 
            
            loss = criterion(pred_delta, target_delta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {epoch_loss/len(dl):.6f}")
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"physics_epoch_{epoch+1}.pth"))

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_frames", type=int, default=16)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/physics")
    args = parser.parse_args()
    train(args)
