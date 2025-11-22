import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

class SyntheticPhysicsDataset(Dataset):
    """
    Generates synthetic physics simulation data for pretraining the Physics Expert.
    
    Simulates simple Newtonian dynamics with:
    - Multiple objects with random masses, positions, velocities
    - Gravity and simple collision detection
    - Ground truth next-state deltas
    
    This replaces the need for Kubric/ThreeDWorld for initial prototyping.
    """
    
    def __init__(self, num_samples=10000, num_objects_range=(2, 8), dt=0.033, device='cpu'):
        self.num_samples = num_samples
        self.num_objects_range = num_objects_range
        self.dt = dt
        self.device = device
        
        # Physics constants
        self.gravity = torch.tensor([0.0, 9.8, 0.0], device=device)
        self.bounce_damping = 0.8
        self.ground_y = 100.0  # Ground plane at y=100
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_states: [N, 9] - [pos(3), vel(3), mass(1), friction(1), restitution(1)]
            edge_index: [2, E] - fully connected graph
            edge_attr: [E, 1] - pairwise distances
            target_deltas: [N, 6] - [delta_pos(3), delta_vel(3)]
        """
        # Random number of objects
        num_objs = np.random.randint(*self.num_objects_range)
        
        # Generate initial states
        positions = torch.rand(num_objs, 3, device=self.device) * 50  # Random positions 0-50
        positions[:, 1] = torch.rand(num_objs, device=self.device) * 30 + 20  # y: 20-50 (above ground)
        
        velocities = (torch.rand(num_objs, 3, device=self.device) - 0.5) * 10  # -5 to 5
        
        masses = torch.rand(num_objs, 1, device=self.device) * 9 + 1  # 1-10 kg
        frictions = torch.rand(num_objs, 1, device=self.device) * 0.5 + 0.3  # 0.3-0.8
        restitutions = torch.rand(num_objs, 1, device=self.device) * 0.5 + 0.3  # 0.3-0.8
        
        # Concatenate state
        node_states = torch.cat([positions, velocities, masses, frictions, restitutions], dim=1)
        
        # Build fully connected edges
        edge_index = self._build_edges(num_objs)
        edge_attr = self._compute_edge_features(positions, edge_index)
        
        # Simulate forward step (ground truth)
        target_deltas = self._simulate_step(node_states)
        
        return node_states, edge_index, edge_attr, target_deltas
    
    def _build_edges(self, num_nodes: int) -> torch.Tensor:
        """Builds fully connected edge index."""
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        adj = torch.ones(num_nodes, num_nodes, device=self.device) - torch.eye(num_nodes, device=self.device)
        edge_index = adj.nonzero().t().contiguous()
        return edge_index
    
    def _compute_edge_features(self, positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Computes pairwise distances."""
        if edge_index.size(1) == 0:
            return torch.empty((0, 1), device=self.device)
        
        row, col = edge_index
        dist = torch.norm(positions[row] - positions[col], dim=1, keepdim=True)
        return dist
    
    def _simulate_step(self, node_states: torch.Tensor) -> torch.Tensor:
        """
        Simulates one physics step and returns deltas.
        
        Physics:
        1. Apply gravity
        2. Check ground collision
        3. Simple inter-object repulsion (to avoid overlap)
        4. Euler integration
        """
        pos = node_states[:, 0:3]
        vel = node_states[:, 3:6]
        mass = node_states[:, 6:7]
        friction = node_states[:, 7:8]
        restitution = node_states[:, 8:9]
        
        # 1. Gravity force
        accel = self.gravity.unsqueeze(0).expand(pos.size(0), -1)
        
        # 2. Ground collision
        below_ground = pos[:, 1] < self.ground_y
        if below_ground.any():
            # Bounce velocity
            vel[below_ground, 1] = -vel[below_ground, 1].abs() * restitution[below_ground].squeeze()
            # Clamp position
            pos[below_ground, 1] = self.ground_y
        
        # 3. Simple repulsion (if objects too close)
        for i in range(pos.size(0)):
            for j in range(i+1, pos.size(0)):
                diff = pos[j] - pos[i]
                dist = torch.norm(diff)
                if dist < 2.0:  # Collision threshold
                    # Push apart
                    force_dir = diff / (dist + 1e-6)
                    repulsion = force_dir * 0.5
                    vel[i] -= repulsion
                    vel[j] += repulsion
        
        # 4. Euler integration
        new_vel = vel + accel * self.dt
        new_pos = pos + vel * self.dt
        
        # Compute deltas
        delta_pos = new_pos - pos
        delta_vel = new_vel - vel
        
        return torch.cat([delta_pos, delta_vel], dim=1)


def create_synthetic_dataloader(batch_size=32, num_samples=10000, num_workers=0, device='cpu'):
    """Creates a DataLoader for synthetic physics data."""
    dataset = SyntheticPhysicsDataset(num_samples=num_samples, device=device)
    
    # Custom collate function to handle variable-sized graphs
    def collate_fn(batch):
        # Each item is (node_states, edge_index, edge_attr, target_deltas)
        # For simplicity, process one at a time (batch_size=1 per graph)
        return batch[0]
    
    loader = DataLoader(
        dataset, 
        batch_size=1,  # One graph at a time
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader
