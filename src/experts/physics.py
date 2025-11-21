import torch
import torch.nn as nn
from typing import Tuple
from ..sgw.graph import SceneGraph

class PhysicsExpert(nn.Module):
    """
    The 'Simulator'. 
    Predicts future states based on physical laws learned from simulation.
    FROZEN during real-world inference.
    """
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=6):
        super().__init__()
        # Input: [pos(3), vel(3), mass(1), friction(1), restitution(1)] = 9
        # Output: [delta_pos(3), delta_vel(3)] = 6
        
        # Simple Interaction Network / GNN
        # 1. Node Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Edge Encoder (Interaction)
        # Input: [node_i, node_j, dist]
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Node Updater
        self.node_updater = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim), # self + aggregated_edges
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.frozen = False
        self.use_newtonian_residual = True # Use GNN as residual to Newtonian physics

    def load_weights(self, path: str):
        """Loads pretrained weights (e.g. from Kubric Sim)."""
        try:
            self.load_state_dict(torch.load(path))
            print(f"PhysicsExpert: Loaded weights from {path}")
            self.use_newtonian_residual = False # Assume trained model handles everything
        except FileNotFoundError:
            print(f"PhysicsExpert: Weights not found at {path}. Using Newtonian Fallback.")
            self.use_newtonian_residual = True

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Predicts next state deltas.
        """
        num_nodes = node_states.shape[0]
        
        # 1. Encode Nodes
        h_v = self.node_encoder(node_states)
        
        # 2. Compute Edge Messages
        if edge_index.size(1) > 0:
            row, col = edge_index
            # Concatenate source node, target node, and edge attributes
            edge_inputs = torch.cat([h_v[row], h_v[col], edge_attr], dim=1)
            h_e = self.edge_encoder(edge_inputs)
            
            # 3. Aggregate Messages (Sum)
            agg_messages = torch.zeros(num_nodes, h_v.shape[1], device=node_states.device)
            agg_messages.index_add_(0, col, h_e)
        else:
            agg_messages = torch.zeros(num_nodes, h_v.shape[1], device=node_states.device)
        
        # 4. Update Nodes
        node_inputs = torch.cat([h_v, agg_messages], dim=1)
        gnn_deltas = self.node_updater(node_inputs)
        
        return gnn_deltas

    def predict_next_state(self, graph: SceneGraph, dt: float) -> torch.Tensor:
        """
        Wrapper to convert SceneGraph to tensors and run forward pass.
        """
        # Extract node states
        # ...existing code...
        nodes = list(graph.objects.values())
        if not nodes:
            return torch.empty(0)
            
        # Ensure deterministic order (by ID) matching to_pyg_data
        nodes.sort(key=lambda n: n.id)
        
        node_tensors = torch.stack([n.to_tensor() for n in nodes])
        
        # Build fully connected edge index (simplified)
        num_nodes = len(nodes)
        if num_nodes > 1:
            # Create all pairs
            adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            edge_index = adj.nonzero().t()
            
            # Calculate distances for edge attributes
            pos = node_tensors[:, :3]
            row, col = edge_index
            dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
            edge_attr = dist
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1))

        # Run model
        gnn_deltas = self(node_tensors, edge_index, edge_attr)
        
        if self.use_newtonian_residual:
            # Calculate Newtonian Baseline (Gravity + Inertia)
            # node_tensors: [pos(3), vel(3), mass(1), fric(1), rest(1)]
            vel = node_tensors[:, 3:6]
            mass = node_tensors[:, 6:7]
            
            # Gravity (assuming Y is down in image coords, or Z is up in 3D)
            # Let's assume Y is down for 2D video: [0, 9.8, 0]
            gravity = torch.tensor([0.0, 9.8, 0.0], device=node_tensors.device).unsqueeze(0) # [1, 3]
            
            # F = ma -> a = F/m (Gravity is constant acceleration though)
            accel = gravity.expand(node_tensors.shape[0], -1) # [N, 3]
            
            # Euler Integration:
            # delta_vel = a * dt
            # delta_pos = v * dt
            newton_d_vel = accel * dt
            newton_d_pos = vel * dt
            
            newton_deltas = torch.cat([newton_d_pos, newton_d_vel], dim=1)
            
            # Combine: Baseline + Residual (scaled down if untrained)
            # If untrained, GNN outputs ~0.0 or random noise. 
            # We scale it down to let physics dominate until trained.
            return newton_deltas + 0.01 * gnn_deltas
            
        return gnn_deltas

    def compute_consistency_score(self, visual_nodes: list, predicted_deltas: torch.Tensor) -> float:
        """
        Computes the divergence between what Physics predicted and what Vision saw.
        High score = High Conflict (Anomaly).
        """
        if not visual_nodes or predicted_deltas.numel() == 0:
            return 0.0
            
        # Extract visual deltas (this requires tracking history, simplified here)
        # We assume visual_nodes contains the CURRENT observed state
        # We compare predicted velocity vs observed velocity
        
        pred_vel = predicted_deltas[:, 3:6]
        obs_vel = torch.stack([n.velocity for n in visual_nodes])
        
        # MSE
        error = torch.mean((pred_vel - obs_vel) ** 2)
        return error.item()
