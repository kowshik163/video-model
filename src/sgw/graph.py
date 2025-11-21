import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class ObjectNode:
    """
    Represents a tracked object in the scene.
    """
    id: int
    class_id: int
    confidence: float
    
    # Spatiotemporal State
    position: torch.Tensor  # [x, y, z] (or [x, y] if 2D initially)
    velocity: torch.Tensor  # [vx, vy, vz]
    bbox: torch.Tensor      # [x1, y1, x2, y2]
    
    # Visual Features
    embedding: Optional[torch.Tensor] = None  # ReID / Visual feature vector
    mask: Optional[torch.Tensor] = None       # Segmentation mask
    
    # Physical Properties (Inferred)
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.5  # Bounciness
    
    # Metadata
    last_seen_timestamp: float = 0.0
    is_active: bool = True

    def to_tensor(self) -> torch.Tensor:
        """Concatenates state into a single tensor for GNN input."""
        # Example: [pos(3), vel(3), mass(1), friction(1)]
        return torch.cat([
            self.position, 
            self.velocity, 
            torch.tensor([self.mass, self.friction, self.restitution], device=self.position.device)
        ])

@dataclass
class CameraNode:
    """
    Represents the camera state.
    """
    id: int = 0
    pose: torch.Tensor = field(default_factory=lambda: torch.eye(4)) # 4x4 matrix
    intrinsics: torch.Tensor = field(default_factory=lambda: torch.eye(3)) # 3x3 matrix
    timestamp: float = 0.0

@dataclass
class BackgroundNode:
    """
    Represents the background / environment.
    """
    id: int = -1
    global_flow: torch.Tensor = field(default_factory=lambda: torch.zeros(2)) # Average flow vector
    static_regions_mask: Optional[torch.Tensor] = None

@dataclass
class Edge:
    """
    Represents a relationship between two nodes.
    """
    source_id: int
    target_id: int
    edge_type: str  # 'spatial', 'semantic', 'contact'
    features: torch.Tensor = field(default_factory=lambda: torch.zeros(1)) # e.g., distance, force magnitude

class SceneGraph:
    """
    Container for the entire graph state at a specific timestamp.
    """
    def __init__(self, timestamp: float):
        self.timestamp = timestamp
        self.objects: Dict[int, ObjectNode] = {}
        self.camera: CameraNode = CameraNode()
        self.background: BackgroundNode = BackgroundNode()
        self.edges: List[Edge] = []

    def add_object(self, obj: ObjectNode):
        self.objects[obj.id] = obj

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def to_pyg_data(self):
        """
        Converts the scene graph to a PyTorch Geometric Data object.
        Useful for passing to the Physics Expert (GNN).
        """
        # Try importing PyG Data, fallback to simple namespace if missing
        try:
            from torch_geometric.data import Data
        except ImportError:
            class Data:
                def __init__(self, x, edge_index, edge_attr):
                    self.x = x
                    self.edge_index = edge_index
                    self.edge_attr = edge_attr

        # 1. Collect Nodes & Stack Features
        # Sort by ID to ensure deterministic order in the tensor
        sorted_nodes = sorted(self.objects.values(), key=lambda n: n.id)
        
        if not sorted_nodes:
            # Return empty data structure
            return Data(
                x=torch.empty((0, 9)), # 9 is the feature dim from to_tensor()
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1))
            )

        # Stack node features: [N, 9]
        x = torch.stack([n.to_tensor() for n in sorted_nodes])
        
        # Map ID to index for edge construction
        id_to_idx = {n.id: i for i, n in enumerate(sorted_nodes)}
        
        # 2. Collect Edges
        edge_indices = []
        edge_attrs = []
        
        for edge in self.edges:
            if edge.source_id in id_to_idx and edge.target_id in id_to_idx:
                src = id_to_idx[edge.source_id]
                dst = id_to_idx[edge.target_id]
                edge_indices.append([src, dst])
                edge_attrs.append(edge.features)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            # Ensure edge_attr is at least 2D [E, F]
            if len(edge_attrs) > 0:
                edge_attr = torch.stack(edge_attrs)
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(1)
            else:
                edge_attr = torch.empty((0, 1))
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1))

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
