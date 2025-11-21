import torch
from typing import List, Dict, Optional
from collections import deque
import threading
from .graph import SceneGraph, ObjectNode, Edge

class SharedGlobalWorkspace:
    """
    The central 'Blackboard' or 'Workspace' that holds the single source of truth.
    All experts read from here and write updates to here.
    """
    def __init__(self, max_history: int = 50):
        self.history: deque[SceneGraph] = deque(maxlen=max_history)
        self.current_time: float = 0.0
        self.lock = threading.Lock()
        
        # Initialize with an empty state
        self.history.append(SceneGraph(timestamp=0.0))

    @property
    def current_graph(self) -> SceneGraph:
        return self.history[-1]

    def read_state(self, time_window: int = 1) -> List[SceneGraph]:
        """
        Returns the last N states.
        """
        with self.lock:
            return list(self.history)[-time_window:]

    def write_update(self, 
                     timestamp: float, 
                     new_objects: List[ObjectNode], 
                     updated_edges: List[Edge],
                     camera_pose: Optional[torch.Tensor] = None):
        """
        Transactional update to the workspace.
        Creates a NEW graph state for the new timestamp based on the previous one.
        
        NOTE: Currently assumes 'new_objects' is the COMPLETE list of active objects
        from the Tracker. Objects not in 'new_objects' are effectively dropped.
        """
        with self.lock:
            prev_graph = self.current_graph
            
            # Create new graph state
            new_graph = SceneGraph(timestamp=timestamp)
            
            # 1. Copy over existing objects (persistence)
            # In a real system, we'd apply a Kalman Filter prediction here if no update provided
            for obj_id, obj in prev_graph.objects.items():
                if obj.is_active:
                    # Deep copy or reference? For now, reference, but ideally copy state
                    # We assume 'new_objects' contains the UPDATED versions of these
                    pass

            # 2. Apply updates / Add new objects
            for obj in new_objects:
                new_graph.add_object(obj)
                
            # 3. Copy/Update Camera
            if camera_pose is not None:
                new_graph.camera.pose = camera_pose
            else:
                new_graph.camera = prev_graph.camera # Keep previous pose
                
            # 4. Update Edges
            new_graph.edges = updated_edges # Full replace for now, or merge
            
            # Commit
            self.history.append(new_graph)
            self.current_time = timestamp
            
    def prune_memory(self):
        """
        Moves very old states to long-term storage (not implemented) 
        or just lets deque handle it.
        """
        pass

    def get_object_trajectory(self, obj_id: int, length: int = 10) -> torch.Tensor:
        """
        Helper to get the past positions of a specific object.
        Returns tensor of shape [length, 3]
        """
        positions = []
        # Iterate backwards
        for i in range(len(self.history) - 1, -1, -1):
            graph = self.history[i]
            if obj_id in graph.objects:
                positions.append(graph.objects[obj_id].position)
            else:
                # Object didn't exist or was lost
                break
            if len(positions) >= length:
                break
        
        if not positions:
            return torch.empty(0, 3)
            
        # Reverse to get chronological order
        return torch.stack(positions[::-1])
