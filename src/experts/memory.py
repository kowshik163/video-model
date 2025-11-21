import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict
from ..sgw.graph import ObjectNode

class ObjectMemory:
    """
    The 'Tracker'.
    Maintains object identity over time using Slots.
    """
    def __init__(self, max_age=30):
        self.tracks: Dict[int, ObjectNode] = {}
        self.next_id = 0
        self.max_age = max_age # Frames to keep a lost track

    def compute_cost_matrix(self, tracks: List[ObjectNode], detections: List[ObjectNode]) -> np.ndarray:
        """
        Computes cost based on IoU (spatial) and Cosine Distance (visual).
        """
        if not tracks or not detections:
            return np.array([])
            
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                # 1. Spatial Cost (1 - IoU) - Simplified as center distance for now
                dist = torch.norm(t.position[:2] - d.position[:2]).item()
                spatial_cost = dist / 100.0 # Normalize roughly
                
                # 2. Visual Cost (Cosine Distance)
                if t.embedding is not None and d.embedding is not None:
                    vis_cost = 1.0 - torch.nn.functional.cosine_similarity(t.embedding.unsqueeze(0), d.embedding.unsqueeze(0)).item()
                else:
                    vis_cost = 0.5
                    
                cost_matrix[i, j] = 0.4 * spatial_cost + 0.6 * vis_cost
                
        return cost_matrix

    def update(self, detections: List[ObjectNode], timestamp: float) -> List[ObjectNode]:
        """
        Updates tracks with new detections.
        Returns the list of active tracks (both matched and unmatched-but-alive).
        """
        active_track_ids = [tid for tid, t in self.tracks.items() if t.is_active]
        active_tracks = [self.tracks[tid] for tid in active_track_ids]
        
        # Match
        cost_matrix = self.compute_cost_matrix(active_tracks, detections)
        
        matched_indices = []
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.5: # Threshold
                    matched_indices.append((r, c))
                    
                    # Update Track
                    track = active_tracks[r]
                    det = detections[c]
                    
                    # Update state
                    track.position = det.position
                    track.bbox = det.bbox
                    track.embedding = det.embedding # Update appearance
                    track.last_seen_timestamp = timestamp
                    
                    # Update physics props (average or Kalman update)
                    track.mass = 0.9 * track.mass + 0.1 * det.mass
                    
                    # Calculate velocity (simple finite difference)
                    dt = timestamp - track.last_seen_timestamp
                    if dt > 0:
                        track.velocity = (det.position - track.position) / dt
                    
        # Handle Unmatched Detections (New Objects)
        matched_det_indices = {c for _, c in matched_indices}
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                # Create new track
                det.id = self.next_id
                self.tracks[self.next_id] = det
                self.next_id += 1
                
        # Handle Unmatched Tracks (Lost Objects)
        matched_track_indices = {r for r, _ in matched_indices}
        for i, track in enumerate(active_tracks):
            if i not in matched_track_indices:
                # Mark as lost or increment age
                # In a real system, we'd use Physics Expert to predict position here
                pass
                
        # Prune dead tracks
        # (Simplified: just return all currently active ones)
        return list(self.tracks.values())
