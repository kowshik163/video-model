import torch
import torch.nn as nn
from typing import List, Dict
from ..sgw.graph import ObjectNode

class GeometryExpert(nn.Module):
    """
    The '3D Specialist'.
    Runs heavy 3D reconstruction or 3D CNNs on demand.
    Only triggered by Supervisor when Consistency is low.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        print("GeometryExpert: Loading MiDaS (Small) for depth estimation...")
        try:
            # Load MiDaS small model for efficiency
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(device)
            self.model.eval()
            print("GeometryExpert: Loaded MiDaS successfully.")
        except Exception as e:
            print(f"GeometryExpert: Failed to load MiDaS ({e}). Using Identity fallback.")
            self.model = None

    def refine_roi(self, frame_crop: torch.Tensor, depth_prior: torch.Tensor) -> Dict:
        """
        Performs detailed 3D analysis on a Region of Interest (ROI).
        frame_crop: [C, H, W] tensor, normalized 0-1
        depth_prior: [1] tensor, estimated z-depth
        """
        if self.model is None:
             return {
                "refined_depth": depth_prior, # No change
                "contact_points": [],
                "3d_bbox": torch.zeros(8, 3)
            }

        try:
            # Resize to multiple of 32 (MiDaS expects specific sizes, but is flexible)
            # Simple interpolation to 256x256 for speed
            input_batch = torch.nn.functional.interpolate(
                frame_crop.unsqueeze(0).to(self.device), 
                size=(256, 256), 
                mode="bicubic", 
                align_corners=False
            )
            
            # Normalize (MiDaS specific normalization usually required, simplified here)
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            input_batch = (input_batch - mean) / std

            with torch.no_grad():
                prediction = self.model(input_batch)
            
            # MiDaS outputs inverse depth (disparity). 
            # We need to calibrate it using the depth_prior (if available and reliable)
            # or just return the mean disparity as a proxy for now.
            
            # Use the mean of the center region as the object depth
            # Prediction is [1, H, W] (after resize inside model usually, or output size)
            # MiDaS small output size matches input size usually? No, it might be different.
            
            pred_h, pred_w = prediction.shape[1], prediction.shape[2]
            center_h, center_w = pred_h // 2, pred_w // 2
            
            # Take a small window
            depth_val = prediction[0, center_h-5:center_h+5, center_w-5:center_w+5].mean()
            
            # Invert for metric depth (simplified)
            # Avoid div by zero. MiDaS output is relative inverse depth.
            # Higher value = closer. 
            refined_depth = 100.0 / (depth_val + 1e-6)
            
            return {
                "refined_depth": refined_depth.view(1),
                "contact_points": [],
                "3d_bbox": torch.zeros(8, 3)
            }
            
        except Exception as e:
            print(f"GeometryExpert: Error during refinement ({e})")
            return {
                "refined_depth": depth_prior,
                "contact_points": [],
                "3d_bbox": torch.zeros(8, 3)
            }
