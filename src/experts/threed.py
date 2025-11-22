import torch
import torch.nn as nn
import torchvision.models.video as video_models
from typing import List, Dict, Optional

class ThreeDExpert(nn.Module):
    """
    The 'On-Demand' 3D Expert.
    Uses a 3D CNN (ResNet3D-18) to analyze spatiotemporal features.
    Triggered only when the Supervisor deems the situation ambiguous or complex.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        print("ThreeDExpert: Loading ResNet3D-18...")
        try:
            # Load pretrained R3D-18
            # We use the default weights (Kinetics-400)
            self.model = video_models.r3d_18(pretrained=True)
            
            # Remove the classification head to get features
            # The original fc is Linear(in_features=512, out_features=400)
            # We replace it with Identity or just return the features before it.
            # But for simplicity, let's just use the model as is and take the penultimate layer output if possible,
            # or just use the class logits as a high-level descriptor.
            # Better: Replace fc with Identity to get 512-dim embedding.
            self.model.fc = nn.Identity()
            
            self.model.to(device)
            self.model.eval()
            print("ThreeDExpert: Loaded ResNet3D-18 successfully.")
        except Exception as e:
            print(f"ThreeDExpert: Failed to load R3D-18 ({e}).")
            self.model = None

    def forward(self, video_clip: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Args:
            video_clip: Tensor of shape (B, C, T, H, W). 
                        Values should be normalized [0, 1] or standardized.
                        T should be around 16 frames for R3D-18.
        Returns:
            features: Tensor of shape (B, 512) representing spatiotemporal features.
        """
        if self.model is None:
            return None
            
        with torch.no_grad():
            # Ensure input is on device
            video_clip = video_clip.to(self.device)
            
            # R3D expects (B, C, T, H, W)
            features = self.model(video_clip)
            
        return features

    def analyze_roi(self, frames: List[torch.Tensor], roi_box: List[int]) -> Dict:
        """
        Analyze a specific Region of Interest over time.
        
        Args:
            frames: List of T tensors (C, H, W), normalized.
            roi_box: [x1, y1, x2, y2] bounding box.
            
        Returns:
            Dict containing 3D features and analysis.
        """
        if self.model is None or len(frames) < 8:
            return {"error": "Insufficient frames or model not loaded"}

        # 1. Crop and Resize
        # We need a fixed size for the 3D CNN, typically 112x112 or similar.
        target_size = (112, 112)
        clip_tensor = []
        
        x1, y1, x2, y2 = map(int, roi_box)
        
        for frame in frames:
            # frame is (C, H, W)
            # Crop
            crop = frame[:, y1:y2, x1:x2]
            # Resize (using interpolate which expects 4D input)
            crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
            clip_tensor.append(crop) # (1, C, H, W)
            
        # Stack along temporal dimension
        # (B, C, T, H, W) -> (1, 3, T, 112, 112)
        input_tensor = torch.stack(clip_tensor, dim=2).to(self.device)
        
        # 2. Inference
        features = self.forward(input_tensor) # (1, 512)
        
        return {
            "3d_embedding": features.cpu().numpy().tolist(),
            "status": "success"
        }
