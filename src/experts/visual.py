import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple, Dict
from ..sgw.graph import ObjectNode

class VisualExpert(nn.Module):
    """
    The 'Eye' of the system. 
    Handles Detection, Optical Flow, and Visual Feature Extraction.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # 1. Object Detector (YOLOv5/v8)
        self.use_yolo = None
        try:
            # Try loading YOLOv5 from torch hub
            print("VisualExpert: Loading YOLOv5s from torch.hub...")
            self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.detector.to(device)
            self.detector.eval()
            self.use_yolo = "v5"
            print("VisualExpert: Loaded YOLOv5s successfully.")
        except Exception as e:
            print(f"VisualExpert: Failed to load YOLOv5 ({e}). Falling back to mock detector.")
            self.detector = None
        
        # 2. Optical Flow (RAFT)
        self.flow_model = None
        self.prev_frame_tensor = None
        try:
            from torchvision.models.optical_flow import raft_small
            # Weights enum is usually Raft_Small_Weights.DEFAULT or similar in newer torchvision
            # We'll try loading with pretrained=True (deprecated but often works) or weights string
            self.flow_model = raft_small(pretrained=True)
            self.flow_model = self.flow_model.to(device)
            self.flow_model.eval()
            print("VisualExpert: Loaded RAFT (small) successfully.")
        except Exception as e:
            print(f"VisualExpert: Failed to load RAFT ({e}). Flow will be zero.")
        
        # 3. Visual Encoder (ResNet18 as feature extractor)
        # Replaced basic CNN with ResNet18 for better features
        try:
            import torchvision.models as models
            # Use older weights syntax for compatibility or just pretrained=True
            resnet = models.resnet18(pretrained=True)
            # Remove classification head (fc) and flatten
            self.encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            self.encoder_dim = 512
            print("VisualExpert: Loaded ResNet18 backbone.")
        except Exception as e:
            print(f"VisualExpert: Could not load ResNet18 ({e}). Using basic CNN.")
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 512) # 512-dim embedding
            )
            self.encoder_dim = 512
        
        self.encoder.to(device)
        
        # 4. System ID Head (The "Sim-to-Real" Adapter)
        # Predicts [mass, friction, restitution] from visual embedding
        self.system_id_head = nn.Sequential(
            nn.Linear(self.encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3), 
            nn.Sigmoid() # Normalized 0-1 for friction/restitution, scale mass later
        ).to(device)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs object detection.
        Returns list of dicts: {'bbox': [x1,y1,x2,y2], 'class': int, 'conf': float}
        """
        if self.detector is None:
            # Mock implementation for testing
            h, w, _ = frame.shape
            # Return a dummy object in the center
            return [{
                'bbox': [w//2 - 50, h//2 - 50, w//2 + 50, h//2 + 50],
                'class': 0,
                'conf': 0.95
            }]

        # Real Inference
        detections = []
        try:
            if self.use_yolo == "v5":
                # YOLOv5 expects RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector(frame_rgb)
                # results.xyxy[0] is [x1, y1, x2, y2, conf, cls]
                # We need to move to cpu if on cuda
                preds = results.xyxy[0].cpu().numpy()
                
                for *xyxy, conf, cls in preds:
                    detections.append({
                        'bbox': [float(x) for x in xyxy],
                        'class': int(cls),
                        'conf': float(conf)
                    })
        except Exception as e:
            print(f"VisualExpert: Detection failed ({e}). Returning empty.")
            
        return detections

    def get_features_and_props(self, frame: np.ndarray, bboxes: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts visual embeddings and infers physical properties for each bbox.
        """
        # Convert frame to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device) / 255.0
        
        embeddings = []
        props = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Crop
            crop = frame_tensor[:, y1:y2, x1:x2]
            if crop.numel() == 0:
                # Handle edge case of empty crop
                emb = torch.zeros(512, device=self.device)
                prop = torch.tensor([1.0, 0.5, 0.5], device=self.device)
            else:
                # Resize to fixed size for simple encoder (e.g., 224x224)
                crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=(224, 224))
                emb = self.encoder(crop).squeeze(0)
                prop = self.system_id_head(emb)
                
            embeddings.append(emb)
            props.append(prop)
            
        if not embeddings:
            return torch.empty(0, 512), torch.empty(0, 3)
            
        return torch.stack(embeddings), torch.stack(props)

    def compute_flow(self, current_frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes optical flow between prev_frame and current frame.
        Input tensors should be [1, 3, H, W] in range [0, 255].
        Returns flow [2, H, W].
        """
        if self.flow_model is None or self.prev_frame_tensor is None:
            return torch.zeros(2, current_frame_tensor.shape[2], current_frame_tensor.shape[3], device=self.device)
        
        with torch.no_grad():
            # RAFT expects images to be contiguous
            list_of_flows = self.flow_model(self.prev_frame_tensor, current_frame_tensor)
            predicted_flow = list_of_flows[-1][0] # [2, H, W]
            
        return predicted_flow

    def estimate_camera_motion(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Estimates camera motion (ego-motion) from dense flow.
        Returns 4x4 pose matrix (delta).
        Simplified: Uses average flow to estimate translation.
        """
        # flow is [2, H, W]
        # Calculate median flow to ignore moving objects (outliers)
        # Downsample for speed
        flow_small = torch.nn.functional.interpolate(flow.unsqueeze(0), scale_factor=0.1, mode='bilinear')
        median_flow = flow_small.view(2, -1).median(dim=1).values # [dx, dy]
        
        dx, dy = median_flow[0].item(), median_flow[1].item()
        
        # Construct 4x4 matrix (Translation only for now)
        # Assuming Z-translation is 0
        pose = torch.eye(4, device=self.device)
        pose[0, 3] = -dx # Camera moves opposite to flow
        pose[1, 3] = -dy
        
        return pose

    def process_frame(self, timestamp: float, frame: np.ndarray) -> Tuple[List[ObjectNode], torch.Tensor, torch.Tensor]:
        """
        Full pipeline for a single frame.
        Returns: (new_nodes, camera_pose_delta, global_flow_vector)
        """
        # Prepare frame for RAFT (0-255, [1, 3, H, W])
        frame_tensor_raft = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        
        # 1. Compute Flow & Camera Motion
        flow = self.compute_flow(frame_tensor_raft)
        camera_pose = self.estimate_camera_motion(flow)
        global_flow = flow.mean(dim=(1, 2)) # Simple average for background node
        
        # Update history
        self.prev_frame_tensor = frame_tensor_raft
        
        # 2. Detect Objects
        detections = self.detect(frame)
        bboxes = [d['bbox'] for d in detections]
        
        embeddings, phys_props = self.get_features_and_props(frame, bboxes)
        
        new_nodes = []
        for i, det in enumerate(detections):
            # Unpack inferred physics
            mass_norm, fric, rest = phys_props[i]
            mass = mass_norm * 10.0 # Scale mass to 0-10kg range
            
            # Estimate object velocity from flow if possible
            # (Sample flow at bbox center)
            cx, cy = int((det['bbox'][0]+det['bbox'][2])/2), int((det['bbox'][1]+det['bbox'][3])/2)
            h, w = frame.shape[:2]
            cx, cy = min(max(0, cx), w-1), min(max(0, cy), h-1)
            
            obj_flow = flow[:, cy, cx] # [vx, vy] (pixels/frame)
            velocity = torch.tensor([obj_flow[0], obj_flow[1], 0.0], device=self.device) * 30.0 # Scale to pixels/sec (assuming 30fps)

            node = ObjectNode(
                id=i, # Temporary ID, Tracker will assign real ID
                class_id=det['class'],
                confidence=det['conf'],
                position=torch.tensor([(det['bbox'][0]+det['bbox'][2])/2, (det['bbox'][1]+det['bbox'][3])/2, 0.0], device=self.device),
                velocity=velocity, 
                bbox=torch.tensor(det['bbox'], device=self.device),
                embedding=embeddings[i],
                mass=mass.item(),
                friction=fric.item(),
                restitution=rest.item(),
                last_seen_timestamp=timestamp
            )
            new_nodes.append(node)
            
        return new_nodes, camera_pose, global_flow
