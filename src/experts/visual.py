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
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_raft=True):
        super().__init__()
        self.device = device
        self.use_raft = use_raft
        
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
        if self.use_raft:
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
        else:
            print("VisualExpert: RAFT disabled. Using Farneback (OpenCV) for flow.")
        
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
        
        # Bus projections (for cross-expert KV sharing)
        self.bus_dim = 128
        self.k_proj = nn.Linear(self.encoder_dim, self.bus_dim).to(device)
        self.v_proj = nn.Linear(self.encoder_dim, self.bus_dim).to(device)
        self._last_embeddings = None

        # 5. KV projections for Attention Bus
        self.d_model = 128
        self.k_proj = nn.Linear(self.encoder_dim, self.d_model).to(device)
        self.v_proj = nn.Linear(self.encoder_dim, self.d_model).to(device)

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
            self._last_embeddings = torch.empty(0, self.encoder_dim, device=self.device)
            return torch.empty(0, 512), torch.empty(0, 3)
        
        embs = torch.stack(embeddings)
        self._last_embeddings = embs
        return embs, torch.stack(props)

    def produce_kv(self, embeddings: torch.Tensor = None):
        """Produce key/value tensors for the bus from embeddings.

        embeddings: [N, E]
        returns k: [N, D], v: [N, D]
        """
        if embeddings is None:
            embeddings = self._last_embeddings
        if embeddings is None or embeddings.numel() == 0:
            return torch.empty((0, self.bus_dim), device=self.device), torch.empty((0, self.bus_dim), device=self.device)

        k = self.k_proj(embeddings)
        v = self.v_proj(embeddings)
        return k, v

    def compute_flow(self, current_frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes optical flow between prev_frame and current frame.
        Input tensors should be [1, 3, H, W] in range [0, 255].
        Returns flow [2, H, W].
        """
        if self.prev_frame_tensor is None:
            return torch.zeros(2, current_frame_tensor.shape[2], current_frame_tensor.shape[3], device=self.device)
        
        if self.flow_model is not None:
            # Use RAFT
            with torch.no_grad():
                # RAFT expects images to be contiguous
                list_of_flows = self.flow_model(self.prev_frame_tensor, current_frame_tensor)
                predicted_flow = list_of_flows[-1][0] # [2, H, W]
            return predicted_flow
        else:
            # Use Farneback (CPU fallback)
            # Convert tensors to numpy grayscale
            prev_gray = self._tensor_to_gray_numpy(self.prev_frame_tensor)
            curr_gray = self._tensor_to_gray_numpy(current_frame_tensor)
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # flow is [H, W, 2], convert to [2, H, W] tensor
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).to(self.device)
            return flow_tensor

    def _tensor_to_gray_numpy(self, tensor):
        # tensor: [1, 3, H, W], 0-255
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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

    def produce_kv(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce key/value tensors from per-object embeddings.
        embeddings: [N, encoder_dim]
        returns k:[N, d_model], v:[N, d_model]
        """
        if embeddings.numel() == 0:
            return torch.empty((0, self.d_model), device=self.device), torch.empty((0, self.d_model), device=self.device)
        k = self.k_proj(embeddings)
        v = self.v_proj(embeddings)
        return k, v

    def refine_with_slots(self, frame: np.ndarray, predicted_slots: List[Dict]) -> List[ObjectNode]:
        """
        Performs targeted detection based on predicted slot locations.
        Used to recover objects lost by the global detector.
        
        predicted_slots: List of dicts with 'bbox' (predicted), 'id', and optionally 'embedding'.
        """
        recovered_nodes = []
        if not predicted_slots:
            return recovered_nodes

        # YOLOv5 expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for slot in predicted_slots:
            # Define search region (e.g., 2x the predicted bbox)
            pred_bbox = slot['bbox']
            x1, y1, x2, y2 = pred_bbox
            w_box, h_box = x2 - x1, y2 - y1
            
            # Expand search region
            margin_w = w_box * 0.5
            margin_h = h_box * 0.5
            
            sx1 = max(0, int(x1 - margin_w))
            sy1 = max(0, int(y1 - margin_h))
            sx2 = min(frame.shape[1], int(x2 + margin_w))
            sy2 = min(frame.shape[0], int(y2 + margin_h))
            
            if sx2 <= sx1 or sy2 <= sy1:
                continue
                
            # Crop
            search_crop = frame_rgb[sy1:sy2, sx1:sx2]
            
            # Run Detector on Crop (Lower threshold implicitly by being closer/focused?)
            # Actually, we might want to lower the conf threshold if possible, 
            # but standard YOLO call uses default.
            # However, zooming in often helps small objects.
            
            try:
                if self.use_yolo == "v5":
                    results = self.detector(search_crop)
                    preds = results.xyxy[0].cpu().numpy()
                    
                    # Filter and adjust coordinates
                    best_det = None
                    best_iou = 0.0
                    
                    for *xyxy, conf, cls in preds:
                        # Local coords
                        lx1, ly1, lx2, ly2 = xyxy
                        
                        # Global coords
                        gx1, gy1, gx2, gy2 = lx1 + sx1, ly1 + sy1, lx2 + sx1, ly2 + sy1
                        
                        # Check overlap with predicted bbox to ensure we found the RIGHT object
                        # (Simple IoU check with the prediction)
                        # Intersection
                        ix1 = max(x1, gx1); iy1 = max(y1, gy1)
                        ix2 = min(x2, gx2); iy2 = min(y2, gy2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        union = (w_box * h_box) + ((gx2-gx1)*(gy2-gy1)) - inter
                        iou = inter / (union + 1e-6)
                        
                        if iou > 0.1: # Loose overlap required
                            if best_det is None or conf > best_det['conf']:
                                best_det = {
                                    'bbox': [gx1, gy1, gx2, gy2],
                                    'class': int(cls),
                                    'conf': float(conf)
                                }
                                
                    if best_det:
                        # Extract features for this new detection
                        # We need to call get_features_and_props for just this one
                        # But get_features_and_props expects a list of bboxes and full frame
                        # We can reuse the existing method
                        
                        # Create a temporary node
                        # We will fill embedding later in batch or now
                        # Let's do it now for simplicity
                        embs, props = self.get_features_and_props(frame, [best_det['bbox']])
                        
                        # Create Node
                        # Note: We don't assign the ID here, the Tracker will match it.
                        # But since we searched FOR a specific slot, we strongly suspect it's that ID.
                        # However, to keep architecture clean, we return it as a "candidate" 
                        # and let the tracker merge it (or we force the ID if we are sure).
                        # Let's return it as a candidate with a hint? 
                        # For now, just a candidate.
                        
                        mass_norm, fric, rest = props[0]
                        
                        node = ObjectNode(
                            id=-1, # Temporary
                            class_id=best_det['class'],
                            confidence=best_det['conf'],
                            position=torch.tensor([(best_det['bbox'][0]+best_det['bbox'][2])/2, (best_det['bbox'][1]+best_det['bbox'][3])/2, 0.0], device=self.device),
                            velocity=torch.zeros(3, device=self.device), # Unknown velocity initially
                            bbox=torch.tensor(best_det['bbox'], device=self.device),
                            embedding=embs[0],
                            mass=mass_norm.item() * 10.0,
                            friction=fric.item(),
                            restitution=rest.item(),
                            last_seen_timestamp=0.0 # Will be set by caller
                        )
                        # Add a hint about which slot triggered this
                        node.attributes['suggested_id'] = slot['id']
                        recovered_nodes.append(node)
                        
            except Exception as e:
                print(f"VisualExpert: Refinement failed for slot {slot['id']} ({e})")
                
        return recovered_nodes
