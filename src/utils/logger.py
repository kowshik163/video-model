import os
import json
import cv2
import numpy as np
from typing import Dict, Any

class HierarchicalLogger:
    def __init__(self, video_name: str, base_dir: str = "logs"):
        self.video_name = os.path.splitext(os.path.basename(video_name))[0]
        self.root_dir = os.path.join(base_dir, self.video_name)
        self.models = ['visual', 'physics', 'supervisor', 'geometry', 'memory', 'bus']
        self._setup_dirs()
        print(f"Logging initialized at: {self.root_dir}")

    def _setup_dirs(self):
        for model in self.models:
            os.makedirs(os.path.join(self.root_dir, model), exist_ok=True)
            
        # Create crops directory specifically
        os.makedirs(os.path.join(self.root_dir, "visual", "crops"), exist_ok=True)

    def log_object(self, model: str, obj_id: int, data: Dict[str, Any], timestamp: float):
        """Log data specific to an object instance."""
        # Create a subfolder for object logs if we want to keep them very separate, 
        # or just files. User asked for "sub files for each object logs in eah model".
        # Let's use a file per object.
        file_path = os.path.join(self.root_dir, model, f"obj_{obj_id}.jsonl")
        
        # Convert numpy/torch types to serializable
        clean_data = self._sanitize(data)
        entry = {"timestamp": timestamp, **clean_data}
        
        with open(file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_model_event(self, model: str, event_type: str, data: Dict[str, Any], timestamp: float):
        """Log general model-level events (e.g. Supervisor decisions, Global Flow)."""
        file_path = os.path.join(self.root_dir, model, f"{event_type}.jsonl")
        clean_data = self._sanitize(data)
        entry = {"timestamp": timestamp, **clean_data}
        
        with open(file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_crop(self, obj_id: int, frame: np.ndarray, bbox: list, timestamp: float):
        """Save the visual crop of an object for dataset building/debugging."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            # Save as timestamp_objID.jpg
            filename = f"{int(timestamp*1000):06d}_obj{obj_id}.jpg"
            path = os.path.join(self.root_dir, "visual", "crops", filename)
            cv2.imwrite(path, crop)

    def _sanitize(self, data):
        """Recursively convert numpy/torch types to python natives."""
        if hasattr(data, 'tolist'):  # Tensor/Numpy array
            return data.tolist()
        if hasattr(data, 'item'):  # Tensor/Numpy scalar
            try:
                return data.item()
            except ValueError:
                return data.tolist()
        if isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize(v) for v in data]
        return data

    def generate_summary(self):
        """Reads logs and produces a high-level summary."""
        summary_path = os.path.join(self.root_dir, "video_understanding_summary.txt")
        
        # Count frames
        frames = 0
        if os.path.exists(os.path.join(self.root_dir, "visual", "global_flow.jsonl")):
            with open(os.path.join(self.root_dir, "visual", "global_flow.jsonl")) as f:
                frames = sum(1 for _ in f)
                
        # Count 3D triggers
        triggers = 0
        if os.path.exists(os.path.join(self.root_dir, "supervisor", "decisions.jsonl")):
            with open(os.path.join(self.root_dir, "supervisor", "decisions.jsonl")) as f:
                for line in f:
                    if json.loads(line).get("needs_3d", False):
                        triggers += 1
                        
        # Unique objects
        obj_files = [f for f in os.listdir(os.path.join(self.root_dir, "memory")) if f.startswith("obj_")]
        num_objects = len(obj_files)
        
        with open(summary_path, "w") as f:
            f.write(f"--- Video Understanding Summary: {self.video_name} ---\n")
            f.write(f"Total Frames Processed: {frames}\n")
            f.write(f"Unique Objects Tracked: {num_objects}\n")
            f.write(f"3D Expert Triggers: {triggers} ({(triggers/frames*100) if frames else 0:.1f}% of frames)\n")
            f.write("\n")
            f.write("Narrative:\n")
            f.write(f"The video contains {num_objects} distinct entities.\n")
            if triggers > frames * 0.5:
                f.write("High complexity/inconsistency detected. The system frequently relied on 3D geometry refinement.\n")
            else:
                f.write("The scene was relatively stable, handled mostly by the fast visual-physics loop.\n")
                
        print(f"Summary generated at: {summary_path}")
