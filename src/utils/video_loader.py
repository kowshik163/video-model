import cv2
import numpy as np
from typing import Generator, Tuple

class VideoLoader:
    """
    Efficiently loads video frames.
    """
    def __init__(self, video_path: str, resize_dim: Tuple[int, int] = (640, 640)):
        self.video_path = video_path
        self.resize_dim = resize_dim
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Yields (timestamp, frame) tuples.
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.resize_dim:
                frame = cv2.resize(frame, self.resize_dim)
                
            timestamp = frame_idx / self.fps
            yield timestamp, frame
            frame_idx += 1

    def close(self):
        self.cap.release()
