import torch
import numpy as np
from src.sgw.workspace import SharedGlobalWorkspace
from src.experts.visual import VisualExpert
from src.experts.physics import PhysicsExpert
from src.experts.memory import ObjectMemory
from src.experts.geometry import GeometryExpert
from src.supervisor.router import Supervisor
from src.utils.video_loader import VideoLoader

def main():
    print("Initializing Unified SGW Video Analysis System...")
    
    # 1. Initialize Components
    sgw = SharedGlobalWorkspace()
    visual_expert = VisualExpert()
    physics_expert = PhysicsExpert()
    geometry_expert = GeometryExpert()
    memory = ObjectMemory()
    supervisor = Supervisor()
    
    # 2. Load Video (Mocking for now if file doesn't exist)
    try:
        loader = VideoLoader("test_video.mp4")
    except ValueError:
        print("No video found, using mock frame generator.")
        loader = mock_video_generator()

    print("Starting Inference Loop...")
    
    for timestamp, frame in loader:
        print(f"--- Processing Frame at t={timestamp:.2f}s ---")
        
        # A. Visual Perception
        # 1. Detect & Extract Features
        # Returns: nodes, camera_pose, global_flow
        raw_nodes, camera_pose, global_flow = visual_expert.process_frame(timestamp, frame)
        print(f"Visual Expert: Detected {len(raw_nodes)} objects. Global Flow: {global_flow.tolist()}")
        
        # B. Memory & Tracking
        # 2. Associate with existing tracks
        tracked_nodes = memory.update(raw_nodes, timestamp)
        print(f"Memory: Tracking {len(tracked_nodes)} objects.")
        
        # C. Physics Prediction (The "Imagination")
        # 3. Predict where objects SHOULD be at t+1
        # (In a real loop, we'd compare t's prediction with t's observation)
        current_graph = sgw.current_graph
        # Populate current graph with tracked nodes for the physics engine
        for node in tracked_nodes:
            current_graph.add_object(node)
            
        predicted_deltas = physics_expert.predict_next_state(current_graph, dt=1/30.0)
        
        # D. Supervision & Routing
        # 4. Check Consistency
        # (Simplified: comparing current observation with previous prediction would happen here)
        consistency_score = 0.1 # Mock score
        needs_3d, _ = supervisor.decide_routing(consistency_score)
        
        if needs_3d:
            print("Supervisor: CONSISTENCY VIOLATION. Triggering 3D Expert...")
            # Trigger 3D analysis on specific nodes
            # In a real system, Supervisor provides specific node IDs to refine.
            # Here, we refine all tracked nodes as a fallback.
            for node in tracked_nodes:
                # Extract crop
                x1, y1, x2, y2 = map(int, node.bbox.tolist())
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                    
                    # Mock depth prior (z-coord)
                    depth_prior = node.position[2].unsqueeze(0)
                    
                    # Run 3D Expert
                    refined_data = geometry_expert.refine_roi(crop_tensor, depth_prior)
                    
                    # FEEDBACK LOOP: Update the Node state with refined 3D data
                    # This is the critical step where "Thinking" updates "Perception"
                    # e.g., Update Z-position based on refined depth
                    
                    new_depth = refined_data["refined_depth"].to(node.position.device)
                    node.position[2] = new_depth.item()
                    
                    print(f"  -> Refined Node {node.id}: Updated 3D State from Geometry Expert. New Depth: {node.position[2]:.2f}")
        else:
            print("Supervisor: Consistency OK. Using Fast Path.")
            
        # E. Update Global Workspace
        # 5. Commit state
        sgw.write_update(timestamp, tracked_nodes, [], camera_pose)
        
    print("Processing Complete.")

def mock_video_generator():
    """Generates dummy frames for testing."""
    for i in range(5):
        # Create a black frame with a moving white square
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        x = (i * 50) % 640
        cv2.rectangle(frame, (x, 300), (x+50, 350), (255, 255, 255), -1)
        yield i * 0.033, frame
        
import cv2 # Import here for the mock generator

if __name__ == "__main__":
    main()
