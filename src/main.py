import argparse
import torch
import numpy as np
from src.sgw.workspace import SharedGlobalWorkspace
from src.experts.visual import VisualExpert
from src.experts.physics import PhysicsExpert
from src.experts.memory import ObjectMemory
from src.experts.geometry import GeometryExpert
from src.experts.threed import ThreeDExpert
from src.experts.bus import AttentionBus
from src.supervisor.router import Supervisor
from src.utils.video_loader import VideoLoader
from src.utils.logger import HierarchicalLogger
from src.utils.transition_detector import detect_cut, motion_spike, detect_transition_sequence
from src.utils.audio_beats import detect_audio_beats

def run_inference(video_path: str, output_dir: str = "logs", device: str = None, use_raft: bool = True):
    """
    Runs the full SGW inference loop on a single video.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print(f"Initializing Unified SGW Video Analysis System on {device}...")
    
    # 1. Initialize Components
    sgw = SharedGlobalWorkspace()
    visual_expert = VisualExpert(device=device, use_raft=use_raft)
    
    # Load trained visual weights if available
    try:
        visual_expert.load_state_dict(torch.load('visual_trained.pth', map_location=device))
        print("Loaded trained Visual Expert weights.")
    except Exception as e:
        print(f"Could not load trained visual weights ({e}). Using initialized weights.")

    physics_expert = PhysicsExpert()
    physics_expert.to(device)
    
    # Load pretrained physics weights and freeze
    try:
        physics_expert.load_weights('physics_pretrained.pth')
        physics_expert.freeze()
        print("Loaded and froze pretrained Physics Expert.")
    except Exception as e:
        print(f"Could not load physics weights ({e}). Using Newtonian fallback.")
    
    geometry_expert = GeometryExpert(device=device)
    three_d_expert = ThreeDExpert(device=device)
    memory = ObjectMemory()
    supervisor = Supervisor()
    
    # Load trained supervisor weights if available
    try:
        supervisor.load_state_dict(torch.load('supervisor_trained.pth', map_location=device))
        print("Loaded trained Supervisor weights.")
    except Exception as e:
        print(f"Could not load trained supervisor weights ({e}). Using initialized weights.")
        
    supervisor.to(device)
    
    # Initialize Attention Bus
    bus = AttentionBus(d_model=128, device=device)

    print(f"Starting Inference Loop for {video_path}...")
    
    # Load Video
    logger = HierarchicalLogger(video_path, base_dir=output_dir)
    
    try:
        loader = VideoLoader(video_path)
        print(f"Loaded video: {video_path}")
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        print("Falling back to mock generator.")
        loader = mock_video_generator()
    
    frame_count = 0
    
    prev_frame = None
    recent_frames = []
    frame_buffer = [] # Buffer for 3D Expert (tensors)
    # Optional: precompute audio beats (best-effort)
    try:
        beat_times = detect_audio_beats(video_path)
        if beat_times:
            print(f"Detected {len(beat_times)} audio beats (will log to visual/).")
    except Exception:
        beat_times = []

    for timestamp, frame in loader:
        frame_count += 1
        print(f"--- Processing Frame {frame_count} at t={timestamp:.2f}s ---")
        
        # Update 3D Expert Buffer
        # Convert to tensor (C, H, W) normalized [0, 1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_buffer.append(frame_tensor)
        if len(frame_buffer) > 16:
            frame_buffer.pop(0)
        
        # Clear bus for new frame
        bus.clear()
        
        # A. Visual Perception
        # 1. Detect & Extract Features
        # Returns: nodes, camera_pose, global_flow
        raw_nodes, camera_pose, global_flow = visual_expert.process_frame(timestamp, frame)
        print(f"Visual Expert: Detected {len(raw_nodes)} objects. Global Flow: {global_flow.tolist()}")
        
        # Log Visual Events
        logger.log_model_event('visual', 'global_flow', {'flow': global_flow}, timestamp)

        # Transition detection (cut/hard-change)
        transition_detected = False
        transition_info = {}
        if prev_frame is not None:
            cut, info = detect_cut(prev_frame, frame)
            spike, spinfo = motion_spike(prev_frame, frame)
            # Also consider short sequence gradual transitions
            recent_frames.append(frame.copy())
            if len(recent_frames) > 5:
                recent_frames.pop(0)
            seq_tr, seq_info = detect_transition_sequence(recent_frames, window=5)
            if cut or spike or seq_tr:
                transition_detected = True
                transition_info.update({'cut': cut, 'cut_info': info, 'motion_spike': spike, 'motion_info': spinfo, 'seq_tr': seq_tr, 'seq_info': seq_info})
                logger.log_model_event('visual', 'transitions', transition_info, timestamp)
        else:
            recent_frames.append(frame.copy())
        
        # Publish Visual KV to Bus
        if raw_nodes:
            # We need embeddings to produce KV. 
            # visual_expert.process_frame calls get_features_and_props which stores _last_embeddings
            embeddings = torch.stack([n.embedding for n in raw_nodes])
            k_vis, v_vis = visual_expert.produce_kv(embeddings)
            bus.publish('visual', k_vis, v_vis)
            
            # Log Raw Detections & Save Crops
            for node in raw_nodes:
                logger.save_crop(node.id, frame, node.bbox.tolist(), timestamp)
                logger.log_object('visual', node.id, {
                    'bbox': node.bbox,
                    'confidence': node.confidence,
                    'class_id': node.class_id
                }, timestamp)
        
        # B. Memory & Tracking
        # 2. Associate with existing tracks
        tracked_nodes = memory.update(raw_nodes, timestamp)
        print(f"Memory: Tracking {len(tracked_nodes)} objects.")
        
        # Log Memory State
        for node in tracked_nodes:
            logger.log_object('memory', node.id, {
                'position': node.position,
                'velocity': node.velocity,
                'mass': node.mass
            }, timestamp)
        
        # C. Physics Prediction (The "Imagination")
        # 3. Predict where objects SHOULD be at t+1
        # (In a real loop, we'd compare t's prediction with t's observation)
        current_graph = sgw.current_graph
        # Populate current graph with tracked nodes for the physics engine
        for node in tracked_nodes:
            current_graph.add_object(node)
            
        predicted_deltas = physics_expert.predict_next_state(current_graph, dt=1/30.0)
        
        # Log Physics Predictions
        logger.log_model_event('physics', 'predictions', {'deltas': predicted_deltas}, timestamp)
        
        # Publish Physics KV to Bus (from current state)
        if tracked_nodes:
            # Sort by ID to match physics expert internal logic
            tracked_nodes.sort(key=lambda n: n.id)
            node_states = torch.stack([n.to_tensor() for n in tracked_nodes]).to(device)
            k_phys, v_phys = physics_expert.produce_kv(node_states)
            bus.publish('physics', k_phys, v_phys)
        
        # D. Supervision & Routing
        # 4. Check Consistency
        # (Simplified: comparing current observation with previous prediction would happen here)
        consistency_score = 0.1 # Mock score
        # If a transition or audio beat occurs at this timestamp, increase complexity
        complexity_score = 0.5
        if transition_detected:
            complexity_score = 1.0
        # audio beat: if close to beat time, raise complexity slightly
        if beat_times:
            # find if there's a beat within 0.06s of this frame
            for bt in beat_times:
                if abs(bt - timestamp) < 0.06:
                    complexity_score = max(complexity_score, 0.9)
                    logger.log_model_event('visual', 'audio_beat', {'beat_time': bt}, timestamp)
                    break
        
        # Supervisor attends to bus to get context
        # Query is a learned parameter or simple aggregate. 
        # For now, let's use a zero query to get global context
        query = torch.zeros(1, 128, device=device) 
        bus_context = bus.read_and_attend(query)
        
        needs_3d, _ = supervisor.decide_routing(consistency_score, complexity_score=complexity_score, bus_context=bus_context)
        
        # Log Supervisor Decision
        logger.log_model_event('supervisor', 'decisions', {
            'consistency': consistency_score,
            'needs_3d': needs_3d
        }, timestamp)
        
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
                    
                    # Store 3D BBox and Contact Points in attributes
                    node.attributes["3d_bbox"] = refined_data["3d_bbox"].tolist()
                    node.attributes["contact_points"] = refined_data["contact_points"].tolist()
                    
                    print(f"  -> Refined Node {node.id}: Updated 3D State from Geometry Expert. New Depth: {node.position[2]:.2f}")
                    
                    # Log Geometry Refinement
                    logger.log_object('geometry', node.id, {
                        'refined_depth': new_depth,
                        '3d_bbox': node.attributes["3d_bbox"],
                        'contact_points': node.attributes["contact_points"]
                    }, timestamp)
            
            # Trigger 3D Spatiotemporal Expert
            if len(frame_buffer) >= 8:
                print("  -> Triggering 3D Spatiotemporal Expert (R3D-18)...")
                for node in tracked_nodes:
                    analysis = three_d_expert.analyze_roi(frame_buffer, node.bbox.tolist())
                    if "3d_embedding" in analysis:
                        node.attributes["3d_embedding"] = analysis["3d_embedding"]
                        logger.log_object('threed', node.id, {'embedding_dim': len(analysis["3d_embedding"])}, timestamp)
        else:
            print("Supervisor: Consistency OK. Using Fast Path.")
            
        # E. Update Global Workspace
        # 5. Commit state
        sgw.write_update(timestamp, tracked_nodes, [], camera_pose)
        prev_frame = frame.copy()
        
    print("Processing Complete.")
    logger.generate_summary()

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
    parser = argparse.ArgumentParser(description="Run SGW Inference on a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu).")
    parser.add_argument("--fast_flow", action="store_true", help="Use fast optical flow (Farneback) instead of RAFT.")
    
    args = parser.parse_args()
    
    run_inference(args.video_path, args.output_dir, args.device, use_raft=not args.fast_flow)
