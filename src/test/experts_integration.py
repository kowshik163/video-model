import sys
import os
import torch
import numpy as np
import cv2

# Make repo root importable when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experts.visual import VisualExpert
from src.experts.physics import PhysicsExpert
from src.experts.bus import AttentionBus


def run_integration():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    vis = VisualExpert(device=device)
    phys = PhysicsExpert()
    bus = AttentionBus(d_model=128, device=device)

    # Create a mock frame (640x640 with a white square)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (300, 300), (350, 350), (255, 255, 255), -1)

    ts = 0.0
    nodes, cam_pose, global_flow = vis.process_frame(ts, frame)
    print('Detected nodes:', len(nodes))

    # Get embeddings from visual by calling get_features_and_props on the bbox list
    if nodes:
        bboxes = [node.bbox.cpu().numpy().tolist() for node in nodes]
        embeddings, props = vis.get_features_and_props(frame, bboxes)
        k_vis, v_vis = vis.produce_kv(embeddings)
        bus.publish('vision', k_vis, v_vis)

    # Physics produces KV from node tensors
    # Build a simple node_states tensor to feed physics.produce_kv
    if nodes:
        node_states = torch.stack([n.to_tensor() for n in nodes]).to(device)
        k_phys, v_phys = phys.produce_kv(node_states)
        bus.publish('physics', k_phys, v_phys)

    # Let a query be the mean visual embedding
    if nodes:
        query = embeddings.mean(dim=0, keepdim=True)  # [1, D]
        # Project query into bus d_model size (use vis.k_proj as projection)
        query_k = vis.k_proj(query.to(device))
        context = bus.read_and_attend(query_k)
        print('Context shape:', context.shape)
    else:
        print('No nodes detected; nothing to attend.')


if __name__ == '__main__':
    run_integration()
