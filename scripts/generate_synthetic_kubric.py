"""
Synthetic data generator for Phase 6 using Kubric.
This script is a documented stub: if `kubric` is available it will attempt to generate simple scenes.
Otherwise it prints instructions and produces a small synthetic numpy dataset for quick debugging.

Outputs:
 - `data/synthetic/<split>/*.npz` containing: frames (T,H,W,3), object_params (list of dicts), metadata.json

Usage:
    python scripts/generate_synthetic_kubric.py --out_dir data/synthetic --num_clips 100 --frames 64

Note: Kubric installation is optional. See README_PHASE6.md for full Kubric instructions.
"""
import argparse
import os
import json
import numpy as np

try:
    import kubric as kb
    KUBRIC_AVAILABLE = True
except Exception:
    KUBRIC_AVAILABLE = False


def generate_with_kubric(out_dir, num_clips, frames):
    # Minimal kubric example - more complex sampling should be added in real use
    for i in range(num_clips):
        scene_name = f"clip_{i:05d}"
        clip_dir = os.path.join(out_dir, scene_name)
        os.makedirs(clip_dir, exist_ok=True)
        # Placeholder: in a proper Kubric run you'd create assets, randomize mass/friction, render RGB and depth
        meta = {"source": "kubric_stub", "id": scene_name}
        with open(os.path.join(clip_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        # Create dummy frames for compatibility
        frames_arr = (np.random.rand(frames, 128, 128, 3) * 255).astype(np.uint8)
        np.savez(os.path.join(clip_dir, "frames.npz"), frames=frames_arr)


def generate_placeholder(out_dir, num_clips, frames):
    # Quick debug generator when kubric not installed
    for i in range(num_clips):
        scene_name = f"clip_{i:05d}"
        clip_dir = os.path.join(out_dir, scene_name)
        os.makedirs(clip_dir, exist_ok=True)
        meta = {"source": "placeholder", "id": scene_name, "mass": float(np.random.uniform(0.1, 5.0)), "friction": float(np.random.uniform(0.0, 1.0)), "restitution": float(np.random.uniform(0.0, 1.0))}
        with open(os.path.join(clip_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        frames_arr = (np.random.rand(frames, 128, 128, 3) * 255).astype(np.uint8)
        np.savez(os.path.join(clip_dir, "frames.npz"), frames=frames_arr)
        # Save per-clip object_params placeholder
        obj_params = [{"id": 0, "mass": meta["mass"], "friction": meta["friction"], "restitution": meta["restitution"]}]
        np.savez(os.path.join(clip_dir, "object_params.npz"), params=obj_params)
        
        # Generate dummy trajectories (linear motion for testing)
        # Shape: (T, NumObjects, 3) for pos, (T, NumObjects, 3) for vel
        # Let's simulate a simple projectile or linear path
        T = frames
        pos = np.zeros((T, 1, 3))
        vel = np.zeros((T, 1, 3))
        
        # Initial state
        p0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([1.0, 1.0, 0.0])
        
        for t in range(T):
            pos[t, 0] = p0 + v0 * (t * 0.1)
            vel[t, 0] = v0
            
        np.savez(os.path.join(clip_dir, "trajectories.npz"), pos=pos, vel=vel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/synthetic")
    parser.add_argument("--num_clips", type=int, default=10)
    parser.add_argument("--frames", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if KUBRIC_AVAILABLE:
        print("Kubric detected. Generating synthetic scenes (basic).")
        generate_with_kubric(args.out_dir, args.num_clips, args.frames)
    else:
        print("Kubric not available. Creating placeholder synthetic clips for debug.")
        generate_placeholder(args.out_dir, args.num_clips, args.frames)

    print(f"Done. Synthetic clips written to {args.out_dir}")

if __name__ == '__main__':
    main()
