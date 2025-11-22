"""
Master Orchestration Script for the Video Analysis System (Phase 1-6).

This script automates the entire lifecycle of the project:
1.  **Data Generation** (Phase 6): Generates synthetic physics data (Kubric/Placeholder).
2.  **Physics Training** (Phase 6): Trains the Physics Expert on synthetic data.
3.  **Distillation / Fine-tuning** (Phase 6): Fine-tunes the Visual Expert & Supervisor using the frozen Physics Expert as a teacher (Self-Supervised Consistency).
4.  **Inference** (Phase 1-5): Runs the fully trained system on a target video.

Usage:
    python run_full_pipeline.py --video_path video1.mp4 --output_dir runs/run_001

Options:
    --skip_data_gen     Skip synthetic data generation.
    --skip_training     Skip physics training.
    --skip_finetuning   Skip visual/supervisor fine-tuning.
    --skip_inference    Skip final inference.
    --fast              Use fast settings (fewer epochs, smaller datasets) for testing.
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, step_name):
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"CMD: {command}")
    print(f"{'='*60}\n")
    
    try:
        # Run command and stream output
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=os.environ.copy() # Pass current environment (PYTHONPATH etc)
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(f"[{step_name}] {line}", end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\nERROR: {step_name} failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
        print(f"\nSUCCESS: {step_name} completed.")
        
    except Exception as e:
        print(f"\nEXCEPTION: {step_name} failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full Video Analysis Pipeline (Phase 1-6)")
    parser.add_argument("--video_path", type=str, default="video1.mp4", help="Path to video for final inference")
    parser.add_argument("--output_dir", type=str, default="runs/latest", help="Root directory for all outputs")
    
    # Flags to skip steps
    parser.add_argument("--skip_data_gen", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--skip_training", action="store_true", help="Skip physics training")
    parser.add_argument("--skip_finetuning", action="store_true", help="Skip fine-tuning/distillation")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference")
    
    # Config
    parser.add_argument("--fast", action="store_true", help="Run in fast/debug mode")
    parser.add_argument("--install_deps", action="store_true", help="Install dependencies before running (useful for Colab)")
    
    args = parser.parse_args()
    
    # ---------------------------------------------------------
    # 0. Setup / Install Dependencies
    # ---------------------------------------------------------
    if args.install_deps:
        print("Installing dependencies from requirements_phase6.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_phase6.txt"])
        # Also install src as editable if needed, or just rely on PYTHONPATH
    
    # Setup Paths
    base_dir = os.getcwd()
    # Ensure src is in PYTHONPATH
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = base_dir
    else:
        os.environ["PYTHONPATH"] = f"{base_dir}{os.pathsep}{os.environ['PYTHONPATH']}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = out_dir / "data"
    checkpoints_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    
    # Settings based on --fast flag
    num_clips = 10 if args.fast else 1000
    epochs_phys = 1 if args.fast else 10
    epochs_fine = 1 if args.fast else 5
    
    # ---------------------------------------------------------
    # 1. Data Generation (Phase 6)
    # ---------------------------------------------------------
    if not args.skip_data_gen:
        cmd = (
            f"python scripts/generate_synthetic_kubric.py "
            f"--out_dir {data_dir}/synthetic "
            f"--num_clips {num_clips} "
            f"--frames 32"
        )
        run_command(cmd, "Data Generation")
    else:
        print("Skipping Data Generation...")

    # ---------------------------------------------------------
    # 2. Train Physics Expert (Phase 6)
    # ---------------------------------------------------------
    if not args.skip_training:
        cmd = (
            f"python scripts/train_physics.py "
            f"--data_dir {data_dir}/synthetic "
            f"--checkpoint_dir {checkpoints_dir} "
            f"--epochs {epochs_phys} "
            f"--batch_size 4"
        )
        run_command(cmd, "Train Physics Expert")
    else:
        print("Skipping Physics Training...")

    # ---------------------------------------------------------
    # 3. Fine-tune / Distillation (Phase 6)
    # ---------------------------------------------------------
    # This step uses the "Physics Expert" (trained above) to teach the "Visual Expert"
    # on real (or synthetic) data.
    if not args.skip_finetuning:
        # We use the synthetic data for now as 'real' data proxy if no real data provided
        # In a real scenario, this would point to a folder of real videos
        ft_data = f"{data_dir}/synthetic" 
        
        cmd = (
            f"python scripts/finetune_visual_supervisor.py "
            f"--data_dir {ft_data} "
            f"--epochs {epochs_fine} "
            f"--fast_flow" # Use fast flow for speed in pipeline
        )
        run_command(cmd, "Fine-tune (Distillation)")
    else:
        print("Skipping Fine-tuning...")

    # ---------------------------------------------------------
    # 4. Inference (Phase 1-5)
    # ---------------------------------------------------------
    if not args.skip_inference:
        if not os.path.exists(args.video_path):
            print(f"WARNING: Video file {args.video_path} not found. Skipping inference.")
        else:
            # We run the main inference script
            # Note: In a real deployment, we would pass the checkpoints trained above
            # e.g., --visual_ckpt {checkpoints_dir}/visual.pth
            # For now, main.py loads default/hardcoded paths, but we'll run it to demonstrate flow.
            
            cmd = (
                f"python src/main.py {args.video_path} "
                f"--output_dir {logs_dir} "
                f"--fast_flow"
            )
            run_command(cmd, "Full Inference")
    else:
        print("Skipping Inference...")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Outputs located in: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
