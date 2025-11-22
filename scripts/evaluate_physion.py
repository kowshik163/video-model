import os
import glob
import argparse
import pandas as pd
import time
import json
import torch
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import run_inference

def evaluate_dataset(input_dir: str, output_dir: str):
    """
    Runs inference on all videos in input_dir and aggregates metrics.
    """
    video_files = glob.glob(os.path.join(input_dir, "*.mp4")) + \
                  glob.glob(os.path.join(input_dir, "*.avi"))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos. Starting evaluation...")
    
    results = []
    
    for video_path in tqdm(video_files):
        video_name = os.path.basename(video_path)
        print(f"\nProcessing {video_name}...")
        
        start_time = time.time()
        try:
            # Run inference (this will generate logs in output_dir/video_name)
            run_inference(video_path, output_dir=output_dir)
            status = "Success"
        except Exception as e:
            print(f"Failed to process {video_name}: {e}")
            status = "Failed"
            
        duration = time.time() - start_time
        
        # Parse the generated summary log to get metrics
        # We assume run_inference calls logger.generate_summary() which writes to video_understanding_summary.txt
        # But simpler is to parse the raw logs we just created.
        
        log_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])
        metrics = parse_logs(log_dir)
        
        results.append({
            "video": video_name,
            "status": status,
            "duration_sec": duration,
            **metrics
        })
        
    # Save aggregate report
    df = pd.DataFrame(results)
    report_path = os.path.join(output_dir, "evaluation_report.csv")
    df.to_csv(report_path, index=False)
    print(f"\nEvaluation complete. Report saved to {report_path}")
    print(df)

def parse_logs(log_dir):
    """Reads the JSONL logs to extract metrics."""
    metrics = {
        "frames": 0,
        "objects_tracked": 0,
        "3d_triggers": 0,
        "avg_consistency": 0.0
    }
    
    if not os.path.exists(log_dir):
        return metrics
        
    # Count frames
    flow_log = os.path.join(log_dir, "visual", "global_flow.jsonl")
    if os.path.exists(flow_log):
        with open(flow_log) as f:
            metrics["frames"] = sum(1 for _ in f)
            
    # Count objects
    memory_dir = os.path.join(log_dir, "memory")
    if os.path.exists(memory_dir):
        metrics["objects_tracked"] = len([f for f in os.listdir(memory_dir) if f.startswith("obj_")])
        
    # Supervisor stats
    decision_log = os.path.join(log_dir, "supervisor", "decisions.jsonl")
    if os.path.exists(decision_log):
        triggers = 0
        total_consistency = 0.0
        count = 0
        with open(decision_log) as f:
            for line in f:
                data = json.loads(line)
                if data.get("needs_3d"):
                    triggers += 1
                total_consistency += data.get("consistency", 0.0)
                count += 1
        
        metrics["3d_triggers"] = triggers
        metrics["avg_consistency"] = total_consistency / max(count, 1)
        
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SGW System on a dataset.")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing video files.")
    parser.add_argument("--output_dir", type=str, default="eval_logs", help="Directory to save logs and report.")
    
    args = parser.parse_args()
    
    evaluate_dataset(args.input_dir, args.output_dir)
