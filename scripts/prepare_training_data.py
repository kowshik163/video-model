import os
import shutil
import argparse
import glob
import random
from pathlib import Path

def prepare_dataset(logs_dir, output_dir, sample_rate=1.0):
    """
    Harvests image crops from the logging directory to build a training dataset.
    
    Args:
        logs_dir: Root directory of logs (e.g., 'eval_logs')
        output_dir: Where to save the dataset (e.g., 'data/training_set')
        sample_rate: Fraction of images to keep (0.0 to 1.0)
    """
    print(f"Harvesting crops from {logs_dir}...")
    
    # Find all crop images
    # Structure: logs_dir / video_name / visual / crops / *.jpg
    search_pattern = os.path.join(logs_dir, "**", "visual", "crops", "*.jpg")
    crop_files = glob.glob(search_pattern, recursive=True)
    
    if not crop_files:
        print("No crops found. Make sure you've run inference with logging enabled.")
        return

    print(f"Found {len(crop_files)} total crops.")
    
    # Setup output directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    count = 0
    for crop_path in crop_files:
        if random.random() > sample_rate:
            continue
            
        # Create a unique name: video_timestamp_objID.jpg
        # crop_path example: .../video1/visual/crops/001520_obj0.jpg
        path_parts = Path(crop_path).parts
        # Assuming structure ends in: video_name / visual / crops / filename
        try:
            video_name = path_parts[-4]
            filename = path_parts[-1]
            new_name = f"{video_name}_{filename}"
            
            dest_path = os.path.join(images_dir, new_name)
            shutil.copy2(crop_path, dest_path)
            count += 1
        except IndexError:
            print(f"Skipping malformed path: {crop_path}")
            
    print(f"Successfully prepared {count} images in {images_dir}")
    print("Next steps:")
    print("1. Use a labeling tool (e.g., LabelImg, CVAT, Roboflow) to annotate these images.")
    print("2. Save annotations in YOLO format.")
    print("3. Use the 'train_detector.py' script (to be created) to fine-tune the Visual Expert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset from logged crops.")
    parser.add_argument("--logs_dir", type=str, default="eval_logs", help="Path to logs directory.")
    parser.add_argument("--output_dir", type=str, default="data/dataset_v1", help="Path to output dataset.")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Fraction of crops to sample.")
    
    args = parser.parse_args()
    
    prepare_dataset(args.logs_dir, args.output_dir, args.sample_rate)
