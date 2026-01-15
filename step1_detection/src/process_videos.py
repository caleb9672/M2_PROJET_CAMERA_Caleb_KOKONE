import os
import yaml
import multiprocessing
from detector import VideoProcessor

def process_single_video(args):
    video_path, config = args
    processor = VideoProcessor(config)
    try:
        processor.process_video(video_path)
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {e}")

def main():
    # Load configuration
    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Get all videos from the directory
    video_dir = config['video_dir']
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.MP4', '.avi'))]
    
    if not video_files:
        print(f"No videos found in {video_dir}")
        return

    print(f"Found {len(video_files)} videos to process.")
    
    # Prepare arguments for the pool
    tasks = [(os.path.join(video_dir, f), config) for f in video_files]
    
    # Use multiprocessing Pool
    num_workers = config.get('num_workers', 1)
    print(f"Starting processing with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_single_video, tasks)

    print("All videos have been processed.")

if __name__ == "__main__":
    main()
