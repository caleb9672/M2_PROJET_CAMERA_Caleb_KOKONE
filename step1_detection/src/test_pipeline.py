import yaml
import os
from detector import VideoProcessor

def test_single_video():
    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Force only one video for testing
    config['save_video'] = True
    processor = VideoProcessor(config)
    
    video_path = os.path.join(config['video_dir'], "CAMERA_HALL_PORTE_DROITE.MP4")
    if os.path.exists(video_path):
        processor.process_video(video_path)
    else:
        print(f"Video not found: {video_path}")

if __name__ == "__main__":
    test_single_video()
