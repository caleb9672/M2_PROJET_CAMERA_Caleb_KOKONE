import cv2
import os
import csv
import time
from datetime import datetime, timedelta
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.model_path = config['model_path']
        
        # OpenVINO Optimization
        if config.get('use_openvino', False):
            # Check if already exported
            ov_path = self.model_path.replace('.pt', '_openvino_model')
            if not os.path.exists(ov_path):
                print(f"Exporting {self.model_path} to OpenVINO...")
                model_pt = YOLO(self.model_path)
                model_pt.export(format="openvino")
            self.model_path = ov_path
            print(f"Using OpenVINO model: {self.model_path}")

        self.model = YOLO(self.model_path, task='detect')
        self.start_time_dt = datetime.strptime(config['start_time'], "%H:%M:%S")
        
        # Ensure output dir exists relative to the script execution
        self.output_dir = config['output_dir']
        if not os.path.isabs(self.output_dir):
            # If relative, make it relative to the src folder's parent (step1_detection)
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.output_dir)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_video(self, video_path):
        video_name = os.path.basename(video_path)
        print(f"Processing: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare CSV
        csv_path = os.path.join(self.output_dir, f"{os.path.splitext(video_name)[0]}_detections.csv")
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['video_name', 'frame_id', 'timestamp', 'real_time', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy'])

        # Prepare Video Output if enabled
        out_video = None
        if self.config.get('save_video', False):
            out_video_path = os.path.join(self.output_dir, f"{os.path.splitext(video_name)[0]}_output.mp4")
            out_video = cv2.VideoWriter(
                out_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

        # Run Inference with stream=True for RAM efficiency
        results = self.model.predict(
            source=video_path,
            conf=self.config['conf'],
            classes=self.config['target_classes'],
            imgsz=self.config['imgsz'],
            device=self.config['device'],
            stream=True,
            show=False
        )

        for frame_id, result in enumerate(results):
            timestamp = frame_id / fps
            real_time = (self.start_time_dt + timedelta(seconds=timestamp)).strftime("%H:%M:%S")
            
            # Get annotated frame if saving video
            if out_video:
                annotated_frame = result.plot()
                out_video.write(annotated_frame)

            # Extract detections
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = self.model.names[cls]
                
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                csv_writer.writerow([
                    video_name, frame_id, round(timestamp, 2), real_time,
                    class_name, round(conf, 3), 
                    round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                    round(cx, 1), round(cy, 1)
                ])

        # Cleanup
        csv_file.close()
        if out_video:
            out_video.release()
        cap.release()
        print(f"Finished: {video_name}. Results saved to {csv_path}")

if __name__ == "__main__":
    # Simple test logic
    import yaml
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    processor = VideoProcessor(cfg)
    # Test with the first video found in the directory
    video_files = [f for f in os.listdir(cfg['video_dir']) if f.endswith(('.mp4', '.MP4'))]
    if video_files:
        processor.process_video(os.path.join(cfg['video_dir'], video_files[0]))
