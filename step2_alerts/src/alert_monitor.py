import cv2
import os
import csv
import json
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from tqdm import tqdm


class AlertMonitor:
    def __init__(self, config_path, zone_path):
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load zone
        with open(zone_path, 'r') as f:
            self.zone_points = np.array(json.load(f), np.int32)
        
        # Model setup
        self.model_path = self.config['model_path']
        if self.config.get('use_openvino', False):
            ov_path = self.model_path.replace('.pt', '_openvino_model')
            if os.path.exists(ov_path):
                self.model_path = ov_path
        
        self.model = YOLO(self.model_path, task='detect')
        self.start_time_dt = datetime.strptime(self.config['start_time'], "%H:%M:%S")
        
        # Output setup
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(zone_path)), 'results_alerts')
        self.captures_dir = os.path.join(self.output_dir, 'captures')
        os.makedirs(self.captures_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.output_dir, 'alerts_log.csv')
        self.summary_path = os.path.join(self.output_dir, 'alerts_summary.json')
        
        # Tracking state
        self.entered_ids = set()
        self.best_captures = {} # track_id -> {'conf': float, 'frame': ndarray, 'time': str, 'timestamp': float}
        
        self.init_csv()

    def init_csv(self):
        # Always overwrite the CSV to ensure a clean state and correct header
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'real_time', 'track_id', 'confidence', 'capture_path'])

    def is_in_zone(self, cx, cy):
        return cv2.pointPolygonTest(self.zone_points, (cx, cy), False) >= 0


    def process_video(self, video_path):
        video_name = os.path.basename(video_path)
        print(f"\n--- Starting Monitoring: {video_name} ---")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tracking with persistence
        results = self.model.track(
            source=video_path,
            conf=self.config['conf'],
            classes=[0], # person
            imgsz=self.config['imgsz'],
            device=self.config['device'],
            stream=True,
            persist=True,
            tracker="bytetrack.yaml" # ByteTrack is robust for crowd/occlusion
        )

        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing Frames", unit="fr")
        
        for frame_id, result in enumerate(results):
            frame = result.orig_img
            timestamp = frame_id / fps
            real_time = (self.start_time_dt + timedelta(seconds=timestamp)).strftime("%H:%M:%S")
            
            if result.boxes is not None and result.boxes.id is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    track_id = int(box.id[0].item())
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    if self.is_in_zone(cx, cy):
                        # If new ID or better confidence for existing ID
                        is_new = track_id not in self.entered_ids
                        is_better = track_id in self.best_captures and conf > self.best_captures[track_id]['conf']
                        
                        if is_new or is_better:
                            # Draw zone on a copy for the capture
                            annotated_frame = frame.copy()
                            cv2.polylines(annotated_frame, [self.zone_points], True, (0, 255, 0), 2)
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, f"ID:{track_id} {conf:.2f}", (int(x1), int(y1)-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            capture_name = f"id_{track_id}_{real_time.replace(':', '')}.jpg"
                            capture_path = os.path.join(self.captures_dir, capture_name)
                            cv2.imwrite(capture_path, annotated_frame)
                            
                            self.best_captures[track_id] = {
                                'conf': conf,
                                'time': real_time,
                                'timestamp': timestamp,
                                'capture_name': capture_name
                            }
                            
                            if is_new:
                                self.entered_ids.add(track_id)
                                # Log new entry immediately
                                with open(self.csv_path, mode='a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([round(timestamp, 2), real_time, track_id, round(conf, 3), capture_name])

            pbar.update(1)
            pbar.set_postfix({"Entries": len(self.entered_ids)})

        pbar.close()
        
        # Final Summary
        summary = {
            "video_name": video_name,
            "total_unique_entries": len(self.entered_ids),
            "processed_at": datetime.now().isoformat(),
            "status": "Completed"
        }
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        cap.release()
        print(f"\n--- Monitoring Finished ---")
        print(f"Total Unique Entries: {len(self.entered_ids)}")
        print(f"Results saved in: {self.output_dir}")

if __name__ == "__main__":
    config_p = r'c:\Users\DELL\Documents\APEKE\step1_detection\config.yaml'
    zone_p = r'c:\Users\DELL\Documents\APEKE\step2_alerts\zone_coords.json'
    video_p = r'c:\Users\DELL\Documents\APEKE\videos\CAMERA_HALL_PORTE_DROITE.MP4'
    
    monitor = AlertMonitor(config_p, zone_p)
    monitor.process_video(video_p)