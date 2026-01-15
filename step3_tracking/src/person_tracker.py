import cv2
import os
import torch
import numpy as np
import yaml
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from tqdm import tqdm
from datetime import datetime

class PersonTracker:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = self.config.get('device', 'cpu')
        self.base_dir = os.path.dirname(config_path)
        
        # Paths
        yolo_path = os.path.abspath(os.path.join(self.base_dir, self.config['yolo_model_path']))
        ref_images_list = self.config.get('reference_images', [])
        
        # OpenVINO Optimization
        if self.config.get('use_openvino', False):
            print("Checking OpenVINO export...")
            ov_model_path = yolo_path.replace('.pt', '_openvino_model')
            if not os.path.exists(ov_model_path):
                print(f"Exporting {yolo_path} to OpenVINO...")
                model = YOLO(yolo_path)
                model.export(format='openvino')
            yolo_path = ov_model_path
            print(f"Using OpenVINO model: {yolo_path}")

        # Load YOLOv8 for detection
        self.detector = YOLO(yolo_path, task='detect')
        
        # Load ResNet50 for feature extraction (Re-ID)
        print("Loading Re-ID model (ResNet50)...")
        self.extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.extractor.fc = torch.nn.Identity() # Remove classification layer
        self.extractor.to(self.device)
        self.extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Start time for real-time calculation
        self.start_time_dt = datetime.strptime(self.config.get('start_time', '00:00:00'), "%H:%M:%S")
        
        # Extract embeddings for all reference images
        self.reference_embeddings = []
        if not ref_images_list:
            print("Warning: No reference images provided in config.")
        
        for img_name in ref_images_list:
            img_path = os.path.abspath(os.path.join(self.base_dir, img_name))
            if os.path.exists(img_path):
                emb = self.get_embedding(img_path)
                self.reference_embeddings.append(emb)
                print(f"Reference embedding extracted for: {img_name}")
            else:
                print(f"Warning: Reference image not found: {img_path}")
        
        if not self.reference_embeddings:
            raise ValueError("No valid reference embeddings could be extracted.")

    def get_embedding(self, image_source):
        if isinstance(image_source, str):
            img = Image.open(image_source).convert('RGB')
        else:
            # Assume it's a numpy array (OpenCV format)
            img = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
            
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.extractor(img_t)
        return embedding.cpu().numpy().flatten()

    def track_person(self):
        video_path = os.path.abspath(os.path.join(self.base_dir, self.config['video_path']))
        output_dir = os.path.join(self.base_dir, self.config['output_dir'])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self.config['output_video_name'])
        csv_path = os.path.join(output_dir, self.config.get('csv_output_name', 'tracking_results.csv'))
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Prepare CSV
        import csv
        from datetime import timedelta
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['video_name', 'frame_id', 'timestamp_sec', 'real_time', 'similarity'])

        print(f"Processing video: {os.path.basename(video_path)}")
        pbar = tqdm(total=total_frames, desc="Tracking")
        
        target_count = 0
        video_name = os.path.basename(video_path)
        
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp_sec = frame_id / fps
            real_time = (self.start_time_dt + timedelta(seconds=timestamp_sec)).strftime("%H:%M:%S")
                
            # Detect persons (class 0)
            results = self.detector.predict(
                frame, 
                classes=[0], 
                conf=self.config['detection_conf'], 
                imgsz=self.config['imgsz'],
                verbose=False
            )
            
            best_match = None
            max_sim = -1
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Crop person
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                        
                    # Extract embedding for current detection
                    det_embedding = self.get_embedding(person_crop)
                    
                    # Compare with ALL reference embeddings and take the BEST match
                    current_max_sim = -1
                    for ref_emb in self.reference_embeddings:
                        sim = 1 - cosine(ref_emb, det_embedding)
                        if sim > current_max_sim:
                            current_max_sim = sim
                    
                    if current_max_sim > max_sim:
                        max_sim = current_max_sim
                        best_match = (x1, y1, x2, y2, max_sim)
            
            # If best match is above threshold, annotate and log
            if best_match and max_sim > self.config['similarity_threshold']:
                x1, y1, x2, y2, sim = best_match
                target_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"TARGET ({sim:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                csv_writer.writerow([video_name, frame_id, round(timestamp_sec, 2), real_time, round(sim, 3)])
            
            out.write(frame)
            pbar.update(1)
            frame_id += 1
            
        cap.release()
        out.release()
        csv_file.close()
        pbar.close()
        print(f"\nTracking completed.")
        print(f"Target detected in {target_count} frames.")
        print(f"Output saved to: {output_path}")
        print(f"CSV log saved to: {csv_path}")

if __name__ == "__main__":
    # Path to config file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CONFIG_P = os.path.join(os.path.dirname(script_dir), 'config.yaml')
    
    try:
        tracker = PersonTracker(CONFIG_P)
        tracker.track_person()
    except Exception as e:
        print(f"Error during tracking: {e}")
