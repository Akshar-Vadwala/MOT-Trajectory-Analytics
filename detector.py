import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
from embedder import FeatureExtractor 

def run_detection(video_path, output_path=None):
    model = YOLO('yolov8m.pt') 
    
    mot_tracker = Tracker(max_age=90, min_hits=3, iou_threshold=0.15, lambda_weight=0.5)
    embedder = FeatureExtractor() 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    writer = None
    if output_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Starting DeepSORT tracking... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True, verbose=False)
        
        detections = []
        features = [] 

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if cls_id == 0 and conf > 0.3: 
                    bbox = [x1, y1, x2, y2]
                    detections.append(bbox)
                    
                    feat = embedder.extract(frame, bbox)
                    features.append(feat)

        detections = np.array(detections)
        features = np.array(features)
        
        if len(detections) == 0:
            detections = np.empty((0, 4))
            features = np.empty((0, 512))
            
        track_results = mot_tracker.update(detections, features)

        for track in track_results:
            x1, y1, x2, y2 = track[:4].astype(int)
            track_id = int(track[4])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("Multi-Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    INPUT_VIDEO = "MOT/MOT_sample_2.mp4"  
    OUTPUT_VIDEO = "MOT_outputs/MOT_sample_2_output.mp4" 

    run_detection(INPUT_VIDEO, OUTPUT_VIDEO)