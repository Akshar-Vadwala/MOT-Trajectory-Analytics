import cv2
import time
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import matplotlib.pyplot as plt
from embedder import FeatureExtractor 


def run_detection(video_path, output_path=None, output_txt_path="Evaluations/tracker_output_13.txt"):
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

        if fps == 0 or fps < 1:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    eval_log = open(output_txt_path, "w")
    frame_num = 0

    fps_history = []
    crowd_density_history = []

    print("Starting DeepSORT tracking... Press 'q' to quit.")

    start_time = time.time()

    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1 

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

        active_people = len(track_results)
        total_unique_people = mot_tracker.track_id_counter - 1

        cv2.rectangle(frame, (10, 10), (380, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Active on Screen: {active_people}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Total People Counted: {total_unique_people}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for track in track_results:
            x1, y1, x2, y2 = track[:4].astype(int)
            track_id = int(track[4])
            
            # MOT16 Format: [Frame], [ID], [Box Left (X)], [Box Top (Y)], [Box Width], [Box Height], [Confidence], [3D X], [3D Y], [3D Z]
            w = x2 - x1
            h = y2 - y1
            eval_log.write(f"{frame_num},{track_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("Multi-Object Tracking", frame)

        loop_time = time.time() - loop_start
        current_fps = 1.0 / loop_time if loop_time > 0 else 0
        fps_history.append(current_fps)
        crowd_density_history.append(active_people)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    eval_log.close() 
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_num / total_time
    print(f"Tracking complete. Evaluation log saved to {output_txt_path}")
    print(f"Pipeline Performance: {avg_fps:.2f} Frames Per Second (FPS)")

    print("Generating Analytics Dashboard...")
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(crowd_density_history, color='blue', linewidth=2)
    plt.title('Crowd Density Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Active People on Screen')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(fps_history, color='orange', linewidth=2)
    plt.title('Real-Time Processing Speed (FPS)')
    plt.xlabel('Frame Number')
    plt.ylabel('Frames Per Second')
    plt.axhline(y=avg_fps, color='red', linestyle='--', label=f'Avg FPS: {avg_fps:.2f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    dashboard_path = "Sample_outputs/analytics_dashboard_sample_2.png"
    plt.tight_layout()
    plt.savefig(dashboard_path)
    print(f"Dashboard saved successfully to {dashboard_path}")

if __name__ == "__main__":

    # INPUT_VIDEO = "MOT16/train/MOT16-13/img1/%06d.jpg"  
    INPUT_VIDEO = "Sample/MOT_sample_2.mp4"  
    OUTPUT_VIDEO = "Sample_outputs/MOT_sample_2_output_states.mp4" 

    run_detection(INPUT_VIDEO, OUTPUT_VIDEO)