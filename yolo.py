import cv2
from ultralytics import YOLO

def run_detection(video_path, output_path=None):
    """
    Reads a video, runs YOLO object detection frame-by-frame, 
    and draws bounding boxes.
    """

    model = YOLO('yolov8m.pt') 

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

    print("Starting detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        results = model(frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if cls_id == 0 and conf > 0.5: 
                  
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if writer:
            writer.write(frame)

        cv2.imshow("YOLO Detection Baseline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    INPUT_VIDEO = "MOT/MOT_sample_2.mp4"  
    OUTPUT_VIDEO = "YOLO_outputs/yolo_sample_output_2.mp4" 
    
    run_detection(INPUT_VIDEO, OUTPUT_VIDEO)