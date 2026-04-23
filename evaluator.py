import motmetrics as mm
import numpy as np

def evaluate_tracking(ground_truth_file, tracker_output_file):
    """
    Evaluates tracking performance using standard MOT metrics.
    """
    print("Loading Ground Truth and Tracker Output...")
    
    gt = mm.io.loadtxt(ground_truth_file, fmt="mot15-2D", min_confidence=1)
    ts = mm.io.loadtxt(tracker_output_file, fmt="mot15-2D")

   
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'num_switches', 'idf1', 'num_objects', 'num_predictions'], name='DeepSORT')

    print("\n=== FINAL EVALUATION METRICS ===")
    
    strsummary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap={'mota': 'MOTA', 'num_switches': 'ID Switches', 'idf1': 'IDF1 Score'}
    )
    print(strsummary)

if __name__ == "__main__":
   
    GT_PATH = "MOT16/train/MOT16-13/gt/gt.txt"
    TRACKER_PATH = "Evaluations/tracker_output_13.txt"
    
    evaluate_tracking(GT_PATH, TRACKER_PATH)