import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

   
    interArea = max(0, xB - xA) * max(0, yB - yA)

   
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def associate_detections_to_tracks(detections, tracks, iou_threshold=0.3):
    """
    Assigns YOLO detections to existing Kalman filter tracks using the Hungarian Algorithm.
    Kalman filter for predicting movement
    Hungarian algorithm for matching 
    """
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    
    iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(tracks):
            iou_matrix[d, t] = calculate_iou(det, trk)

   
    cost_matrix = 1 - iou_matrix 
    matched_indices = linear_sum_assignment(cost_matrix)
    matched_indices = np.asarray(matched_indices).T

  
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    unmatched_tracks = []
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)

def convert_bbox_to_z(bbox):
    """Converts [x1,y1,x2,y2] to center format [cx, cy, scale, aspect_ratio] for the Kalman Filter."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.
    cy = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([cx, cy, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """Converts center format [cx, cy, scale, aspect_ratio] back to [x1,y1,x2,y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    cx, cy = x[0], x[1]
    return np.array([cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2.]).reshape((1, 4))

class Track:
    """Represents a single tracked object with its own Kalman Filter."""
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # F: State Transition Matrix (Constant Velocity Model)
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
        # H: Measurement Function
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])
        
        self.kf.R[2:,2:] *= 10.     # Measurement noise 
        self.kf.P[4:,4:] *= 1000.   # Initial velocity uncertainty
        self.kf.P *= 10.            # Initial state uncertainty
        self.kf.Q[-1,-1] *= 0.01    # Process noise
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = convert_bbox_to_z(bbox) # Initialize state with first detection
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        """Advances the state vector to the next frame."""
        # Prevent area/scale from going negative
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def update(self, bbox):
        """Updates the state vector with the newly associated detection."""
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(convert_bbox_to_z(bbox))

class Tracker:
    """Manages all active tracks and the data association logic."""
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.track_id_counter = 1

    def update(self, detections):
        """
        Takes new YOLO detections, predicts current track locations, 
        and associates them using your Hungarian Algorithm logic.
        """
        self.frame_count += 1
        
        # Predict new locations for all existing tracks using their Kalman Filters
        predicted_boxes = np.zeros((len(self.tracks), 4))
        for t, trk in enumerate(self.tracks):
            predicted_boxes[t, :] = trk.predict()[0]
            
        # Associate new detections to the predicted tracks
        matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
            detections, predicted_boxes, self.iou_threshold)

        # Update matched tracks with new YOLO detections
        for m in matched:
            track_idx = m[1]
            det_idx = m[0]
            self.tracks[track_idx].update(detections[det_idx])

        for i in unmatched_dets:
            new_track = Track(detections[i], self.track_id_counter)
            self.tracks.append(new_track)
            self.track_id_counter += 1

        ret = []
        for trk in self.tracks:           
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = convert_x_to_bbox(trk.kf.x)[0]
                ret.append(np.concatenate((bbox, [trk.id])).reshape(1,-1))
            
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))