import cv2
import numpy as np
import torch
import sys
from sort import Sort
from scipy.spatial.distance import cdist
from kalmanfilter import KalmanFilter

# Paths Configuration
yolov5_path = r"C:\Users\jitin\us\yolov5"
sort_path = r"C:\Users\jitin\us\sort"  # Path where SORT is located
weights_path = r"C:\Users\jitin\Downloads\yolov5s.pt"
video_path = r"C:\nj\ABA Therapy - Play.mp4"

# Append SORT library to system path
sys.path.append(sort_path)

# Initialize SORT tracker
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.4)

# Initialize Kalman filter
dim_x = 4  # State vector dimension (x, y, vx, vy)
dim_z = 2  # Measurement vector dimension (x, y)
F = np.array([[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])  # State transition matrix
H = np.array([[1, 0, 0, 0],
             [0, 1, 0, 0]])  # Observation matrix
Q = np.diag([0.1, 0.1, 0.01, 0.01])  # Process noise covariance
R = np.diag([0.02, 0.02])  # Measurement noise covariance
kf = KalmanFilter(dim_x, dim_z, F, H, Q, R)

# Load YOLOv5 model
model = torch.hub.load(yolov5_path, 'custom', path=weights_path, source='local')
model.eval()

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    sys.exit()

# Track history for visualization
track_history = {}

# Function to calculate visual similarity between two bounding boxes
def calculate_similarity(bbox1, bbox2):
    # Calculate IoU
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    inter_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    box2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Processing the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get detections from YOLO model
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Filter out low-confidence detections
    detections = [det for det in detections if det[4] > 0.5]
    sort_detections = np.array([[x1, y1, x2, y2, score] for (x1, y1, x2, y2, score, cls) in detections])

    # Update SORT tracker
    tracked_objects = tracker.update(sort_detections)

    # Update Kalman filter for each tracked object
    for track in tracked_objects:
        track_id = int(track[4])
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])

        # Draw bounding box and ID on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Kalman prediction and update step
        if track_id in track_history:
            prev_bbox = track_history[track_id][-1]
            similarity = calculate_similarity(prev_bbox, (x1, y1, x2, y2))

            if similarity < 0.5:  # Lost track
                # Re-associate the track based on visual similarity
                for det in sort_detections:
                    det_x1, det_y1, det_x2, det_y2, _ = det
                    det_bbox = (det_x1, det_y1, det_x2, det_y2)
                    new_similarity = calculate_similarity(det_bbox, (x1, y1, x2, y2))
                    if new_similarity > 0.5:  # Adjust threshold as needed
                        # Re-associate the track
                        kf.x = np.array([det_x1, det_y1, 0, 0])
                        break

        # Save the track to history
        track_history[track_id] = track_history.get(track_id, []) + [(x1, y1, x2, y2)]

    # Display the resulting frame
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
