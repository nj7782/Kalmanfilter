# Facedetection_with_uniqueID: Using YoloV5+SORT+KalmanFilter for Therapist and Child Detection and Tracking


This project focuses on detecting and tracking people (therapist and child) in video footage using YOLOv5 and the SORT tracking algorithm. The system assigns unique IDs to individuals and tracks them consistently throughout the video, even if they re-enter the frame or are partially occluded.

While the implementation works, **it may have limitations in tracking accuracy**, particularly in scenarios where a person goes out of the frame or after an occlusion. The Kalman filter helps improve tracking but may not always handle re-identification perfectly in complex scenarios.

## Features
- **YOLOv5**: State-of-the-art object detection for identifying people.
- **SORT**: Efficient tracking algorithm for assigning unique IDs.
- **Kalman Filter**: Used to smooth tracking and predict object positions during occlusions.

## Prerequisites

1. **Download the required files**:
   - **YOLOv5** from the official repository: [Download YOLOv5](https://github.com/ultralytics/yolov5)
   - **SORT** for tracking: [Download SORT](https://github.com/abewley/sort)
   - **YOLOv5 Weights**: Pre-trained weights such as `yolov5s.pt` can be used.
     - Download example weights: [yolov5s.pt](https://github.com/ultralytics/yolov5/releases)
   - **Kalman Filter Implementation**: This is integrated into the codebase, no additional setup is required.

2. **Install dependencies**:
   - Make sure to install all required packages by running the following command:
     ```bash
     pip install -r requirements.txt
     ```

## How to Run the Code

1. **Clone or download** this repository.
2. **Update the paths** in the `inference.py` script:
   - Specify paths to YOLOv5, SORT, Kalman Filter, and weights files.
   - Update the video file path for your input video.
3. **Run the script**:
   ```bash
   python inference.py
