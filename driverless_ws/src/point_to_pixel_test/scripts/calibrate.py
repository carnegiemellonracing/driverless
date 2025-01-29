import numpy as np
import os
import sys
import cv2
import yaml
from dataclasses import dataclass
from typing import List, Tuple


min_points = 6
ct = 0.05

@dataclass
class CalibrationPoints:
    """Store corresponding 2D camera and 3D LiDAR points."""
    camera_points: List[Tuple[float, float]]
    lidar_points: List[Tuple[float, float, float]]

    def __init__(self):
        self.camera_points = []
        self.lidar_points = []

    def __len__(self) -> int:
        return len(self.camera_points)

class CalibrationError(Exception):
    pass

def calculate_projection_matrix(cam_points: np.ndarray, lidar_points: np.ndarray) -> np.ndarray:
    """
    Calculate projection matrix using Direct Linear Transform.
    """
    if cam_points.shape[0] < min_points:
        raise CalibrationError(f"At least {min_points} points required")
    
    mat = []

    for i in range(cam_points.shape[0]):
        X, Y, Z = lidar_points[i]
        u, v = cam_points[i]

        mat.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u])
        mat.append([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])
        
    mat = np.stack(mat)
    
    _, _, Vt = np.linalg.svd(mat)

    return Vt[-1].reshape(3, 4)

def get_lidar_input() -> Tuple[float, float, float]:
    """Get LiDAR coordinates."""
    while True:
        try:
            coords = input('Enter LiDAR coordinates (x y z): ').split()
            if len(coords) != 3:
                raise ValueError("Please enter exactly 3 coordinates")
            return tuple(map(float, coords))
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def mouse_callback(event: int, x: int, y: int, flags: int, 
                  calibration_data: CalibrationPoints) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
        
    if len(calibration_data) >= 6:
        return
        
    lidar_coords = get_lidar_input()
    calibration_data.lidar_points.append(lidar_coords)
    calibration_data.camera_points.append((float(x), float(y)))
    print(f'Points collected: {len(calibration_data)}/{min_points}')

def collect_calibration_points(window_name: str, camera_id: int) -> CalibrationPoints:

    calibration_data = CalibrationPoints()
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise CalibrationError(f"Could not open camera {camera_id}")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, calibration_data)

    try:
        while len(calibration_data) < 6:
            ret, frame = cap.read()
            if not ret:
                raise CalibrationError("Failed to grab frame")
                
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
                
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        
    return calibration_data

def main():
    """Main calibration routine."""
    try:
        # Collect points for both cameras
        print("Collecting points for camera 1...")
        cam1_data = collect_calibration_points("Camera 1", 0)
        
        print("Collecting points for camera 2...")
        cam2_data = collect_calibration_points("Camera 2", 1)

        # Calculate projection matrices
        proj_mat1 = calculate_projection_matrix(
            np.array(cam1_data.camera_points),
            np.array(cam1_data.lidar_points)
        )
        
        proj_mat2 = calculate_projection_matrix(
            np.array(cam2_data.camera_points),
            np.array(cam2_data.lidar_points)
        )

        # Save results in YAML file
        calibration_data = {
            "proj_mat1": proj_mat1.flatten().tolist(),
            "proj_mat2": proj_mat2.flatten().tolist(),
            "confidence_threshold": ct
        }

        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/params.yaml"), "w") as f:
            yaml.dump(calibration_data, f)
            
        print("Calibration completed successfully")
        
    except (CalibrationError, KeyboardInterrupt) as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
