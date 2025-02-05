import numpy as np
import os
import sys
import cv2
import yaml
from dataclasses import dataclass
from typing import List, Tuple

min_points = 10
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


def normalize_points(points_2d: np.ndarray, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hartley normalization for numerical stability."""
    # Normalize 2D points
    mean_2d = np.mean(points_2d, axis=0)
    centered_2d = points_2d - mean_2d
    dist_2d = np.sqrt(np.sum(centered_2d**2, axis=1))
    scale_2d = np.sqrt(2) / np.mean(dist_2d)
    T = np.array([
        [scale_2d, 0, -scale_2d * mean_2d[0]],
        [0, scale_2d, -scale_2d * mean_2d[1]],
        [0, 0, 1]
    ])
    
    # Normalize 3D points
    mean_3d = np.mean(points_3d, axis=0)
    centered_3d = points_3d - mean_3d
    dist_3d = np.sqrt(np.sum(centered_3d**2, axis=1))
    scale_3d = np.sqrt(3) / np.mean(dist_3d)
    U = np.array([
        [scale_3d, 0, 0, -scale_3d * mean_3d[0]],
        [0, scale_3d, 0, -scale_3d * mean_3d[1]],
        [0, 0, scale_3d, -scale_3d * mean_3d[2]],
        [0, 0, 0, 1]
    ])
    
    # Apply normalization
    homogeneous_2d = np.hstack([points_2d, np.ones((len(points_2d), 1))])
    normalized_2d = (T @ homogeneous_2d.T).T[:, :2]
    
    homogeneous_3d = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    normalized_3d = (U @ homogeneous_3d.T).T[:, :3]
    
    return normalized_2d, normalized_3d, T, U


def calculate_projection_matrix(cam_points: np.ndarray, lidar_points: np.ndarray) -> np.ndarray:
    """
    Calculate projection matrix using normalized Direct Linear Transform.
    """
    if cam_points.shape[0] < min_points:
        raise CalibrationError(f"At least {min_points} points required")
    
    # Normalize points
    norm_cam, norm_lidar, T, U = normalize_points(cam_points, lidar_points)
    
    # Build DLT matrix with normalized coordinates
    mat = []
    for i in range(norm_cam.shape[0]):
        X, Y, Z = norm_lidar[i]
        u, v = norm_cam[i]
        mat.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u])
        mat.append([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])
        
    # Solve using SVD
    _, _, Vt = np.linalg.svd(np.array(mat))
    P_normalized = Vt[-1].reshape(3, 4)
    
    # Denormalize the projection matrix
    T_inv = np.linalg.inv(T)
    P = T_inv @ P_normalized @ U
    
    return P

def get_lidar_input() -> Tuple[float, float, float]:
    """Get LiDAR coordinates with support for pasted 'x:val' format."""
    while True:
        try:
            # Process x input
            x_line = input('Enter Lidar Point:').strip()
            if ':' in x_line:
                x_str = x_line.split(':', 1)[1].strip()
            else:
                x_str = x_line
            x = float(x_str)
            
            # Process y input
            y_line = input().strip()
            if ':' in y_line:
                y_str = y_line.split(':', 1)[1].strip()
            else:
                y_str = y_line
            y = float(y_str)
            
            # Process z input
            z_line = input().strip()
            if ':' in z_line:
                z_str = z_line.split(':', 1)[1].strip()
            else:
                z_str = z_line
            z = float(z_str)
            
            return (x, y, z)
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def mouse_callback(event: int, x: int, y: int, flags: int,
                  calibration_data: CalibrationPoints) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(calibration_data) >= min_points:
        return
    lidar_coords = get_lidar_input()
    calibration_data.lidar_points.append(lidar_coords)
    calibration_data.camera_points.append((float(x), float(y)))
    print(f'Camera point: {(x, y)} | LiDAR point: {lidar_coords}')
    print(f'Points collected: {len(calibration_data)}/{min_points}')

def collect_calibration_points(window_name: str, camera_id: int) -> CalibrationPoints:
    calibration_data = CalibrationPoints()
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, (2560*2))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise CalibrationError(f"Could not open camera {camera_id}")
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, calibration_data)
    try:
        while len(calibration_data) < min_points:
            ret, frame = cap.read()

            if frame is None:
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
        print("Collecting points for camera 1...")
        cam1_data = collect_calibration_points("Camera 1", 0)
        # cam2_data = collect_calibration_points("Camera 2", 2)
        
        proj_mat1 = calculate_projection_matrix(
            np.array(cam1_data.camera_points),
            np.array(cam1_data.lidar_points)
        )

        # proj_mat2 =  calculate_projection_matrix(
        #     np.array(cam2_data.camera_points),
        #     np.array(cam2_data.lidar_points)
        # )
        
        calibration_data = {
            "point_to_pixel": {
                "ros__parameters": {
                    "projection_matrix": proj_mat1.flatten().tolist(),
                }
            }
        }
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "config/params.yaml")
        with open(config_path, "w") as f:
            yaml.dump(calibration_data, f)
        print(f"Calibration completed. Results saved to {config_path}")
    except (CalibrationError, KeyboardInterrupt) as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()