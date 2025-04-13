import numpy as np
import os
import sys
import cv2
import yaml
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pyperclip
import argparse

@dataclass
class CalibrationPoint:
    camera_x: float
    camera_y: float
    lidar_x: float
    lidar_y: float
    lidar_z: float
    index: int

    def as_tuple(self) -> Tuple[Tuple[float, float], Tuple[float, float, float]]:
        return ((self.camera_x, self.camera_y), (self.lidar_x, self.lidar_y, self.lidar_z))

class CalibrationData:
    def __init__(self):
        self.points: List[CalibrationPoint] = []
        self._next_index = 1

    def add_point(self, camera_point: Tuple[float, float], lidar_point: Tuple[float, float, float]) -> int:
        point = CalibrationPoint(
            camera_x=camera_point[0],
            camera_y=camera_point[1],
            lidar_x=lidar_point[0],
            lidar_y=lidar_point[1],
            lidar_z=lidar_point[2],
            index=self._next_index
        )
        self.points.append(point)
        self._next_index += 1
        return point.index

    def remove_point(self, index: int) -> None:
        self.points = [p for p in self.points if p.index != index]

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.points:
            return np.array([]), np.array([])
        camera_points = np.array([(p.camera_x, p.camera_y) for p in self.points])
        lidar_points = np.array([(p.lidar_x, p.lidar_y, p.lidar_z) for p in self.points])
        return camera_points, lidar_points

    def __len__(self) -> int:
        return len(self.points)

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

def calculate_projection_matrix(cam_points: np.ndarray, lidar_points: np.ndarray, min_points: int = 10) -> np.ndarray:
    """Calculate projection matrix using normalized Direct Linear Transform."""
    if cam_points.shape[0] < min_points:
        raise ValueError(f"At least {min_points} points required")
    
    # # Normalize points
    # norm_cam, norm_lidar, T, U = normalize_points(cam_points, lidar_points)
    
    # Build DLT matrix with normalized coordinates
    mat = []
    for i in range(lidar_points.shape[0]):
        X, Y, Z = lidar_points[i]
        u, v = cam_points[i]
        mat.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u])
        mat.append([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(np.array(mat))
    P = Vt[-1].reshape(3, 4)
    # P_normalized = Vt[-1].reshape(3, 4)
    
    # Denormalize the projection matrix
    # T_inv = np.linalg.inv(T)
    # P = T_inv @ P_normalized @ U
    
    return P

class LidarInputDialog:
    def __init__(self, parent, initial_coords: Optional[Tuple[float, float, float]] = None):
        self.top = tk.Toplevel(parent)
        self.top.title("Enter LiDAR Coordinates")
        self.result = None
        
        # Create main frame
        main_frame = ttk.Frame(self.top, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Manual Input Section
        manual_frame = ttk.LabelFrame(main_frame, text="Manual Input", padding="5")
        manual_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(manual_frame, text="X:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(manual_frame, text="Y:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Label(manual_frame, text="Z:").grid(row=2, column=0, padx=5, pady=5)
        
        self.x_var = tk.StringVar(value=str(initial_coords[0]) if initial_coords else "")
        self.y_var = tk.StringVar(value=str(initial_coords[1]) if initial_coords else "")
        self.z_var = tk.StringVar(value=str(initial_coords[2]) if initial_coords else "")
        
        ttk.Entry(manual_frame, textvariable=self.x_var).grid(row=0, column=1, padx=5, pady=5)
        ttk.Entry(manual_frame, textvariable=self.y_var).grid(row=1, column=1, padx=5, pady=5)
        ttk.Entry(manual_frame, textvariable=self.z_var).grid(row=2, column=1, padx=5, pady=5)
        
        # Paste Section
        paste_frame = ttk.LabelFrame(main_frame, text="Paste Coordinates", padding="5")
        paste_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        paste_btn = ttk.Button(paste_frame, text="Paste from Clipboard", command=self.paste_from_clipboard)
        paste_btn.grid(row=2, column=0, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(btn_frame, text="OK", command=self.ok).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.cancel).grid(row=0, column=1, padx=5)
        
        # Make dialog modal
        self.top.transient(parent)
        self.top.wait_visibility()
        self.top.grab_set()
        
    def paste_from_clipboard(self):
        """Handle pasting coordinates from clipboard."""
        try:
            text = pyperclip.paste()
            # Try different formats
            if ':' in text:  # Format: x:val y:val z:val
                coords = {}
                for line in text.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        coords[key.strip().lower()] = value.strip()
                if all(k in coords for k in ['x', 'y', 'z']):
                    self.x_var.set(coords['x'])
                    self.y_var.set(coords['y'])
                    self.z_var.set(coords['z'])
                    return
            else:  # Try comma/space separated format
                values = [float(x.strip()) for x in text.replace(',', ' ').split()]
                if len(values) >= 3:
                    self.x_var.set(str(values[0]))
                    self.y_var.set(str(values[1]))
                    self.z_var.set(str(values[2]))
                    return
            raise ValueError("Invalid format")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not parse clipboard content: {str(e)}")
    
    def ok(self):
        """Validate and accept the input."""
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            self.result = (x, y, z)
            self.top.destroy()
        except ValueError:
            tk.messagebox.showerror("Error", "Please enter valid numbers")
    
    def cancel(self):
        """Cancel the input."""
        self.top.destroy()

class CalibrationUI:
    def __init__(self, min_points: int = 10, width=800, height=450, frame_path: str = "", cam: str = ""):
        self.min_points = min_points
        self.calibration_data = CalibrationData()
        self.current_frame = None
        self.dragging_point_index: Optional[int] = None  
        self.width = width
        self.height = height
        self.frame_path = frame_path  # Path to the image you want to use as the frame
        self.scaling_factor = 1280 /1920
        self.cam = cam
        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Camera Calibration")
        
        # Main layout
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Camera view
        self.canvas = tk.Canvas(self.left_frame, width=self.width, height=self.height)
        self.canvas.pack()
        # Bind for dragging and new point creation
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # F for finish!
        self.root.bind('<f>', self.finish_calibration_with_key)

        # Point list
        self.point_frame = ttk.LabelFrame(self.right_frame, text="Calibration Points")
        self.point_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("Camera X", "Camera Y", "Lidar X", "Lidar Y", "Lidar Z")
        self.point_list = ttk.Treeview(self.point_frame, columns=columns, show="headings")
        for col in columns:
            self.point_list.heading(col, text=col)
            self.point_list.column(col, width=100)
        self.point_list.pack(fill=tk.BOTH, expand=True)
        self.point_list.bind("<Double-1>", self.on_treeview_double_click)

        # Scrollbar for point list
        scrollbar = ttk.Scrollbar(self.point_frame, orient=tk.VERTICAL, command=self.point_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.point_list.configure(yscrollcommand=scrollbar.set)

        # Controls
        self.control_frame = ttk.Frame(self.right_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.remove_btn = ttk.Button(self.control_frame, text="Remove Point", command=self.remove_selected_point)
        self.remove_btn.pack(side=tk.LEFT, padx=5)

        self.finish_btn = ttk.Button(self.control_frame, text="Finish Calibration", command=self.finish_calibration)
        self.finish_btn.pack(side=tk.RIGHT, padx=5)

        # Status
        self.status_var = tk.StringVar()
        self.status_var.set(f"Collected 0/{self.min_points} points")
        self.status_label = ttk.Label(self.right_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)

    def finish_calibration_with_key(self, event=None):
        self.finish_calibration()
    
    def upsize(self, n):
        return n / self.scaling_factor

    def downsize(self, n):
        return n * self.scaling_factor

    def update_camera_view(self):
        # Load the static image from the specified path
        frame = cv2.imread(self.frame_path)
        if frame is not None:
            # Convert OpenCV BGR to RGB for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            # Resize the image to fit the canvas instead of cropping it
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            image_pil = image_pil.resize((canvas_width, canvas_height))
            self.current_frame = ImageTk.PhotoImage(image_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_frame)
            
            # Draw existing points (coordinates may need adjusting if they are relative to image size)
            for i in range(len(self.calibration_data.points)):
                point = self.calibration_data.points[i]
                self.canvas.create_oval( self.downsize(point.camera_x) - 2, self.downsize(point.camera_y)- 2, self.downsize(point.camera_x) +2, self.downsize(point.camera_y)+ 2, fill='red')
                self.canvas.create_text(self.downsize(point.camera_x) + 6, self.downsize(point.camera_y) + 6, text=str(i + 1), fill='red')
        else:
            print(f"Error: Unable to load image at {self.frame_path}")
        self.root.after(30, self.update_camera_view)

    def on_canvas_press(self, event):
        click_point = (self.upsize(event.x), self.upsize(event.y))
        # Check if click is near an existing camera point 
        for point in self.calibration_data.points:
            dist = ((point.camera_x - click_point[0]) ** 2 + (point.camera_y - click_point[1]) ** 2) ** 0.5
            if dist <= 10:
                self.dragging_point_index = point.index
                return
        # Otherwise, add a new point
        self.add_new_point(click_point)

    def on_canvas_drag(self, event):
        if self.dragging_point_index is not None:
            # Update camera coordinates of the dragged point
            for point in self.calibration_data.points:
                if point.index == self.dragging_point_index:
                    point.camera_x = self.upsize(float(event.x))
                    point.camera_y = self.upsize(float(event.y))
                    break
            self.update_point_list()

    def on_canvas_release(self, event):
        self.dragging_point_index = None

    def add_new_point(self, camera_point: Tuple[float, float]):
        # Get LiDAR coordinates through dialog
        lidar_dialog = LidarInputDialog(self.root)
        self.root.wait_window(lidar_dialog.top)
        
        if lidar_dialog.result:
            self.calibration_data.add_point(camera_point, lidar_dialog.result)
            self.update_point_list()
            self.status_var.set(f"Collected {len(self.calibration_data)}/{self.min_points} points")

    def on_treeview_double_click(self, event):
        item = self.point_list.identify_row(event.y)
        if item:
            index = int(self.point_list.item(item)["text"])
            # Find the corresponding calibration point
            for point in self.calibration_data.points:
                if point.index == index:
                    selected_point = point
                    break
            else:
                return
            # Open dialog with current LiDAR values
            dialog = LidarInputDialog(self.root, initial_coords=(
                selected_point.lidar_x,
                selected_point.lidar_y,
                selected_point.lidar_z,
            ))
            self.root.wait_window(dialog.top)
            if dialog.result:
                selected_point.lidar_x, selected_point.lidar_y, selected_point.lidar_z = dialog.result
                self.update_point_list()

    def update_point_list(self):
        self.point_list.delete(*self.point_list.get_children())
        for point in self.calibration_data.points:
            self.point_list.insert("", "end", text=point.index, 
                                 values=(f"{point.camera_x:.1f}", f"{point.camera_y:.1f}",
                                        f"{point.lidar_x:.3f}", f"{point.lidar_y:.3f}", 
                                        f"{point.lidar_z:.3f}"))

    def remove_selected_point(self):
        selected = self.point_list.selection()
        if selected:
            index = int(self.point_list.item(selected[0])["text"])
            self.calibration_data.remove_point(index)
            self.update_point_list()
            self.status_var.set(f"Collected {len(self.calibration_data)}/{self.min_points} points")

    def finish_calibration(self):
        if len(self.calibration_data) < self.min_points:
            tk.messagebox.showerror("Error", f"Need at least {self.min_points} points for calibration")
            return

        camera_points, lidar_points = self.calibration_data.get_arrays()

        for i in range(len(camera_points)):
            print (camera_points[i], lidar_points[i])


        try:
            proj_matrix = calculate_projection_matrix(camera_points, lidar_points, self.min_points)
            self.save_calibration(proj_matrix)
            tk.messagebox.showinfo("Success", "Calibration completed successfully!")
            self.root.quit()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Calibration failed: {str(e)}")

    def save_calibration(self, proj_matrix): 
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "config/params.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'r') as file:
                data = yaml.safe_load(file)

                curr = data['/point_to_pixel']['ros__parameters']

                curr[f"projection_matrix_{self.cam}"] = [proj_matrix.flatten().tolist()]

            with open(config_path, "w") as f:
                yaml.dump(data, f)
                print("Calibration Saved", f"Calibration data saved to {config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save calibration: {str(e)}")

    def run(self):
        self.update_camera_view()
        self.root.mainloop()



def main():
    # # Set up argument parser
    parser = argparse.ArgumentParser(description="Specify which camera you're calibrating for (specify ll, lr, rr, or rl).")
    parser.add_argument("-c", "--camera", required=True, help="camera (specify ll, lr, rr, or rl)", type=str)
    args = parser.parse_args()

    print()

    if args.camera not in {"ll", "lr", "rr", "rl"}:
        print(f"Invalid args, specify ll, lr, rr, or rl")
        sys.exit(1)

    cam = args.camera

    # Read the image from the provided file path
    path = f"/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel/config/freeze_{cam}.png"
    im = cv2.imread(path)

    if im is None:
        print(f"Error: Unable to load image from path {path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize CalibrationUI with the image size and file path
        print(im.shape)
        ui = CalibrationUI(width=1280, height=720, frame_path=path, cam=cam)
        ui.run()
    except Exception as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()