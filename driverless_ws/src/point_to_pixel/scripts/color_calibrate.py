import os
import sys
import numpy as np
import cv2
import yaml
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.ttk import Notebook
from PIL import Image, ImageTk
class HSVCalibrationUI:
    def __init__(self, frame_path, preview_size=(300, 225), init_display_size=(600, 450)):
        self.root = tk.Tk()
        self.root.title("HSV Color Calibration")
        self.root.minsize(800, 600)  # Set a minimum size for usability
        self.frame_path = frame_path
        self.preview_size = preview_size
        self.init_display_size = init_display_size
        # Load the original image.
        self.orig_image = cv2.imread(self.frame_path)
        if self.orig_image is None:
            print(f"Error: Unable to load image from path {self.frame_path}", file=sys.stderr)
            sys.exit(1)
        # Order for preview panels.
        self.color_order = ["blue", "orange", "yellow"]
        # Default HSV ranges.
        self.defaults = {
            "yellow": {"lh": 20, "ls": 100, "lv": 100, "uh": 30, "us": 255, "uv": 255},
            "blue":   {"lh": 100, "ls": 150, "lv": 0,  "uh": 140, "us": 255, "uv": 255},
            "orange": {"lh": 10, "ls": 100, "lv": 20, "uh": 25,  "us": 255, "uv": 255}
        }
        # Create IntVars for slider values.
        self.color_vars = {}
        for color in self.defaults:
            self.color_vars[color] = {
                key: tk.IntVar(master=self.root, value=val) for key, val in self.defaults[color].items()
            }
        # Dictionary to hold preview image labels.
        self.preview_labels = {}
        self.setup_ui()
    def setup_ui(self):
        # Create a horizontal PanedWindow for resizable left (image/preview) and right (controls) areas.
        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        # Left pane: screenshot and preview panels.
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, minsize=400)
        # Screenshot frame.
        self.screenshot_frame = ttk.Frame(self.left_frame)
        self.screenshot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.screenshot_canvas = tk.Canvas(self.screenshot_frame, bg="black")
        self.screenshot_canvas.pack(fill=tk.BOTH, expand=True)
        self.screenshot_canvas.bind("<Configure>", self.on_screenshot_resize)
        # Preview frame for filtered outputs.
        self.preview_frame = ttk.Frame(self.left_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        for color in self.color_order:
            frame = ttk.Frame(self.preview_frame, width=self.preview_size[0], height=self.preview_size[1])
            frame.pack_propagate(False)
            frame.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.BOTH)
            label_title = ttk.Label(frame, text=f"{color.capitalize()} Preview")
            label_title.pack()
            label_preview = ttk.Label(frame)
            label_preview.pack(expand=True, fill=tk.BOTH)
            self.preview_labels[color] = label_preview
        # Right pane: slider controls and Save button.
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, minsize=300)
        # Create a scrollable canvas for controls.
        self.control_canvas = tk.Canvas(self.right_frame)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.control_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.configure(yscrollcommand=scrollbar.set)
        # Create a frame inside the canvas to hold the controls.
        self.controls_frame = ttk.Frame(self.control_canvas)
        self.controls_window = self.control_canvas.create_window((0, 0), window=self.controls_frame, anchor="n")
        self.control_canvas.bind("<Configure>", self.on_control_canvas_configure)
        # Update the scroll region whenever the contents of the controls frame change.
        self.controls_frame.bind("<Configure>", self.update_scroll_region)
        # Notebook for slider controls.
        self.notebook = Notebook(self.controls_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for color in self.color_order:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"{color.capitalize()} Controls")
            self.create_sliders_for_color(tab, color)
        # Save Calibration button.
        self.save_button = ttk.Button(self.controls_frame, text="Save Calibration", command=self.save_calibration)
        self.save_button.pack(pady=10)
        # Initialize preview panels.
        for color in self.color_order:
            self.update_preview(color)
    def update_scroll_region(self, event=None):
        """Update the scroll region of the canvas to match the size of the controls_frame."""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
    def on_control_canvas_configure(self, event):
        """Center the controls_frame horizontally within the canvas."""
        canvas_width = event.width
        frame_width = self.controls_frame.winfo_reqwidth()
        if frame_width < canvas_width:
            # Center the frame in the canvas if it is smaller than the canvas width.
            x_offset = (canvas_width - frame_width) // 2
            self.control_canvas.itemconfig(self.controls_window, width=canvas_width)
            self.control_canvas.move(self.controls_window, x_offset - self.control_canvas.canvasx(0), 0)
        else:
            # Ensure the controls window uses its natural size if larger than the canvas width.
            self.control_canvas.itemconfig(self.controls_window, width=frame_width)
    def on_screenshot_resize(self, event):
        # Maintain aspect ratio when resizing the screenshot.
        canvas_width = event.width
        canvas_height = event.height
        orig_height, orig_width = self.orig_image.shape[:2]
        aspect_ratio = orig_width / orig_height
        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
        resized = cv2.resize(self.orig_image, (new_width, new_height))
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_rgb)
        self.photo_main = ImageTk.PhotoImage(im_pil)
        self.screenshot_canvas.delete("all")
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        self.screenshot_canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo_main)
        self.screenshot_canvas.image = self.photo_main
    def create_slider(self, parent, label_text, frm, to, var, color):
        scale = tk.Scale(parent, label=label_text, from_=frm, to=to, orient=tk.HORIZONTAL,
                         variable=var, command=lambda event: self.update_preview(color))
        scale.pack(fill=tk.X, padx=5, pady=2)
        return scale
    def create_sliders_for_color(self, parent, color):
        ttk.Label(parent, text="Lower HSV:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.create_slider(parent, "H", 0, 179, self.color_vars[color]["lh"], color)
        self.create_slider(parent, "S", 0, 255, self.color_vars[color]["ls"], color)
        self.create_slider(parent, "V", 0, 255, self.color_vars[color]["lv"], color)
        ttk.Label(parent, text="Upper HSV:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.create_slider(parent, "H", 0, 179, self.color_vars[color]["uh"], color)
        self.create_slider(parent, "S", 0, 255, self.color_vars[color]["us"], color)
        self.create_slider(parent, "V", 0, 255, self.color_vars[color]["uv"], color)
    def update_preview(self, color):
        vars = self.color_vars[color]
        lower = np.array([vars["lh"].get(), vars["ls"].get(), vars["lv"].get()])
        upper = np.array([vars["uh"].get(), vars["us"].get(), vars["uv"].get()])
        hsv = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(self.orig_image, self.orig_image, mask=mask)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(result_rgb).resize(self.preview_size)
        photo = ImageTk.PhotoImage(im_pil)
        if color in self.preview_labels:
            self.preview_labels[color].configure(image=photo)
            self.preview_labels[color].image = photo
    def save_calibration(self):
        calib_data = {}
        for color in self.color_vars:
            vars = self.color_vars[color]
            calib_data[color] = {
                "lower": [vars["lh"].get(), vars["ls"].get(), vars["lv"].get()],
                "upper": [vars["uh"].get(), vars["us"].get(), vars["uv"].get()]
            }
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "config/params.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'r') as file:
                data = yaml.safe_load(file)

                curr = data['/point_to_pixel']['ros__parameters']

                for color in self.color_vars:
                    vars = self.color_vars[color]
                    curr[f"{color}_filter_low"] = [vars["lh"].get(), vars["ls"].get(), vars["lv"].get()]
                    curr[f"{color}_filter_high"] = [vars["uh"].get(), vars["us"].get(), vars["uv"].get()]

            with open(config_path, "w") as f:
                yaml.dump(data, f)
            messagebox.showinfo("Calibration Saved", f"Calibration data saved to {config_path}")
        except Exception as e:

            messagebox.showerror("Error", f"Failed to save calibration: {str(e)}")
    def run(self):
        self.root.mainloop()
def main():
    path = os.path.join(os.path.dirname(os.getcwd()), "driverless_ws/src/point_to_pixel/config/freeze_lr.png")  # Update this path as needed.
    if not os.path.exists(path):
        print(f"Error: Image not found: {path}", file=sys.stderr)
        sys.exit(1)
    ui = HSVCalibrationUI(frame_path=path)
    ui.run()
if __name__ == "__main__":
    main()