import torch
from pathlib import Path

# --- Paths ---
pt_model_path = "/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel/config/yolov5_model_params.pt"  # Path to your YOLOv5 .pt model
onnx_output_path = "/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel/config/yolov5_model_params.onnx"  # Desired path for the .onnx model

# --- Load the YOLOv5 model ---
model = torch.hub.load("ultralytics/yolov5", "custom", pt_model_path)# Load the model
model = model.to("cuda")
model.eval()  # Set to evaluation mode

# --- Export the model to ONNX ---
dummy_input = torch.zeros(1, 3, 640, 640)  # Dummy input tensor of shape (batch_size, channels, height, width)

# Export the model (you can specify input and output names, and more if needed)
torch.onnx.export(model, dummy_input, onnx_output_path, input_names=['images'], output_names=['output'], opset_version=12)

print(f"Model successfully converted to {onnx_output_path}")