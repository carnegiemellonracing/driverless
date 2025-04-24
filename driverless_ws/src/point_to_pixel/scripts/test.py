import cv2
import numpy as np
import time
# --- Paths ---
model_path = "/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel/config/yolov5_model_params.onnx"     # Replace with your ONNX model path
image_path = "/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel/config/freeze_ll.png"        # Replace with your test image path
# --- Load image ---
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Failed to load image: {image_path}")
    exit(1)
height, width = image.shape[:2]

# --- Load ONNX model ---
net = cv2.dnn.readNetFromONNX(model_path)

# --- Prepare input ---
input_size = 640
blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
net.setInput(blob)

# --- Measure inference time ---
start_time = time.time()

# --- Run inference ---
outputs = net.forward()[0]  # shape: [1, num_detections, 85] for YOLOv5

end_time = time.time()
inference_time = end_time - start_time

# --- Postprocess and draw boxes ---
conf_threshold = 0.25

x_scale = width / input_size
y_scale = height / input_size

# Initialize a list for detections
detections = 0

for detection in outputs:
    object_conf = detection[4]
    if object_conf < conf_threshold:
        continue

    cx, cy, w, h = detection[:4]
    x = int((cx - w / 2) * x_scale)
    y = int((cy - h / 2) * y_scale)
    w = int(w * x_scale)
    h = int(h * y_scale)

    # Draw the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    detections += 1

# --- Show the result ---
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Print inference time ---
print(f"Inference time: {inference_time:.4f} seconds")
print(f"Detected {detections} objects.")

