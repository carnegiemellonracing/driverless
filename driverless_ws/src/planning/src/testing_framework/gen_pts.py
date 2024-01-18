import cv2
import numpy as np

# List to store clicked points
clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add clicked point to the list
        clicked_points.append((x, y))
        # Display the point on the image
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Interactive Plot', img)

# Create a black image (you can use an existing plot image)
img = np.zeros((512, 512, 3), np.uint8)
cv2.imshow('Interactive Plot', img)

# Set the callback function for mouse events
cv2.setMouseCallback('Interactive Plot', click_event)

# Wait for the user to click (press 'q' to exit)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Export clicked points to a file (e.g., CSV)
np.savetxt('clicked_points.csv', clicked_points, delimiter=',')

cv2.destroyAllWindows()
