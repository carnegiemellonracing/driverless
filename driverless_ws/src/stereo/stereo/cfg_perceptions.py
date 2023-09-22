# Topics
from enum import Enum

# Topics
STEREO_OUT = '/stereo_cones'
LIDAR_OUT = '/lidar_cones'
YOLO_WEIGHT_FILE = 'src/cmrdv_perceptions/stereo_vision/best_april_27.pt'

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# TODO: Update these numbers!!!
ZED_FX = 0
ZED_FY = 0
ZED_CX = 0
ZED_CY = 0

RED_THRESHOLD = 100
class COLORS(Enum):
    BLUE = 1
    YELLOW = 2
    ORANGE = 3
    UNKNOWN = 4
                    
