"""
Interface classes for ZED

The following classes (ZEDSDK and ZEDCV2) enable users to create and interact with instances
of the ZED stereo camera. The following implementations handle querying the camera for images 
and depth maps, and calculating the depth of objects given an input bounding box. 

ZEDSDK uses the api provided by the camera's manufacturers while ZEDCV2 utilizes OpenCV's 
builtin video capture feature.
"""
# from fsdv.perceptions.lidar.transform import WorldImageTransformer
# import fsdv.perceptions.stereo_vision.utils as utils
# from fsdv.perceptions.stereo_vision.DataCollection import DataCollection
from abc import ABC, abstractmethod
import pyzed.sl as sl
import cv2
from enum import Enum
import numpy as np
import statistics
import signal
import sys



class ZEDException(Exception):
    pass

class ZEDSDK():
    def __init__(self,
                 camera_resolution=sl.RESOLUTION.VGA,
                 depth_mode=sl.DEPTH_MODE.ULTRA,
                 coordinate_units=sl.UNIT.METER,
                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP,
                 camera_fps=0,
                 serial_num=None,
                 **kwargs
                 ) -> None:
        """
        ZED interface class that utilizes the ZED SDK
        Args:
            camera_resolution (sl.RESOLUTION): Sets camera resolution
                                            Default = sl.RESOLUTION.HD720
                                            Other Possible Inputs = sl.RESOLUTION.HD1080 
                                                                    sl.RESOLUTION.HD2K
                                                                    sl.RESOLUTION.VGA
            depth_mode (self.DEPTH_MODE): Sets how depth map is captured
                                        Default = sl.DEPTH_MODE.ULTRA
                                        Other Possible Inputs = sl.DEPTH_MODE.QUALITY
                                                                sl.DEPTH_MODE.PERFORMANCE
            coordinate_units (self.UNIT): Sets the units of all measurements
                                        Default = sl.UNIT.METER
                                        Other Possible Inputs = sl.UNIT.MILLIMETER
                                                                sl.UNIT.CENTIMETER
                                                                sl.UNIT.INCH
                                                                sl.UNIT.FOOT
            coordinate_system (self.COORDINATE_SYSTEM): Sets the coordinate system by which measurements are collected
                                                        Refer to the following link for helpful illustrations: https://www.stereolabs.com/docs/api/group__Core__group.html#ga1114207cbac18c1c0e4f77a6b36a8cb2
                                                        Default = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
                                                        Other Possible Inputs = sl.COORDINATE_SYSTEM.IMAGE
                                                                                sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
                                                                                sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
                                                                                sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
                                                                                sl.COORDINATE_SYSTEM.LEFT_HANDED_Z_UP
            camera_fps (int): Sets the camera frames per second.
                            If the requested camera_fps is unsupported, the closest available FPS will be used. 
                            Default: 0 (highest FPS of the specified camera_resolution will be used)
            **kwargs: MUST BE VALID CAMERA PARAMETER - refer to following link for list of all possible parameters
                    https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html#a8aebd3c5eea7c24cfa15a96cbb2ec8aa
        """
        self.init_params = sl.InitParameters(camera_resolution=camera_resolution,
                                             depth_mode=depth_mode,
                                             coordinate_units=coordinate_units,
                                             coordinate_system=coordinate_system,
                                             camera_fps=camera_fps,
                                             **kwargs
                                             )
        
        if serial_num is not None:
            self.init_params.set_from_serial_number(serial_num)

        # ZED Camera object
        self.zed = sl.Camera()

        # Matrices to store left, right images + depth map + point cloud
        self.left_image_mat = sl.Mat()
        self.right_image_mat = sl.Mat()
        self.depth_map_mat = sl.Mat()
        self.point_cloud_mat = sl.Mat()

        # Runtime parameters to be applied to each frame that is grabbed
        # Can be changed via set_runtime_params function
        self.set_runtime_params()
        

        # create a sigint handler to close camera if SIGINT ever called
        def close_handler(signalnum, stack_frame):
            # sys.exit will call destructor for any camera object which will close it
            sys.exit()
        signal.signal(signal.SIGINT, close_handler)

    def open(self) -> None:
        """
        Open the ZED camera
        """
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.opened = False
            print(repr(status))
            print("Possible reasons for failure to open camera:")
            print("1) The ZED is not plugged into the Jetson")
            print("2) A previous call was made to open the ZED")
            print("3) A previous python process did not terminate properly, thereby leaving the ZED opened. Restart the Jetson and try again")
            raise ZEDException
        self.opened = True
        print("ZED Opened!")

    def close(self):
        '''it is ok to call zed.close() multiple times -- even without opening'''

        # note: SIGINT closes the process, SIGQUIT does, SIGTSTP DOES NOT
        # if you stop program via SIGTSTP, process won't be killed like others
        # NOTE: SIGTSTP is done via CTRL-D
        # need to kill using $ kill -9 [pid], then can re-open camera
        # -9 flag is important to send the KILL signal (ensures process killed)
        # NOTE: we could avoid this issue by having a signal handler for SIGTSTP
        # however, this closes the process instead of stopping it which is unintended
        # so for now, we will just manually call $ kill -9 [pid] until we can figure
        # out how to catch the SIGTSTP, close the camera, and then stop the process
        # NOTE: if the process is killed, then the camera is automatically closed without
        # this function being called
        self.opened = False
        print("ZED Closed!")

    def __del__(self):
        self.close()

    def set_runtime_params(self,
                           sensing_mode=None,#sl.SENSING_MODE.STANDARD,
                           measure3D_reference_frame=None,#sl.REFERENCE_FRAME.CAMERA,
                           enable_depth=True,
                           confidence_threshold=100,
                           texture_confidence_threshold=100,
                           remove_saturated_areas=True
                           ) -> None:
        """
        Change the runtime parameters of the ZED camera. This applies to each frame that is grabbed
        Args:
            sensing_mode (sl.SENSING_MODE): Depth sensing mode.
                        Default = sl.SENSING_MODE.STANDARD (this is the suggested mode for autonomous vehicles)
                        Other Possible Inputs = sl.SENSING_MODE.FILL
            measure3D_reference_frame (sl.REFERENCE_FRAME): Defines which type of position matrix is used to store camera path and pose.
                                                            Default = sl.REFERENCE_FRAME.CAMERA
                                                            Other Possible Inputs = sl.REFERENCE_FRAME.WORLD
            enable_depth (bool): Boolean to choose whether depth data is computed
                                Default = True
            confidence_threshold (int): Threshold to reject depth values based on their confidence.
                                        Each depth pixel has a corresponding confidence. (MEASURE.CONFIDENCE), the confidence range is [1,100].
                                        By default, the confidence threshold is set at 100, meaning that no depth pixel will be rejected.
                                        Decreasing this value will remove depth data from both objects edges and low textured areas, to keep only confident depth estimation data. Pixels with a value close to 100 are not to be trusted. Accurate depth pixels tends to be closer to lower values. It can be seen as a probability of error, scaled to 100. 
                                        Default = 100
            texture_confidence_threshold (int): Threshold to reject depth values based on their texture confidence.
                                                The texture confidence range is [1,100].
                                                By default, the texture confidence threshold is set at 100, meaning that no depth pixel will be rejected.
                                                Decreasing this value will remove depth data from image areas which are uniform. Pixels with a value close to 100 are not to be trusted. Accurate depth pixels tends to be closer to lower values. 
                                                Default = 100
            remove_saturated_areas (bool): Defines if the saturated area (Luminance>=255) must be removed from depth map estimation
                                        It is recommended to keep this parameter at true because saturated area can create false detection.
                                        Default = True
        """
        self.runtime_params = sl.RuntimeParameters(#sensing_mode=sensing_mode,
        #                                            measure3D_reference_frame=measure3D_reference_frame,
        #                                            enable_depth=enable_depth,
        #                                            confidence_threshold=confidence_threshold,
        #                                            texture_confidence_threshold=texture_confidence_threshold,
        #                                            remove_saturated_areas=remove_saturated_areas
                                                   )

    def grab_data(self, view=False):
        """
        # TODO: change the default coordinate system that we are using for ZED -- use sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
        #       note: that will require changing the world to image and image to world functions but will make lives easier

        Retrieves the most up-to-date camera frame and stores them in the appropriate sl.Mat object
        Timing information: there’s a warm-up kind of speed, if we query really quickly initially and bring the timing down to a 10s of ms, then every 30ms, we can query for ~15ms
        minor note: depth map and point cloud correspond in NaN values for pixels that distance is not determined
        major note: np.ndarrays that are returned are shallow copies of the data, and as a result, 
        further calls to grab_data() will cause previously existing variables containing the return values of this function to update as well! -- may need to create deep copies
        note: image values update every 33ms as the camera is 30FPS
        Args:
            view (bool): If True, returns normalized depth_map that allows for easy viewing
                         If False, returns depth_map that has absolute depths
        Returns:
            self.left_image_np (np.ndarray): Numpy array containing left image pixels
            self.right_image_np (np.ndarray): Numpy array containing right image pixels
            self.depth_map_np (np.ndarray): Numpy array containing depth map distances
            self.point_cloud_np (np.ndarray): Numpy array containing point cloud data
        """
        errno = self.zed.grab(self.runtime_params)
        if errno == sl.ERROR_CODE.SUCCESS:
            # A new image is available only if grab() returns SUCCESS
            self.zed.retrieve_image(self.left_image_mat, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.right_image_mat, sl.VIEW.RIGHT)

            # Only get depth if enable_depth is True
            if self.runtime_params.enable_depth:
                if view:
                    self.zed.retrieve_image(self.depth_map_mat, sl.VIEW.DEPTH)
                else:
                    self.zed.retrieve_measure(self.depth_map_mat, sl.MEASURE.DEPTH)

                # get the point cloud
                self.zed.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZRGBA)

            # Convert to numpy array
            self.left_image_np = self.left_image_mat.get_data()
            self.right_image_np = self.right_image_mat.get_data()
            self.depth_map_np = self.depth_map_mat.get_data() if self.runtime_params.enable_depth else None
            self.point_cloud_np = self.point_cloud_mat.get_data() if self.runtime_params.enable_depth else None
            return self.left_image_np, self.right_image_np, self.depth_map_np, self.point_cloud_np
        else:
            print(errno)
            print("Camera was unable to grab a new frame")
            raise ZEDException

    def get_object_depth(self, bounding_box, padding=2):
        """
        Calculates the median z position of supplied bounding box
        Args:
            bounding_box: Box containing object of interest
                          bound[0] = x1 (top left - x)
                          bound[1] = y1 (top left - y)
                          bound[2] = x2 (bottom right - x)
                          bound[3] = y2 (bottom right - y)
            padding: Num pixels around center to include in depth calculation
                           REQUIREMENT: padding <= (x2-x1)/2 and padding <= (y2-y1)/2 
                           Default = 2
        Returns:
            float - Depth estimation of object in interest 
        """
        z_vect = []
        x1, y1 = bounding_box[0], bounding_box[1]
        x2, y2 = bounding_box[2], bounding_box[3]
        if padding > (x2-x1)/2 or padding > (y2-y1)/2:
            print("REQUIREMENT: padding <= (x2-x1)/2 and padding <= (y2-y1)/2")
            raise ZEDException
        center_y = int((x1+x2)/2)  # x-coordinate of bounding box center
        center_x = int((y1+y2)/2)  # y-coordinate of bounding box center

        # Iterate through center pixels and append them to z_vect
        for i in range(max(center_x - padding, 0), min(center_x + padding, 720)):
            for j in range(max(center_y - padding, 0), min(center_y + padding, 1280)):
                z = self.depth_map_np[i, j]
                if not np.isnan(z) and not np.isinf(z):
                    z_vect.append(z)
        
        # Try calculating the mean of the depth of the center pixels
        # Catch all exceptions and return -1 if mean is unable to be calculated
        try:
            z_mean = statistics.mean(z_vect)
        except Exception:
            print("Unable to compute z_mean")
            z_mean = -1
        return z_mean

    def get_cone_color(self, box, padding = 2):
        """
        Calculates the cone color based on supplied bounding box
        Args:
            bounding_box: Box containing object of interest
                          bound[0] = x1 (top left - x)
                          bound[1] = y1 (top left - y)
                          bound[2] = x2 (bottom right - x)
                          bound[3] = y2 (bottom right - y)
            padding: Num pixels around center to include in depth calculation
                           REQUIREMENT: padding <= (x2-x1)/2 and padding <= (y2-y1)/2 
                           Default = 2
        Returns: 
            config.COLORS: color of bounding box
            
        """
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

        min_x = max(center_x - padding, 0)
        max_x = min(center_x + padding, self.init_params.camera_resolution.width)

        min_y = max(center_y - padding, 0)
        max_y = min(center_y + padding, self.init_params.camera_resolution.width)

        rgbs = self.left_image_np[min_x:max_x, min_y:max_y, :].reshape(-1,3)
        avg_rgbs = np.average(rgbs, axis=0)
        if avg_rgbs[0] < avg_rgbs[1]:
            return config.COLORS.BLUE
        else:
            return config.COLORS.YELLOW

    def get_zed(self):
        return self.zed

    def get_calibration_params(self):
        return self.calibration_parameters

    def world_to_image(self, points):
        '''
            converts 3D world points in image frame "(Z forward, X right, Y down)"
            to pixel coordinates in 2D image

            input:
                - points: (N,3) np.ndarray where ith row is [x,y,z] coordinate of ith point
                        where Z is forward, X right, and Y down
            output:
                - coords: (N,2) np.ndarray of floating point values representing position
                        where ith point in points would be in coords
        '''
        # following the procedure described here in the below articles
        # converting between 3D and 2D: https://support.stereolabs.com/hc/en-us/articles/4554115218711-How-can-I-convert-3D-world-coordinates-to-2D-image-coordinates-and-viceversa-
        # where pinhole parameters are stored: https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1CameraParameters.html#a94a642b5f1c8511df7584aa6a187f559
        # how to access pinhole parameters: https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#aabca561d73610707323bcec1a4c1d6b0

        cp = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        fx, fy, cx, cy = cp.fx, cp.fy, cp.cx, cp.cy
        N = points.shape[0]

        coords = np.zeros((N, 2))
        coords[:, 0] = (points[:, 0] / points[:, 1]) * fx + cx
        coords[:, 1] = (points[:, 2] / points[:, 1]) * fy + cy

        return coords

    def image_to_world(self, coords, depth):
        '''
            converts 2D pixel coordinates in images with their depth map
            to 3D world points in the image frame with (Z forward, X right, Y down)

            input:
                - coords: (N,2) np.ndarray where ith row is [u,v] coordinate of ith pixel
                - depth: (W,H) np.ndarray storing the depth values of pixels in image
            output:
                - points: (N,3) np.ndarray storing 3D world points corresonding to the pixel coordinates in coords
        '''
        # TODO: update this equation to work with Z-up right handed coordinate system

        cp = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        fx, fy, cx, cy = cp.fx, cp.fy, cp.cx, cp.cy
        N = coords.shape[0]

        u = coords[:, 0]
        v = coords[:, 1]
        Y = depth[u, v]

        # NOTE: had to modify formula to get it to match the structure of point cloud object
        # (X-values: use v instead of u, Y-vals: use u instead of v and negate, Z-vals: negate)
        points = np.zeros((N, 3))
        points[:, 0] = ((v - cx) * Y) / fx
        points[:, 1] = Y
        points[:, 2] = -((u - cy) * Y) / fy

        return points

    def create_worldimage_transformer(self):
        # cp = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        # return WorldImageTransformer(cp.fx, cp.fy, cp.cx, cp.cy)
        return

class ZEDCV2Resolution(Enum):
    HD480 = (640, 480)
    HD720 = (1280, 720)
    HD1080 = (1920, 1080)


class ZEDCV2:
    def __init__(self,
                 camera_index=0,
                 camera_resolution: ZEDCV2Resolution = ZEDCV2Resolution.HD720,
                 camera_fps=30,
                 **kwargs
                 ) -> None:
        """
        ZED interface class that utilizes OpenCV
        Args:
            camera_index (int): Index of camera plugged in via USB
                                Default = 0 (which should be the case if you have just one camera plugged in)
            camera_resolution (ZEDCV2Resolution): Enum object to specify camera resolution
                                                  Default = ZEDCV2Resolution.HD720 - i.e. (1280x720)
                                                  Other Possible Inputs = ZEDCV2Resolution.HD480 - i.e. (640x480)
                                                                          ZEDCV2Resolution.HD1080 - i.e. (1920x1080)
            camera_fps (int): Frame rate of video capture
                              If the camera cannot support the specified frame rate at the specified resolution, 
                              the camera will recorded at the highest frame rate possible
                              Default = 30 - i.e. 30 frames per second
        """
        self.cam = cv2.VideoCapture(camera_index)

        # Set camera parameters
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
        self.cam.set(cv2.CAP_PROP_FPS, camera_fps)
        for param_name, param_value in kwargs.items():
            self.cam.set(param_name, param_value)
        # Numpy arrays containing left, right images + depth map
        self.left_image = np.ndarray()
        self.right_image = np.ndarray()

    def open(self) -> None:
        """
        Open the ZED camera
        """
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            print("Possible reasons for failure to open camera:")
            print("1) The ZED is not plugged into the Jetson")
            print("2) A previous call was made to open the ZED")
            print("3) A previous python process did not terminate properly, thereby leaving the ZED opened. Restart the Jetson and try again")
            raise ZEDException
        print("Camera opened successfully!")

    def grab_data(self) -> None:
        """
        Retrieves the most up-to-date camera frame
        Returns:
            self.left_image_np (np.ndarray): Numpy array containing left image pixels
            self.right_image_np (np.ndarray): Numpy array containing right image pixels
        """
        # Extract the next frame
        _, frame = self.cam.read()

        # Split the frame to separate into left, right frames
        split_frame = np.split(frame, 2, axis=1)
        self.right_image = split_frame[0]
        self.left_image = split_frame[1]
        return self.left_image, self.right_image

    def close(self):
        self.cam.release()

    def get_cam(self):
        return self.cam