import cmrdv_ws.src.cmrdv_common.cmrdv_common.config.perceptions_config as cfg_perceptions
from cmrdv_ws.src.cmrdv_common.cmrdv_common.conversions import numpy_to_image, image_to_numpy, pointcloud2_to_array
# from cmrdv_interfaces.msg import DataFrame, SimDataFrame
import torch
import statistics
import cv2
import numpy as np
import matplotlib.pyplot as plt

import cmrdv_ws.src.cmrdv_perceptions.stereo_vision.ZED as ZED

COLORS = {
    1: (255, 191, 0),
    0: (0, 150, 255),
}

CV2_COLORS = {
    cfg_perceptions.COLORS.BLUE: [255, 191, 0],
    cfg_perceptions.COLORS.YELLOW: [7, 238, 255],
    cfg_perceptions.COLORS.ORANGE: [0, 150, 255]
}


def calc_box_center(box):
    """
    Args:
        bounding_box: Box containing object of interest
                        bound[0] = x1 (top left - x)
                        bound[1] = y1 (top left - y)
                        bound[2] = x2 (bottom right - x)
                        bound[3] = y2 (bottom right - y)
        Returns:
            center_x, center_y - center x, y pixel of bounding box 
    """
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    center_y = int((x1+x2)/2)  # x-coordinate of bounding box center
    center_x = int((y1+y2)/2)  # y-coordinate of bounding box center
    return center_x, center_y


def get_object_depth(depth_map, box, padding=2):
    """
    Calculates the median z position of supplied bounding box
    Args:
        box: Box containing object of interest
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
    center_x, center_y = calc_box_center(box)

    # Iterate through center pixels and append them to z_vect
    for i in range(max(center_x - padding, 0), min(center_x + padding, 720)):
        for j in range(max(center_y - padding, 0), min(center_y + padding, 1280)):
            z = depth_map[i, j]
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


def get_cone_color(left_frame, box, padding=2):
    # return config.COLORS.BLUE
    center_x, center_y = calc_box_center(box)
    min_x = max(center_x - padding, 0)
    max_x = min(center_x + padding, cfg_perceptions.IMAGE_HEIGHT)

    min_y = max(center_y - padding, 0)
    max_y = min(center_y + padding, cfg_perceptions.IMAGE_WIDTH)

    rgbs = left_frame[min_x:max_x, min_y:max_y, :3].reshape(-1, 3)
    avg_rgbs = np.average(rgbs, axis=0)
    if avg_rgbs[2] < cfg_perceptions.RED_THRESHOLD:
        return cfg_perceptions.COLORS.BLUE
    else:
        return cfg_perceptions.COLORS.YELLOW


def visualizePredictions(image, boxes, predictions, window_name="left w/ predictions"):
    # displaay boxes
    for i, box in enumerate(boxes):
        _, _, depth_z, color = predictions[i]
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[2]), int(box[3]))
        c = CV2_COLORS[color]
        image = cv2.rectangle(image.copy(), top_left, bottom_right, c, 3)
    cv2.imshow(window_name, image)
    cv2.waitKey(1)
    return


def predict(model, zed, sim=False):
    # why don't we use the predictions from the model for the color of the cone?
    # instead of re-calculating the color from the image

    left_image_np, _, _, point_cloud_np = zed.grab_data()
    left_img = left_image_np
    zed_pts = point_cloud_np
    print(np.mean(left_img), np.std(left_img))

    pad = 5

    blue_cones, yellow_cones, orange_cones, predictions = [], [], [], []
    boxes_with_depth = []

    # model expects RGB, convert BGR to RGB
    boxes = model(left_img[:,:,[2,1,0]], size=640)
    pred_color_dict = boxes.names

    nr, nc = left_img.shape[:2]

    print("performing prediction")
    num_cones = len(boxes.xyxy[0])
    for i, box in enumerate(boxes.xyxy[0]):
        #depth_y = get_object_depth(box, padding=1) # removing get_object_depth because need depth map (not in DataFrame)
        # and also not as accurate of an indicator of position as the point cloud which is just some func(depth, cp)
        # where cp is the camera parameters
        center_x, center_z = calc_box_center(box)
        color_id = int(box[-1].item())

        xl = max(0, center_x - pad)
        xr = min(center_x + pad, nr)
        zt = max(0, center_z - pad)
        zb = min(center_z + pad, nc)

        #coord = zed_pts[center_x, center_z]
        #world_x, world_y, world_z = coord['x'], coord['y'], coord['z']

        coords = zed_pts[xl:xr, zt:zb]
        if zed is None:
            world_xs, world_ys, world_zs = coords['x'], coords['y'], coords['z']
        else:
            world_xs, world_ys, world_zs = coords[:,:,0].reshape(-1), coords[:,:,1].reshape(-1), coords[:,:,2].reshape(-1)

        idxs = np.logical_and(~np.isnan(world_xs), ~np.isinf(world_xs))
        world_xs = world_xs[idxs]
        world_ys = world_ys[idxs]
        world_zs = world_zs[idxs]

        if world_xs.shape[0] == 0:
            print(f"\t[PERCEPTIONS WARNING] (cone {i} of {num_cones}) detected cone but no depth; throwing away")
            break

        world_x = np.mean(world_xs)
        world_y = np.mean(world_ys)
        world_z = np.mean(world_zs)

        color_str = pred_color_dict[color_id]
        if color_str == "yellow_cone":
            color = cfg_perceptions.COLORS.YELLOW
        elif color_str == "blue_cone":
            color = cfg_perceptions.COLORS.BLUE
        elif color_str == "orange_cone" or color_str == "large_orange_cone":
            color = cfg_perceptions.COLORS.ORANGE
        else:
            print("stereo-vision YOLO: Found unknown cone -- ignoring")
            color = cfg_perceptions.COLORS.UNKNOWN

        color = get_cone_color(left_img, box, padding=2)

        prediction = [world_x,
                      world_y,
                      world_z,
                      color]
        boxes_with_depth.append(box)
        predictions.append(prediction)

        if color == cfg_perceptions.COLORS.YELLOW:
            yellow_cones.append(prediction)
        elif color == cfg_perceptions.COLORS.BLUE:
            blue_cones.append(prediction)
        elif color == cfg_perceptions.COLORS.ORANGE:
            orange_cones.append(prediction)

    # visualizePredictions(left_img, boxes_with_depth, predictions)
    # visualizePredictions(left_img[:, :, [2, 1, 0]], boxes_with_depth, predictions, window_name="our BGR image")
    return blue_cones, yellow_cones, orange_cones


