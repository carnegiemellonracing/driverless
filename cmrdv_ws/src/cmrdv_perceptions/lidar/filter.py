'''
If a filtering function is directly called by the LiDAR pipeline,
it must have the following specification
    Input: pointcloud = numpy array of 3D point positions
    Input: return_mask = boolean that dictates whether mask is returned with points
            - note: variable must be in function specification but is an optional variable
            - look at trim_cloud or plane_fit for example
    Output:
        - if not return_mask: filtered point cloud
        - if return_mask: filtered point cloud, mask

NOTE: pipeline functions must be registered in the bottom of the file
      to be used by overall pipeline in filter_fns
'''

import math
import numpy as np
from skspatial.objects import Plane

def trim_cloud(points, return_mask=False):
    '''
        Trims a cloud of points to reduce to a point cloud of only cone points
        by performing a naive implementation that goes as follows
            1. mask out all points that exceed a specific radius from the center
            2. mask out all points that are too close to the lidar (likely car)
            3. mask out all points that are in the ground or too high
                - this is done by measuring point distance from a
                  pre-defined plane
            4. mask out all points that are outside of a specific FOV angle

        Input: points - np.array of shape (N, M) where N is the number of points
                        and M is at least 3 and the first three columns
                        correspond to the (X, Y, Z) coordinates of the point
               return_mask - boolean whether to return the mask that filters out the points
        Output: if return_mask False
                    np.array of shape (N', M) where N' <= N and the resulting array
                    is from the result of filtering points according to the algo
                else (return_mask True)
                    (np.array described in False case, and numpy mask of length N)
    '''

    # # hyperparameters for naive ground filtering
    max_height = 5
    height = -1
    r_min = 1.5
    r_max = 10
    ang_cut = math.pi / 2
    scaling = 0.015

    # [DIST] select points within a radius of the center (0,0)
    distance = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
    mask_r_min = r_min < distance
    mask_r_max = distance < r_max
    mask_r = np.logical_and(mask_r_min, mask_r_max)

    # [ANGLE] select points that are within a specified angle +x-direction
    angle = np.abs(np.arccos(np.divide(points[:, 0], distance)))
    mask_angle = angle < ang_cut

    # [HEIGHT] select points that are above the ground but below some max height
    # after performing some transformations to consider a tilted ground
    distance = np.subtract(distance, 3.1)
    slope = np.multiply(distance, scaling)
    ground = np.add(height, slope)

    mask_z_l = points[:, 2] > ground
    mask_z_u = points[:, 2] < max_height
    mask_z = np.logical_and(mask_z_l, mask_z_u)

    # combine all masks to create a single mask and then select points
    mask = np.logical_and(mask_r, mask_z)
    mask = np.logical_and(mask, mask_angle)

    if return_mask:
        return points[mask], mask
    else:
        return points[mask]


def remove_ground(points, boxdim=0.5, height_threshold=0.01, xmin=-100, xmax=100, ymin=-100, ymax=100):

    all_points = points
    points = box_range(points, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    xmax, ymax = points[:, :2].max(axis=0)
    xmin, ymin = points[:, :2].min(axis=0)
    # print(xmax, ymax, xmin, ymin)
    LPR = []
    grid_points = []

    # iterate over all cells in the 2D grid overlayed on the x and y dimensions
    for i in range(int((xmax - xmin)//boxdim)):
        for j in range(int((ymax - ymin)//boxdim)):

            # find all points within the grid cell
            bxmin, bxmax = xmin + i*boxdim, xmin + (i+1)*boxdim
            bymin, bymax = ymin + j*boxdim, ymin + (j+1)*boxdim
            mask_x = np.logical_and(points[:, 0] < bxmax, bxmin < points[:, 0])
            mask_y = np.logical_and(points[:, 1] < bymax, bymin < points[:, 1])
            mask = np.logical_and(mask_x, mask_y)
            box = points[mask]

            grid_points.append(box)

            # find lowest point in cell if exists
            if box.size != 0:
                minrow = np.argmin(box[:, 2])
                boxLP = box[minrow].tolist()
                LPR.append(boxLP)

    if len(LPR) > 0:
        # fit lowest points to plane and use to classify ground points
        # P, C = vis.color_matrix(fns=None, pcs=[points, np.array(LPR)])
        # vis.update_visualizer_window(None, P, colors=C)

        plane = Plane.best_fit(LPR)
        A, B, C = tuple([val for val in plane.vector])
        D = np.dot(plane.point, plane.vector)

        dist_from_plane = A*all_points[:, 0] + B*all_points[:, 1] + C*all_points[:, 2] - D

        # store ground plane vals here
        pc_mask = height_threshold <= dist_from_plane
    else:
        pc_mask = np.ones(all_points.shape[0], dtype=np.uint8)
        plane = None

    return all_points[pc_mask], plane


def plane_fit(pointcloud, planecloud=None, return_mask=False, boxdim=0.5, height_threshold=0.01):
    if planecloud is None:
        planecloud = pointcloud

    xmax, ymax = planecloud[:, :2].max(axis=0)
    xmin, ymin = planecloud[:, :2].min(axis=0)
    # print(xmax, ymax, xmin, ymin)
    LPR = []

    # iterate over all cells in the 2D grid overlayed on the x and y dimensions
    for i in range(int((xmax - xmin)//boxdim)):
        for j in range(int((ymax - ymin)//boxdim)):

            # find all points within the grid cell
            bxmin, bxmax = xmin + i*boxdim, xmin + (i+1)*boxdim
            bymin, bymax = ymin + j*boxdim, ymin + (j+1)*boxdim
            mask_x = np.logical_and(planecloud[:, 0] < bxmax, bxmin < planecloud[:, 0])
            mask_y = np.logical_and(planecloud[:, 1] < bymax, bymin < planecloud[:, 1])
            mask = np.logical_and(mask_x, mask_y)
            box = planecloud[mask]

            # find lowest point in cell if exists
            if box.size != 0:
                minrow = np.argmin(box[:, 2])
                boxLP = box[minrow].tolist()
                LPR.append(boxLP)

    plane_vals = np.array([1, 2, 3, 4])

    if len(LPR) > 0:
        # fit lowest points to plane and use to classify ground points
        plane = Plane.best_fit(LPR)
        A, B, C = tuple([val for val in plane.vector])
        D = np.dot(plane.point, plane.vector)
        pc_compare = A*pointcloud[:, 0] + B*pointcloud[:, 1] + C*pointcloud[:, 2]
        # store ground plane vals here
        plane_vals = np.array([A, B, C, D])
        pc_mask = D + height_threshold < pc_compare
        # pc_mask = np.logical_and(D + height_threshold < pc_compare, pc_compare < D+height_max)
    else:
        pc_mask = np.ones(pointcloud.shape[0], dtype=np.uint8)

    if return_mask:
        return pointcloud[pc_mask], pc_mask, plane_vals
    else:
        return pointcloud[pc_mask]


def box_range(pointcloud, return_mask=False, xmin=-100, xmax=100, ymin=-100, ymax=100, zmin=-100, zmax=100):
    '''return points that are within the boudning box specified by the optional input parameters'''
    xrange = np.logical_and(xmin <= pointcloud[:, 0], pointcloud[:, 0] <= xmax)
    yrange = np.logical_and(ymin <= pointcloud[:, 1], pointcloud[:, 1] <= ymax)
    zrange = np.logical_and(zmin <= pointcloud[:, 2], pointcloud[:, 2] <= zmax)
    mask = np.logical_and(np.logical_and(xrange, yrange), zrange)

    points_filtered = pointcloud[mask]
    if return_mask:
        return points_filtered, mask
    else:
        return points_filtered


def circle_range(pointcloud, return_mask=False, radiusmin=0, radiusmax=100):
    '''return points that are within the radius plane in the x-y dimensions only, not in the z dimension!'''
    # get everything within radius
    dists = np.sqrt(np.sum(pointcloud[:, :2]**2, axis=1))
    mask = np.logical_and(radiusmin <= dists, dists <= radiusmax)

    points_filtered = pointcloud[mask]
    if return_mask:
        return points_filtered, mask
    else:
        return points_filtered


def covered_centroid(pointcloud, centroids, radius=0.75, height=0.5, threshold=5):
    '''filters out CENTROIDS that have some points above them with some threshold'''

    centroids_filtered = []

    for i in range(centroids.shape[0]):
        center = centroids[i, :]

        # get all points within a radius along x and y dimensions
        dists = np.sqrt(np.sum((pointcloud[:, :2] - center[:2])**2, axis=-1))
        cone_points = pointcloud[dists < radius]
        high_points = cone_points[cone_points[:, 2] > height]

        if high_points.shape[0] < threshold:
            centroids_filtered.append(centroids[i, :])

    if len(centroids_filtered) > 0:
        centroids_filtered = np.vstack(centroids_filtered)
    centroids_filtered = np.array(centroids_filtered)
    return centroids_filtered


# NOTE: register pipeline filtering fns here
filter_fns = {
    "naive": trim_cloud,
    "plane": plane_fit,
    "composite": lambda pc: plane_fit(circle_range(pc, radiusmin=0, radiusmax=100))
}