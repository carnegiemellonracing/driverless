'''
    file: lidar.py
    brief: contains the high-level implementation and API of the LiDAR Pipeline
    which, upon initialization, will give the functionality to predict and
    return point of estimated cone positions

    Usage:
    Note: must be connected to a LiDAR for this to work
    >>> import fsdv.perceptions.lidar.lidar as lidar
    >>> buffer = lidar.init()
    >>> cones = lidar.predict(buffer)
'''

import cmrdv_ws.src.cmrdv_perceptions.lidar.visualization as vis
import cmrdv_ws.src.cmrdv_perceptions.lidar.collect as collect
import cmrdv_ws.src.cmrdv_perceptions.lidar.filter as filter
import cmrdv_ws.src.cmrdv_perceptions.lidar.cluster as cluster
import cmrdv_ws.src.cmrdv_perceptions.lidar.color as color

import time
import numpy as np


class LidarPredictor:

    def __init__(self):
        pass

    def predict(self, points):
        # perform a box range on the data
        points_ground_plane = filter.box_range(
            points, xmin=-6, xmax=6, ymin=-6, ymax=6, zmin=-1, zmax=1)

        # perform a plane fit and remove ground points
        xbound = 10
        points_filtered_ground, _, ground_planevals = filter.plane_fit(
            points, points_ground_plane, return_mask=True, boxdim=0.5, height_threshold=0.05)

        # perform another filtering algorithm to dissect boxed-region
        points_cluster, mask_cluster = filter.box_range(
            points_filtered_ground, xmin=-xbound, xmax=xbound, ymin=-10, ymax=50, zmin=-10, zmax=100, return_mask=True)

        # predict cones using a squashed point cloud and then unsquash
        cone_centers = cluster.predict_cones_z(
            points_cluster, ground_planevals, hdbscan=False, dist_threshold=0.6, x_threshold_scale=0.15, height_threshold=0.3, scalar=1, x_bound=xbound, x_dist=3)

        P, C = vis.color_matrix(fns=None, pcs=[points, points_filtered_ground, points_cluster])

        # color cones and return them
        cone_output, cone_centers, cone_colors = color.color_cones(cone_centers)

        # correct the positions of the cones to the center of mass of the car
        cone_output = cluster.correct_clusters(cone_output)

        idxs_blue = cone_colors == 0
        idxs_yellow = cone_colors == 1
        idxs_orange = cone_colors == 2
        cone_output = np.hstack([cone_output[:, :2], np.zeros((cone_output.shape[0], 1))])

        # currently LiDAR coloring does not support coloring orange cones
        blue_cones = cone_output[idxs_blue]
        yellow_cones = cone_output[idxs_yellow]
        orange_cones = cone_output[idxs_orange]
        return blue_cones, yellow_cones, orange_cones


def init():
    '''
        The initializing function for the LiDAR pipeline. It starts a background
        process that updates the returned variable `buffer` with the latest
        points in the LiDAR point cloud which MUST be passed into the
        lidar.predict function in order for it to work appropriately.
    '''
    _, buffer = collect.init_concurrent_collect(perceptions_starts=True)
    return buffer


def predict_general(buffer=None, pcfile=None, data=None):
    '''
        assumption: all cones are 0.5 meter radius from other objects and
        are a given cone_height
        goal: this function should be able to detect all such cones that
        follow the above objective
    '''

    if buffer is None and pcfile is None and data is None:
        raise Exception("lidar.predict -- did not pass in data source nor point cloud files")

    # load the data
    if data is None:
        data = collect.concurrent_collect(buffer) if buffer is not None else np.load(pcfile)
    all_points = data[:, :3]

    # TODO: need to replace this with multi-plane ground filtering
    # filter out the ground
    # zmax here should be relative to ground and not lidar height
    points = filter.box_range(all_points, xmin=-5, xmax=5, ymin=0.2, ymax=15, zmax=0.2)
    points_unground, plane = filter.remove_ground(
        points, boxdim=0.25, height_threshold=0.075, xmin=-1.5, xmax=1.5, ymax=3)

    # cluster on those points
    centers, labels, probs = cluster.cluster_points(points_unground, eps=0.25, min_samples=1)

    # filter clusters that have points that are too high
    filtered_centers = cluster.filter_centers(all_points, points_unground, centers, labels, probs)

    # timing and visualize
    P, C = vis.color_matrix(fns=None, pcs=[points, points_unground])
    vis.update_visualizer_window(None, P, colors=C, pred_cones=filtered_centers, plane=plane)

    pass


def predict(buffer=None, pcfile=None):
    '''
        Predict the cone centers from a point cloud
            - if pcfile not specified, assumes LiDAR is connected and uses
              point cloud that is collected from the buffer object from
              lidar.init()
            - if pcfile specified, assumes that it is a ".npy" file that stores
              an np.array of shape (N, M) where M >= 3 and there are N points
              in the point clouds and the first 3 columns of if are (X, Y, Z)
              coordinates of points

        NOTE: this function serves as a model for our general pipeline
            1. get points (either by disk or by collecting with connected LiDAR)
            2. filter any points in the point cloud that are unnecessary/noisy
            3. cluster the resulting filtered point cloud to generate cone
               center predictions
            4. color the cones using some heuristical algorithm

        PRE: if pcfile is specified, must end in ".npy" extension and must
             store an array described above
        Input: buffer - the shared memory buffer variable from lidar.init()
               pcfile - optional file that stores a point cloud
        Outpu: cone_centers - np.array of shape (C,3) where there are C centers
                              representing the centers of each cone in the
                              point cloud and the columns correspond to
                              [x, y, c_id] where x and y are positions in meters
                              and c_id is the id of the cone which can either be
                              1 for yellow, 0 for blue, or -1 for undetermined
                              color
    '''

    # TODO: delete

    if buffer is None and pcfile is None:
        raise Exception("lidar.predict -- did not pass in data source nor point cloud files")

    # load the data
    if buffer is not None:
        data = collect.concurrent_collect(buffer)
        points = data[:, :3]
    else:
        points = np.load(pcfile)[:, :3]

    # perform a box range on the data
    points_ground_plane = filter.box_range(
        points, xmin=-6, xmax=6, ymin=-6, ymax=6, zmin=-1, zmax=1)

    # perform a plane fit and remove ground points
    xbound = 10
    points_filtered_ground, _, ground_planevals = filter.plane_fit(
        points, points_ground_plane, return_mask=True, boxdim=0.5, height_threshold=0.05, height_max=1000000)

    # perform another filtering algorithm to dissect boxed-region
    points_cluster, mask_cluster = filter.box_range(
        points_filtered_ground, xmin=-xbound, xmax=xbound, ymin=-10, ymax=50, zmin=-10, zmax=100, return_mask=True)

    # predict cones using a squashed point cloud and then unsquash
    cone_centers = cluster.predict_cones_z(
        points_cluster, ground_planevals, hdbscan=False, dist_threshold=0.6, x_threshold_scale=0.15, height_threshold=0.3, scalar=1, x_bound=xbound, x_dist=3)

    P, C = vis.color_matrix(fns=None, pcs=[points, points_filtered_ground, points_cluster])

    # color cones and return them
    cone_output, cone_centers, cone_colors = color.color_cones(cone_centers)

    # correct the positions of the cones to the center of mass of the car
    cone_output = cluster.correct_clusters(cone_output)

    # TODO: delete
    vis.update_visualizer_window(None, points=P, pred_cones=cone_centers, colors=C)

    return cone_output


if __name__ == "__main__":
    # PIPELINE 1
    ogdir = 'fsdv/perceptions/lidar/data/fms-fifth/pc/track-12500.npz'
    hdict = np.load(ogdir)

    # v = None
    v = vis.init_visualizer_window()
    avvis = AVVis()

    avg_time = 0
    avg_max_dist = 0
    iter = 0
    while True:
        for i, a in hdict.items():
            s = time.time()
            points = a[:, :3]

            # take a random subsample for testing purposes
            # takes subset of poitns to create the ground plane, uses points within a box range
            points_ground_plane, mask_ground_plane = filter.box_range(
                points, xmin=-6, xmax=6, ymin=-6, ymax=6, zmin=-1, zmax=1, return_mask=True)

            xbound = 10

            # unused
            # points_filtered_smaller_x, mask_smaller_x = filter.box_range(
            #     points, xmin=-xbound, xmax=xbound, ymin=-10, ymax=50, zmin = -10, zmax = 100, return_mask=True)

            # creates ground plane, returns the ground_planevals to define the plane for future use
            # keeps non-ground points in points_filtered_ground
            # currently not bounding the z-height so the height_max is set very high
            points_filtered_ground, mask_filtered_ground, ground_planevals = filter.plane_fit(
                points, points_ground_plane, return_mask=True, boxdim=0.5, height_threshold=0.05)

            # Filter out the points to be used in clustering
            # The points within the box range contain the region where we expect cones to be
            # and where we care about if cones are in that region
            points_cluster, mask_cluster = filter.box_range(
                points_filtered_ground, xmin=-xbound, xmax=xbound, ymin=-10, ymax=50, zmin=-10, zmax=100, return_mask=True)

            # cluster the cones and predict the cone centers
            cone_centers = cluster.predict_cones_z(
                points_cluster, ground_planevals, hdbscan=False, dist_threshold=0.6, x_threshold_scale=0.15, height_threshold=0.3, scalar=1, x_bound=xbound, x_dist=3)

            # color the cones
            cone_output, cone_centers, cone_colors = color.color_cones(cone_centers)

            # correct the positions of the cones to the center of mass of the car
            cone_output = cluster.correct_clusters(cone_output)

            t = (time.time() - s) * 1000
            d = np.max(np.sqrt(np.sum(cone_centers**2, axis=-1)))

            avg_time = ((avg_time) * iter + t) / (iter + 1)
            avg_max_dist = ((avg_max_dist) * iter + d) / (iter + 1)
            iter += 1

            print(f"{t}ms {int(d)}m {avg_time:.1f}ms {avg_max_dist:.2f}m")

            # visualize stuff
            P, C = vis.color_matrix(fns=None, pcs=[points, points_filtered_ground, points_cluster])

            vis.update_visualizer_window(v, points=P, pred_cones=cone_centers, colors=C,
                                         colors_cones=cone_colors/np.max(cone_colors))
