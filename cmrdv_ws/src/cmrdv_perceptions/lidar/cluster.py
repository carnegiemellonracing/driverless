'''
All clustering functions directly called by the LiDAR pipline must adhere
to the following specification
    Input: pointcloud - numpy array of 3D point positions of filtered LiDAR pointcloud
    Output: centroids - numpy array of 3D point positions of cones

NOTE: pipeline functions must be registered in the bottom of the file
      to be used by overall pipeline in cluster_fns
'''

import hdbscan
import math
import numpy as np
import time
from sklearn import cluster

CORRECTION = np.array([0.0693728, 0.12893042])

####################
# HELPER FUNCTIONS #
####################


def run_hdbscan(points, eps=1, min_samples=1):
    '''
        runs hdbscan on a point cloud and returns the resulting 
        fitted clusterer -- has information that is accessible in the follow way
            1. clusterer.labels_ = assigned cluster for each point
            2. clusterer.probabilities_  = probability each point in its cluster

        Input: points - np.array of shape (N, 3) where the 3 columns are
                        (x, y, z) coordinates representing a point cloud
        Output: clusterer - hdbscan.HDBSCAN object that is fit on points
    '''
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples, gen_min_span_tree=True, cluster_selection_epsilon=eps)
    clusterer.fit(points)
    return clusterer


def run_dbscan(points, eps=1, min_samples=1):
    '''
        identical to run_hdbscan but runs using DBSCAN from sklearn library
    '''
    clusterer = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(points)

    # DBSCAN doesn't use probabilities so need to setting to one for now
    clusterer.probabilities_ = np.ones(clusterer.labels_.shape)

    return clusterer


def compute_centers(points, labels, probs):
    points = points[:, :3]
    n_clusters = np.max(labels) + 1
    centroids = []

    for i in range(n_clusters):
        # for each cluster estimate its centroid from its points
        idxs = np.where(labels == i)[0]
        cluster_points = points[idxs]
        cluster_probs = probs[idxs]

        scale = np.sum(cluster_probs)
        center = np.sum(cluster_points * cluster_probs, axis=0) / scale

        centroids.append(center)

    return np.array(centroids)


def filter_centers(all_points, clustering_points, centers, labels, probs):
    # TODO: need parameter file containing physical information about lidar and cones
    CONE_RADIUS = 0.075
    CONE_HEIGHT = 0.3
    CONE_WIDTH = CONE_RADIUS * 2
    DIST_EPS = 0.015
    HIGH_POINT_RADIUS = 0.3
    MAX_CONE_HEIGHT = 0.5   # SHOULD BE RELATIVE TO THE HEIGHT OF THE CONE W.R.T GROUND, NOT W.R.T. LIDAR
    LIDAR_VERT_RESOLUTION = 2
    LIDAR_HORIZ_RESOLUTION = 0.2
    DEG_TO_RAD = np.pi / 180
    NPOINTS_EPS = 50

    clustering_points = clustering_points[:, :3]
    all_points = all_points[:, :3]
    n_clusters = np.max(labels) + 1
    filtered_centers = []

    for i in range(centers.shape[0]):
        # assumes that the ith cluster in clusters correspondes to label i
        idxs = np.where(labels == i)[0]
        cluster_points = clustering_points[idxs]
        cluster_probs = probs[idxs].reshape(-1)
        cluster_center = centers[i]

        # condition 1: filter on average distance from center (should be small)
        dists = np.sqrt(np.sum((cluster_points[:, :2] - cluster_center[:2]) ** 2, axis=1))
        scale = np.sum(cluster_probs)
        avg_dist = np.sum(dists * cluster_probs) / scale
        if not (avg_dist < CONE_RADIUS + DIST_EPS):
            print("cluster to varying to be cone")
            continue

        # condition 2: filter on max point above the center
        dists = np.sqrt(np.sum((all_points[:, :2] - cluster_center[:2]) ** 2, axis=1))
        close_points = all_points[dists < HIGH_POINT_RADIUS]
        max_point_height = np.max(close_points[:, 2])
        if not (max_point_height < MAX_CONE_HEIGHT):
            print("found high points")
            continue

        # condition 3: filter on expected number of points
        dists = np.sqrt(np.sum((all_points[:, :2] - cluster_center[:2]) ** 2, axis=1))
        cluster_dist = np.sqrt(np.sum(cluster_center ** 2))
        cone_points = all_points[dists < CONE_RADIUS]
        num_cone_points = cone_points.shape[0]

        expected_height = CONE_HEIGHT / (2 * cluster_dist * np.tan((LIDAR_VERT_RESOLUTION * DEG_TO_RAD) / 2))
        expected_width = CONE_WIDTH / (2 * cluster_dist * np.tan((LIDAR_HORIZ_RESOLUTION * DEG_TO_RAD) / 2))
        num_points_expected = (1/2) * expected_height * expected_width
        if not (abs(num_points_expected - num_cone_points) < NPOINTS_EPS):
            print("incorrect number of points")
            continue

        print(cluster_center, num_points_expected, num_cone_points)
        filtered_centers.append(cluster_center)

        pass

    return np.array(filtered_centers)


def get_centroids(points, labels, probs=None, filter_distant=False, dist_threshold=0.2):
    '''
        Gets centroid of each cluster designated by the points and labels

        If probs is specified, then calculates centroid via performs 
        weighted average of positions with probabilities as weights, 
        otherwise does normal average of point positions

        PRE: points, labels, and probs (if not None) must have the same length
        PRE: labels are from 0 to K-1 where K is the number of clusters
        PRE: points are (N, M) where M >= 3 and the first 3 elems are X, Y, Z

        Input: points - np.array of shape (N, M) where M >= 3 and first 3 cols
                        are X, Y, Z coordinates of point cloud points
               labels - np.array of shape (N,) where labels[i] is the assigned
                        cluster of the point represented by points[i]
               probs  - np.array of shape (N,) if not None where probs[i]
                        is the probability that points[i] is in the cluster
                        designated by labels[i]
               filter_distant - boolean if True, then if the average distance of
                                points from estimated cluster center is more than
                                dist_threshold, then reject cluster and don't
                                add to return list, if False, then return all clusters
               dist_threshold - the max average distance of point from estimated
                                cluster center to add the center to return array

        Output: centroids - np.array of shape (K', 3) where K' = K if 
                            filter_distant is False, otherwise, K' <= K of
                            [X, Y, Z] points representing the centroids of
                            clusters calculated from the points and given labels

    '''

    # hyperparameter: if centroid has average distance > dist_threshold
    # from cluster points and filter_distant true, then do not add it to result
    # used to solve issue if a two cones' points are in one cluster,
    # then centroid is in between them, for now, just remove

    # only need to consider (x, y, z) position for centroid calculations
    points = points[:, :3]
    n_clusters = np.max(labels) + 1
    centroids = []

    if probs is None:
        probs = np.ones((points.shape[0], 1))

    for i in range(n_clusters):
        # for each cluster estimate its centroid from its points
        idxs = np.where(labels == i)[0]
        cluster_points = points[idxs]

        cluster_probs = probs[idxs]
        scale = np.sum(cluster_probs)
        center = np.sum(cluster_points * cluster_probs, axis=0) / scale

        # do not add cone centroid if it is too far from its cluster
        if not filter_distant:
            centroids.append(center)
        else:
            # calculate the average distance of the cluster from centroid
            dists = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
            cluster_probs = probs[idxs].reshape(-1)
            scale = np.sum(cluster_probs)
            avg_dist = np.sum(dists * cluster_probs) / scale

            if avg_dist <= dist_threshold:
                centroids.append(center)

    return np.array(centroids)


def get_centroids_z(points, labels, ground_planevals, probs=None, filter_distant=True, dist_threshold=0.2, x_threshold_scale=2, height_threshold=4, scalar=1, x_bound=10, x_dist=1):
    '''
        Gets centroid of each cluster designated by the points and labels

        If probs is specified, then calculates centroid via performs 
        weighted average of positions with probabilities as weights, 
        otherwise does normal average of point positions

        PRE: points, labels, and probs (if not None) must have the same length
        PRE: labels are from 0 to K-1 where K is the number of clusters
        PRE: points are (N, M) where M >= 3 and the first 3 elems are X, Y, Z

        Input: points - np.array of shape (N, M) where M >= 3 and first 3 cols
                        are X, Y, Z coordinates of point cloud points
               labels - np.array of shape (N,) where labels[i] is the assigned
                        cluster of the point represented by points[i]
                ground_planevals - np.array of shape 4 where the first 3 elements are
                            vectors that make up the plane and the 4th element 
                            is the vertical translation on the z-axis
               probs  - np.array of shape (N,) if not None where probs[i]
                        is the probability that points[i] is in the cluster
                        designated by labels[i]
               filter_distant - boolean if True, then if the average distance of
                                points from estimated cluster center is more than
                                dist_threshold, then reject cluster and don't
                                add to return list, if False, then return all clusters
               dist_threshold - the max average distance of point from estimated
                                cluster center to add the center to return array
                x_threshold_scale - the scale onto which dist_threshold decreases as 
                            you move further along the x-axis
            height_threshold - clusters with points higher than this threshold are
                            assumed to not be cones and filtered out
            scalar - the magnitude onto which the z dimension is compressed
            x_bound - the bounds on the x-axis onto which clusters are kept, since
                    clusters further along the x-axis are assumed not to be
                    cones
        x-dist - the distance within x_bound in which we remove centroids

        Output: centroids - np.array of shape (K', 3) where K' = K if 
                            filter_distant is False, otherwise, K' <= K of
                            [X, Y, Z] points representing the centroids of
                            clusters calculated from the points and given labels

    '''

    # hyperparameter: if centroid has average distance > dist_threshold
    # from cluster points and filter_distant true, then do not add it to result
    # used to solve issue if a two cones' points are in one cluster,
    # then centroid is in between them, for now, just remove

    # only need to consider (x, y, z) position for centroid calculations
    points = points[:, :3]
    points[:, 2] *= scalar
    n_clusters = np.max(labels) + 1
    centroids = []

    if probs is None:
        probs = np.ones((points.shape[0], 1))

    for i in range(n_clusters):
        s = time.time()
        # for each cluster estimate its centroid from its points
        idxs = np.where(labels == i)[0]
        cluster_points = points[idxs]

        cluster_probs = probs[idxs]
        scale = np.sum(cluster_probs)
        center = np.sum(cluster_points * cluster_probs, axis=0) / scale

        # do not add cone centroid if it is too far from its cluster
        if not filter_distant:
            centroids.append(center)
        else:
            # calculate the average distance of the cluster from centroid
            dists = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
            # outer_points are points that are some radial distance away from
            # centroid. We tried to use these points to detect lines rather
            # than cones exept ran into issues with cones doubling up close to
            # the lidar
            # outer_points = np.sum(np.sum((cluster_points - center) ** 2, axis=1) > 0.1)
            cluster_probs = probs[idxs].reshape(-1)
            scale = np.sum(cluster_probs)
            avg_dist = np.sum(dists * cluster_probs) / scale
            # max_cluster_z = cluster_points[:, 2].max(axis=0)

            # find point heights based on projection from the ground plane
            point_heights = ground_planevals[0]*cluster_points[:, 0] + ground_planevals[1] * \
                cluster_points[:, 1] + ground_planevals[2]*cluster_points[:, 2] - ground_planevals[3]
            max_cluster_z = point_heights.max()

            # find what we want our distance threshold to be for averagedists
            # the distance threshold is equal to dist_threshold for clusters
            # within the xbounds [4, -4]. Going further outwards from x, the
            # dist_threshold decreases by the scalar x_threshold_scale
            curr_dist_thresh = dist_threshold*(1-x_threshold_scale*max(0, abs(center[0])-4))

            # If there is onyl one cone, we just set our avg_dist to be higher
            # than the distance threshold, thereby always labelling single point'
            # clusters as cones
            if(len(cluster_points) == 1 and (abs(center[0]) > 8)):
                avg_dist = curr_dist_thresh+1

            # select centroids based on following criteria:
            # - avg_dist being smaller than curr_dist_thresh
            # - clusters having a max value smaller than height_threshold
            # - center being mores than x_dist away from our x_bounds
            #       - Logic behind the above is that cones that are too far away
            #       from the car left and right we don't really care about anyway
            #       But we want to keep those points within our clutsering stage
            #       because without them. If they were not included, the cluster
            #       that originally included those points would have a centroid
            #       with a much lower x and much lower avg_dist than actual.

            # dist = math.sqrt((center[0] ** 2) + (center[1] ** 2) + (center[2] ** 2))
            if (avg_dist <= curr_dist_thresh) and \
                    (max_cluster_z <= height_threshold) and \
                    (abs(center[0]) <= x_bound-x_dist):
                # (dist < 4 or outer_points < 1):
                # if True:
                # if (max_cluster_z <= height_threshold) and len(outer_points) < 6:
                # print("height: " + str(max_cluster_z))
                # print(" - numpoints: " + str(len(cluster_points)))
                # print(" -- avgdist: " + str(avg_dist))
                # print(" --- xdist: " + str(center[0]) + "\n")
                # print(" ---- outerPoints: " + str(outer_points) + "\n")
                # print(" ---- dists: " + str(dist) + "\n")
                centroids.append(center)

    return np.array(centroids)

#####################################
# PRIMARY CONE PREDICTION FUNCTIONS #
#####################################


def cluster_points(points, hdbscan=False, eps=1, min_samples=1):
    # only care about (X, Y, Z) coordinates of points
    points = points[:, :3]

    # run HDBSCAN and get the resulting labels and probabilities
    if hdbscan:
        clusterer = run_hdbscan(points, eps=eps, min_samples=min_samples)
    else:
        clusterer = run_dbscan(points, eps=eps, min_samples=min_samples)

    labels = clusterer.labels_.reshape((-1, 1))
    probs = clusterer.probabilities_.reshape((-1, 1))
    centers = compute_centers(points, labels, probs)

    return centers, labels, probs


def predict_cones(points, hdbscan=False, get_colors=None):
    '''
        predicts the centers of cones given a point cloud
        by running some clustering algorithm (in this implementation HDBSCAN)
        and then predicting the cone centers (using get_centroids)

        Input: points - np.array of shape (N, M) where M >= 3 and the first 3
                        columns of points represent X, Y, Z coordinates of
                        point cloud points of same units
        Output: centroids - np.array of shape (C, 3) where there were C
                            C predicted cone centers from the point cloud
                            represented by points
    '''

    # only care about (X, Y, Z) coordinates of points
    points = points[:, :3]

    # run HDBSCAN and get the resulting labels and probabilities
    if hdbscan:
        clusterer = run_hdbscan(points)
    else:
        clusterer = run_dbscan(points)
    labels = clusterer.labels_.reshape((-1, 1))
    probs = clusterer.probabilities_.reshape((-1, 1))

    # get the cone centers and return
    centroids = get_centroids(points, labels, probs)

    if get_colors:
        colors = np.hstack([labels, labels, labels]) / np.max(labels)
        return centroids, colors
    else:
        return centroids


def predict_cones_z(points, ground_planevals, hdbscan=False, scalar=1, dist_threshold=0.2, x_threshold_scale=1, height_threshold=4, x_bound=10, x_dist=1):
    '''
        predicts the centers of cones given a point cloud
        by running some clustering algorithm (in this implementation HDBSCAN)
        and then predicting the cone centers (using get_centroids)

        Input: points - np.array of shape (N, M) where M >= 3 and the first 3
                        columns of points represent X, Y, Z coordinates of
                        point cloud points of same units
        ground_planevals - np.array of shape 4 where the first 3 elements are
                            vectors that make up the plane and the 4th element 
                            is the vertical translation on the z-axis
        scalar - the magnitude onto which the z dimension is compressed
        dist_threshold - the starting threshold of max distance from center of
                        cluster to furthest point
        x_threshold_scale - the scale onto which dist_threshold decreases as 
                            you move further along the x-axis
        height_threshold - clusters with points higher than this threshold are
                            assumed to not be cones and filtered out
        x_bound - the bounds on the x-axis onto which clusters are kept, since
                    clusters further along the x-axis are assumed not to be
                    cones
        x-dist - the distance within x_bound in which we remove centroids

        Output: centroids - np.array of shape (C, 3) where there were C
                            C predicted cone centers from the point cloud
                            represented by points
    '''

    # only care about (X, Y, Z) coordinates of points
    # Squish all the points by a value based on the zmax and the scalar (endscal)
    # We do this so our clustering pretty much ignores the z-values and thereby
    # only clusters on the xy-region. We will alter re-expand the point cloud by
    # the endscal.
    points = points[:, :3]
    zmax = points[:, 2].max(axis=0)
    # zmin = points[:, 2].min(axis=0)
    # print("max: " + str(zmax) + "\n")
    # print("min: " + str(zmin) + "\n")
    endscal = scalar*(abs(zmax))
    points[:, 2] /= (endscal)

    # run HDBSCAN and get the resulting labels and probabilities
    if hdbscan:
        clusterer = run_hdbscan(points)
    else:
        clusterer = run_dbscan(points)
    labels = clusterer.labels_.reshape((-1, 1))
    probs = clusterer.probabilities_.reshape((-1, 1))

    # get the cone centers and return
    centroids = get_centroids_z(points, labels, ground_planevals, probs, dist_threshold=dist_threshold,
                                x_threshold_scale=x_threshold_scale, height_threshold=height_threshold, scalar=endscal, x_bound=x_bound, x_dist=x_dist)

    return centroids


def correct_clusters(points):
    points[:, :2] = points[:, :2] + CORRECTION
    return points


# NOTE: reigster pipeline clustering fns here
cluster_fns = {
    "dbscan": lambda pts: predict_cones(pts, hdbscan=False),
    "hdbscan": lambda pts: predict_cones(pts, hdbscan=True)
}