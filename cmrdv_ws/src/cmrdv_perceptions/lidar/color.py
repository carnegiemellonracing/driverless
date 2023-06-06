import numpy as np
import matplotlib.pyplot as plt

# NOTE: REMOVE ONCE DONE
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

C2RGB = {
    "blue": [7, 61, 237],       # cone color: Blue
    "yellow": [255, 209, 92],   # cone color: Yellow
    "nocolor": [0, 0, 0],       # undetermined cone color: Black
    "red": [232, 49, 81]        # midline spline color: Red
}


def split_by_y(points):
    '''
        assumes that points is N x 3 (idx, x, y)
        returns left_arr and right_arr where left_arr is all points in points
        with y < 0 and right_arr is all points in points with y >= 0
    '''
    right_idxs = points[:, 1] >= 0
    return points[np.logical_not(right_idxs)], points[right_idxs]


def next_point_simple(curr_point, points, dir, max_angle_diff=np.pi / 3.5):
    '''
        assume that curr_point is (3,) and points is N x 3 (idx, x, y)
        dir is an angle in radians

        looks for a point in from of the point at some max radius
        and at some thresholded value of how far off of the radius to consider
    '''

    max_dist = 6

    # to ignore the index
    points_index = points[:, 0].reshape((-1, 1))

    # compute the distances and angles of all points relative to curr_point
    deltas = points[:, 1:] - curr_point[1:]
    dists = np.sqrt(np.sum(deltas**2, axis=1))
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    angles = np.where(angles < 0, angles + 2 * np.pi, angles)
    angle_diffs = np.abs(angles - dir)

    # get all points within angle range and distance range
    is_close = np.logical_and(dists < max_dist, angle_diffs < max_angle_diff)

    if np.any(is_close):
        points_close = points[is_close]
        dists_close = dists[is_close]
        angles_close = angles[is_close]
        idx = np.argmin(dists_close)
        return points_close[idx], angles_close[idx]
    else:
        return None, None


def plot_dir(start, dir, scale=3):
    vec_dir = scale * np.array([np.cos(dir), np.sin(dir)])
    end = start + vec_dir
    plt.arrow(start[0], start[1], vec_dir[0], vec_dir[1], width=0.25)


def plot(centers, colors):
    '''
        assumes that centers is N x 2
        and colors is N x 1 and corresponds to centers
    '''
    plt.scatter(centers[:, 0], centers[:, 1], c=colors)
    plt.scatter([0], [0], c="red")
    plt.xlim([-10, 10])
    plt.ylim([-5, 40])
    plt.gca().set_aspect('equal')


def color_cones(centers):
    '''
        assumes that centers is N x 3

        algorithm should check track bounds so that we are not creating
        incorrect predictions
    '''

    # TODO: get a better algorithm for selecting the first point!!!
    # TODO: get a better algorithm for selecting the next point!!!
    # TODO: when performing a scan, should we rotate the centers for a better direction?

    if centers.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    max_angle_diff = np.pi / 3

    # NOTE: these center filtering steps should be center filtering stages
    centers = centers[centers[:, 1] >= 0]

    all_centers = centers
    centers = centers[:, :2]

    N = centers.shape[0]

    # add index to centers
    idxs = np.arange(N).reshape((-1, 1))
    centers = np.hstack([idxs, centers])

    # default colors
    colors = ["nocolor"] * N
    centers_remaining = centers

    # seeding points
    seed_yellow_point = None
    seed_blue_point = None

    # seed the yellow and blue point
    _, right_points = split_by_y(centers_remaining)

    def get_seed(points, centers_remaining, colors, color):
        if points.shape[0] > 0:
            y_avg = np.average(centers_remaining[:, 2])

            # seed is based on distribution of points -- to help with turns
            if y_avg < 4:
                # get closest point by lowest on y-axis
                i = np.argmin(points[:, 2])
            else:
                # get closest point by distance from origin (car)
                # scale points to discourage selecting points far from x-axis
                S = np.array([[2, 0], [0, 1]])
                dists = np.sqrt(np.sum((points[:, 1:3] @ S) ** 2, axis=1))
                i = np.argmin(dists)

            point_curr = points[i, :]
            cidx = int(point_curr[0])
            centers_remaining = centers_remaining[centers_remaining[:, 0] != cidx]

            # angle of approach
            angle = np.pi / 2
            colors[cidx] = color

            return point_curr, centers_remaining, colors
        else:
            return None, centers_remaining, colors

    seed_yellow_point, centers_remaining, colors = get_seed(right_points, centers_remaining, colors, "yellow")

    # YELLOW cone path
    # get closest, right point and update
    if seed_yellow_point is not None:

        # init path search
        point_curr = seed_yellow_point
        angle = np.pi / 2

        while True:
            # get new point
            point_new, angle_new = next_point_simple(
                point_curr, centers_remaining, angle, max_angle_diff=max_angle_diff)
            if point_new is None:
                break

            # update color and state
            cidx = int(point_new[0])
            colors[cidx] = "yellow"

            point_curr = point_new
            angle = angle_new

            # remove from points
            centers_remaining = centers_remaining[centers_remaining[:, 0] != cidx]

    # BLUE cone path
    left_points, _ = split_by_y(centers_remaining)
    seed_blue_point, centers_remaining, colors = get_seed(left_points, centers_remaining, colors, "blue")
    if seed_blue_point is not None:

        # init path search
        point_curr = seed_blue_point
        angle = np.pi / 2

        while True:
            # get new point
            point_new, angle_new = next_point_simple(
                point_curr, centers_remaining, angle, max_angle_diff=max_angle_diff)
            if point_new is None:
                break

            # update color and state
            cidx = int(point_new[0])
            colors[cidx] = "blue"

            point_curr = point_new
            angle = angle_new

            # remove from points
            centers_remaining = centers_remaining[centers_remaining[:, 0] != cidx]

    # create colors as final output
    c2id = {"yellow": 1, "blue": 0, "nocolor": -1}

    color_ids = np.array([c2id[c] for c in colors]).reshape((-1, 1))
    colors = np.array([C2RGB[c] for c in colors]) / 255

    # plot_dir(point_curr[1:], angle)
    # plot_dir(point_curr[1:], angle - np.pi / 3.5)
    # plot_dir(point_curr[1:], angle + np.pi / 3.5)
    # plot(all_centers[:, :2], colors)
    # plt.show()

    cone_output = np.hstack([all_centers[:, :2], color_ids])
    return cone_output, all_centers, colors
