import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.frenet as frenet
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.path_optimization import PathOptimizer
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.raceline import reverse_transformation
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.utils import *

BLUE = 0
YELLOW = 1

### UTILITY FUNCTIONS

def pretty_print(rows, columns):
    # Convert to string
    for i in range(len(rows)):
        rows[i] = list(map(str, rows[i]))

    # Get max for each column
    lengths = list(map(len, columns))
    for row in rows:
        for i in range(len(row)):
            lengths[i] = max(lengths[i], len(row[i]))

    # Print header
    print('-' * (sum(lengths) + len(columns) + 1))
    for i in range(len(columns)):
        print('|' + columns[i] + ' ' * (lengths[i] - len(columns[i])), end='')
    print('|')
    
    # Print header separator
    print('=' * (sum(lengths) + len(columns) + 1))

    # Print rows
    for row in rows:
        for i in range(len(row)):
            print('|' + row[i] + ' ' * (lengths[i] - len(row[i])), end='')
        print('|')

    # Close table
    print('-' * (sum(lengths) + len(columns) + 1))


def get_points(FILE = "fsdv/path_planning/raceline/data/test_tracks/Austin.csv"):
    track = np.genfromtxt(FILE, delimiter=",")

    return track

def get_cones(points):
    # cones description (sample values)
    # x_m | y_m | w_tr_right_m | w_tr_left_m
    # --------------------------------------
    # 0.9   4.02         7.565         7.361
    # 4.9   0.98         7.584         7.382

    xm = points[:, 0]
    ym = points[:, 1]
    right = points[:, 2]
    left = points[:, 3]

    lcones = np.array([xm - left, ym, BLUE * np.ones(ym.shape)]).T
    rcones = np.array([xm + right, ym, YELLOW * np.ones(ym.shape)]).T

    return lcones, rcones

### Graphical tests

def midpoints_graphical_test():
    print('\n### PATH PLANNING GRAPHICAL TEST - Midpoints interpolation ###\n')

    points = get_points()
    points = points[::2, :] # filter points to play with less data

    lcones, rcones = get_cones(points)

    interpolated_points, _ = generate_centerline_from_cones(lcones, rcones, 10)

    # Graphically checking interpolation:

    plt.close()

    plt.scatter(rcones[:, 0], rcones[:, 1], color='blue')
    plt.scatter(lcones[:, 0], lcones[:, 1], color='red')

    plt.scatter(interpolated_points[:, 0], interpolated_points[:, 1])


    plt.show()

def curvature_to_progress_graphical_test():
    print('\n### PATH PLANNING GRAPHICAL TEST - Curvature to progress ###\n')

    points = get_points()
    points = points[::5, :] # filter points to play with less data

    lcones, rcones = get_cones(points)

    _, generator = generate_centerline_from_cones(lcones, rcones, 5)

    path = generator.cumulated_splines
    cumulative_lengths = generator.cumulated_lengths

    total_length = cumulative_lengths[-1]

    progress_steps = np.linspace(0, total_length, 300)
    curv_points = []
    curvatures = []

    print('Computing curvatures...\n')

    last_index = 0
    n = len(progress_steps)
    for i in range(n):
        if (i+1)%100 == 0:
            print(f'{i+1}/{n}')

        progress = progress_steps[i]

        point, spline, local_point, exact, last_index, _ = interpolate_raceline(path, cumulative_lengths, progress, last_index)

        tol = 4

        if abs(progress - exact) > tol:
            exit('Exact progress too far from target progress')

        curvature = frenet.get_curvature(
            spline.first_der, spline.second_der, local_point[0])

        progress_steps[i] = exact # fix progress step with exact value
        curv_points.append(point[0])
        curvatures.append(curvature)
    print('')

    curv_points = np.array(curv_points)
    curvatures = np.array(curvatures)

    marker_period = 150
    marker_color = "slategray"

    marker_points = curv_points[::marker_period]
    marker_progress_steps = progress_steps[::marker_period]
    marker_curvatures = curvatures[::marker_period]

    plt.close()

    # Showing path with curvature information
    plt.subplot(2, 1, 1)
    
    color_map = plt.cm.get_cmap("cool")
    size = mpl.rcParams['lines.markersize'] ** 1.8

    sc = plt.scatter(curv_points[:, 0], curv_points[:, 1], c=curvatures, vmin=np.min(curvatures), vmax=np.max(curvatures), cmap=color_map, s=size)
    plt.colorbar(sc)

    plt.scatter(marker_points[:, 0], marker_points[:, 1], c=marker_color, s=size)

    # Graphically showing curvature as a function of progress
    plt.subplot(2, 1, 2)
    plt.plot(progress_steps, curvatures)

    plt.plot(marker_progress_steps, marker_curvatures, linestyle="None", c=marker_color, marker="o")

    plt.show()

def path_optimization_graphical_test():    
    points = get_points()
    points = points[::5, :] # filter points to play with less data

    lcones, rcones = get_cones(points)

    _, generator = generate_centerline_from_cones(lcones, rcones)

    reference_path = generator.cumulated_splines
    cumulative_lengths = generator.cumulated_lengths

    # Generate optimal path using interpolation as reference path

    path_optimizer = PathOptimizer(reference_path, cumulative_lengths, delta_progress=16)

    solution, torch_solution = path_optimizer.optimize()

    states = solution[:, :8] # (progress_steps, n, mu, vx, vy, r, delta, T)
    controls = solution[:, 8:]

    pretty_print(solution.tolist(), ['progress', 'n', 'mu', 'vx', 'vy', 'r', 'delta', 'T', 'ddelta', 'dT'])

    print("- Constraints")

    for constraint in path_optimizer.constraints:
        f = constraint['fun']
        print("-- " + constraint['name'])
        print(f(torch_solution).detach().numpy())
        print('')

    points = []
    on_lines = []

    last_index = None
    progress_list = np.clip(states[:, 0], 0, None)
    n_list = states[:, 1]

    n = len(progress_list)

    for i in range(n):
        if (i+1)%100 == 0:
            print(f'{i+1}/{n}')

        progress = progress_list[i]
        normal = n_list[i]

        on_line, spline, local_point, exact, last_index, x = interpolate_raceline(reference_path, cumulative_lengths, progress, last_index)

        deriv = spline.first_der(x) # gradient at the point
        tangent = [1, deriv]
        normal_direction = np.array([-tangent[1], tangent[0]])
        normal_direction /= np.linalg.norm(normal_direction)

        shifted = local_point + normal * normal_direction

        point = reverse_transformation(np.array([shifted]), spline.Q, spline.translation_vector)

        progress_list[i] = exact # fix progress step with exact value
        points.append(point[0])
        on_lines.append(on_line[0])
    print('')

    points = np.array(points)
    on_lines = np.array(on_lines)

    reference_points = np.array([spline.points for spline in reference_path])
    reference_points = reference_points.reshape((-1, 2)) # remove points from their spline groups

    plt.close() 

    plt.scatter(on_lines[:, 0], on_lines[:, 1], c='yellow', marker='o')
    plt.scatter(points[:, 0], points[:, 1], c='orange', marker='x')
    plt.scatter(reference_points[:, 0], reference_points[:, 1])

    #plt.scatter(rcones[:, 0], rcones[:, 1], color='blue')
    #plt.scatter(lcones[:, 0], lcones[:, 1], color='red')

    plt.show()

### TESTS

#midpoints_graphical_test()
#curvature_to_progress_graphical_test()
path_optimization_graphical_test()
