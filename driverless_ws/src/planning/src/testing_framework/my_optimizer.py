import numpy as np

# find cublic spline given 2 points
def cubic_spline_least_squares(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    A = np.array([
        [x1**3, x1**2, x1, 1],
        [x2**3, x2**2, x2, 1]
    ])
    y = np.array([y1, y2])

    coefficients, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return tuple(coefficients)

# interpolate points given a cubic spline
def interpolate_cubic_spline(coefficients, x_values):
    a, b, c, d = coefficients
    y_values = [a * x**3 + b * x**2 + c * x + d for x in x_values]
    return y_values

# calculate point distance d from yellow point
def point_along_line(blue_point, yellow_point, distance):
    # vector from yellow to blue
    direction_vector = np.array(blue_point) - np.array(yellow_point)
    unit_vector = direction_vector / np.linalg.norm(direction_vector)
    target_point = np.array(yellow_point) + distance * unit_vector
    return tuple(target_point)

# find initial three points
def initial_points(blue_cones, yellow_cones, d1, d2):
    blue_start = blue_cones[0]
    yellow_start = yellow_cones[0]
    blue_mid = blue_cones[len(blue_cones)//2]
    yellow_mid = yellow_cones[len(yellow_cones)//2]
    blue_end = blue_cones[-1]
    yellow_end = yellow_cones[-1]

    points = []
    points.append(((blue_start[0]+yellow_start[0])/2,(blue_start[1]+yellow_start[1])/2))
    
    #points.append(((blue_mid[0]+yellow_mid[0])/2,(blue_mid[1]+yellow_mid[1])/2))
    second_point = point_along_line(blue_mid, yellow_mid, d1)
    points.append(second_point)

    #points.append(((blue_end[0]+yellow_end[0])/2,(blue_end[1]+yellow_end[1])/2))
    third_point = point_along_line(blue_end, yellow_end, d2)
    points.append(third_point)

    return points

"""change this function, this is where you implement your raceline optimizer"""
def run_optimizer(blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y):
    blue_c = list(zip(blue_cones_x, blue_cones_y))
    yellow_c = list(zip(yellow_cones_x, yellow_cones_y))

    # sort blue and yellow cones by x value
    blue_cones = sorted(blue_c, key=lambda cone: cone[0])
    yellow_cones = sorted(yellow_c, key=lambda cone: cone[0])

    # extract initial points
    # 2 and 4 are d1 and d2 respectively
    points = initial_points(blue_cones,yellow_cones,2,4)

    # test initial points
    # points_x = [point[0] for point in points]
    # points_y = [point[1] for point in points]
    # return [points_x, points_y]

    # find cubic splines between initial points
    spline1 = cubic_spline_least_squares(points[0],points[1])
    spline2 = cubic_spline_least_squares(points[1],points[2])

    # interpolate points on the spline
    spline1_x = np.linspace(blue_cones[0][0], blue_cones[len(blue_cones)//2][0], num=100)
    spline1_y = interpolate_cubic_spline(spline1,spline1_x)
    spline2_x = np.linspace(blue_cones[len(blue_cones)//2][0], blue_cones[-1][0], num=100)
    spline2_y = interpolate_cubic_spline(spline2,spline2_x)

    return [np.append(spline1_x,spline2_x),np.append(spline1_y,spline2_y)]

# b_x = [1,2,3,4,5]
# b_y = [4,2,0,2,4]
# y_x = [1,2,3,4,5]
# y_y = [0,-2,-4,-2,0]
# run_optimizer(b_x,b_y,y_x,y_y)