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
def interpolate_cubic(coefficients, x_values):
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

    # first, second, and third points
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

    # use only if points are sorted
    points = initial_points(blue_c,yellow_c,d1=2,d2=2.5)

    # sort blue and yellow cones by x value
    # use if cones are not sorted
    blue_cones = sorted(blue_c, key=lambda cone: cone[1])
    yellow_cones = sorted(yellow_c, key=lambda cone: cone[1])

    # extract initial points
    # 2 and 4 are d1 and d2 respectively
    # points = initial_points(blue_cones,yellow_cones,d1=2,d2=2.5)
    # test initial points
    # print(points)
    # points_x = [point[0] for point in points]
    # points_y = [point[1] for point in points]
    # return [points_x, points_y]
    
    # find k and l by taking slope from first and last couple points
    k = (blue_cones[0][1]-blue_cones[1][1])/(blue_cones[0][0]-blue_cones[1][0])
    l = (blue_cones[-1][1]-blue_cones[-2][1])/(blue_cones[-1][0]-blue_cones[-2][0])
    # print(k)
    # print(l)

    # set inital points
    x1,y1 = points[0]
    x2,y2 = points[1]
    x3,y3 = points[2]

    # input matrix for x and y values
    b_x = np.array([x1,x2,x2,x3,k,l,0,0])
    b_y = np.array([y1,y2,y2,y3,k,l,0,0])
    # print(b_x)
    # print(b_y)

    # hardcode inverse matrix
    A_inverse = np.array([
        [10, -10, -6, 6, 3, -1, 2, 0.25],
        [-9, 9, 3, -3, -3.5, 0.5, -1, -0.125],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [-6, 6, 10, -10, -1, 3, -2, 0.25],
        [6, -6, -6, 6, 1, -1, 2, -0.25],
        [-1.5, 1.5, -1.5, 1.5, -0.25, -0.25, -0.5, 0.0625],
        [0, 0, 1, 0, 0, 0, 0, 0]
    ])

    # solve large matrix equation
    X = A_inverse @ b_x
    Y = A_inverse @ b_y

    # set coefficients
    a1x, b1x, c1x, d1x, a2x, b2x, c2x, d2x = X
    a1y, b1y, c1y, d1y, a2y, b2y, c2y, d2y = Y

    # dump coefficients into np array
    spline_x1 = np.array([a1x,b1x,c1x,d1x])
    spline_x2 = np.array([a2x,b2x,c2x,d2x])
    spline_y1 = np.array([a1y,b1y,c1y,d1y])
    spline_y2 = np.array([a2y,b2y,c2y,d2y])

    # set base values (500 points each)
    base = np.linspace(0,0.5,num=500)

    # outputs of cubic function
    splinex1 = interpolate_cubic(spline_x1,base)
    splinex2 = interpolate_cubic(spline_x2,base)
    spliney1 = interpolate_cubic(spline_y1,base)
    spliney2 = interpolate_cubic(spline_y2,base)

    # combine the arrays
    spline_x = np.append(splinex1,splinex2)
    spline_y = np.append(spliney1,spliney2)

    return [spline_x,spline_y]