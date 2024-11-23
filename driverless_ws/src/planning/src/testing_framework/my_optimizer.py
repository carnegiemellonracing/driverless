"""This file is where you store your optimizer"""
import numpy as np

def point_along_line(blue_point, yellow_point, distance):
    # vector from yellow to blue
    direction_vector = np.array(blue_point) - np.array(yellow_point)
    unit_vector = direction_vector / np.linalg.norm(direction_vector)
    target_point = np.array(yellow_point) + distance * unit_vector
    return tuple(target_point)

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
    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    return [points_x, points_y]

    #return [np.array([1,2,3]),np.array([1,2,3])]

# b_x = [1,2,3,4,5]
# b_y = [4,2,0,2,4]
# y_x = [1,2,3,4,5]
# y_y = [0,-2,-4,-2,0]
# run_optimizer(b_x,b_y,y_x,y_y)