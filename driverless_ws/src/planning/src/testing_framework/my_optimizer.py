"""This file is where you store your optimizer"""
import numpy as np

def initial_points(blue_cones,yellow_cones):
    points = []
    points.append(((blue_cones[0][0]+yellow_cones[0][0])/2,(blue_cones[0][1]+yellow_cones[0][1])/2))
    points.append(((blue_cones[len(blue_cones)//2][0]+yellow_cones[len(yellow_cones)//2][0])/2,(blue_cones[len(blue_cones)//2][1]+yellow_cones[len(yellow_cones)//2][1])/2))
    points.append(((blue_cones[-1][0]+yellow_cones[-1][0])/2,(blue_cones[-1][1]+yellow_cones[-1][1])/2))
    return points

"""change this function, this is where you implement your raceline optimizer"""
def run_optimizer(blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y):
    blue_cones = list(zip(blue_cones_x, blue_cones_y))
    yellow_cones = list(zip(yellow_cones_x, yellow_cones_y))
    points = initial_points(blue_cones,yellow_cones)
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    print(p1)
    print(p2)
    print(p3)
    return [np.array(points[0]), np.array(points[1])]

b_x = [1,2,3,4,5]
b_y = [4,2,0,2,4]
y_x = [1,2,3,4,5]
y_y = [0,-2,-4,-2,0]
run_optimizer(b_x,b_y,y_x,y_y)