from geometry_msgs.msg import Point


def np2points(cones):
    '''convert list of cones into a Point[] object'''
    arr = []
    for i in range(len(cones)):
        p = Point()
        p.x = float(cones[i][0])
        p.y = float(cones[i][1])
        p.z = float(cones[i][2])

        arr.append(p)
    return arr
