from eufs_msgs.msg import ConeArray as ConeArray
from geometry_msgs.msg import Point
import json

def pointList(cones): 
    lst = []
    for cone in cones:
        p = Point()
        p.x = float(cone["x"])
        p.y = float(cone["y"])
        p.z = float(cone["z"])
        lst.append(p)
    return lst

def read():
    f = open("ex.json")
    json_data = json.load(f)
    f.close()
    blue_cones = json_data["blue_cones"]
    yellow_cones = json_data["yellow_cones"]
    cones = ConeArray()
    cones.blue_cones = pointList(blue_cones)
    cones.yellow_cones = pointList(yellow_cones)
    return cones

if __name__ ==  '__main__':
    print(read())