from numpy_ros import to_numpy, to_message
from geometry_msgs.msg import Point


point = Point(3.141, 2.718, 42.1337)

# ROS to NumPy
as_array = to_numpy(point)


# NumPy to ROS
message = to_message(Point, as_array)