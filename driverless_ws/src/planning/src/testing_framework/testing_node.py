import readfile
import driverless_ws.src.planning.testing_framework.midlineplot as midlineplot
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from eufs_msgs.msg import ConeArray as ConeArray
from geometry_msgs.msg import Point


blue_cones = []
yellow_cones = []
plotnum = 0
class testingnode(Node):

    def _init_(self):
        super()._init_("testingnode")
        self.subscription = self.create_subscription(Point,'FunctionPublisher',self.listener_callback,10)
        self.publisher = self.create_publisher(ConeArray, 'FunctionSubscriber', 10)

        self.publishData()
    
    def midlinecallback(self, msg):
        plotnum += 1
        midlineplot.plot_lin_input(blue_cones,yellow_cones,msg,plotnum)
        midlineplot.plot_cub_input(blue_cones,yellow_cones,msg,plotnum)

    def publishData(self):
        cone_arr = readfile.read()
        blue_cones = cone_arr.blue_cones
        yellow_cones = cone_arr.yellow_cones
        self.publisher.publish(cone_arr)




        


def main(args=None):
    rclpy.init(args=args)
    closed_loop_node = testingnode()
    rclpy.spin(closed_loop_node)
    closed_loop_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()