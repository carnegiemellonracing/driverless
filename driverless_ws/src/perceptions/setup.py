from setuptools import setup
from setuptools import find_namespace_packages

from glob import glob
import os

package_name = 'perceptions'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name] + find_namespace_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dale',
    maintainer_email='geoffthejetson@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # raw data nodes
            'zed_node = perceptions.ros.utils.ZEDNode:main_zed',
            'zed2_node = perceptions.ros.utils.ZEDNode:main_zed2',

            # debugging nodes
            'lidar_vis_node = perceptions.ros.utils.debug.LidarVisNode:main',

            # util nodes
            'data_node = perceptions.ros.utils.DataNode:main',
            'sync_node = perceptions.ros.utils.SyncNode:main',
            'file_node = perceptions.ros.utils.FileNode:main',

            # predictor nodes
            'yolov5_zed_node = perceptions.ros.predictors.YOLOv5Node:main_zed',
            'yolov5_zed2_node = perceptions.ros.predictors.YOLOv5Node:main_zed2',
            'lidar_node = perceptions.ros.predictors.LidarNode:main',

            # cone node
            'cone_node = perceptions.ros.utils.ConeNode:main',

            # midline node
            'midline_node = perceptions.planning_stuff.MidlineNode:main'
        ],
    },
)
