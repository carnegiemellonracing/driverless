from setuptools import setup
from setuptools import find_namespace_packages

package_name = 'perceptions'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name] + find_namespace_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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

            # util nodes
            'data_node = perceptions.ros.utils.DataNode:main',
            'sync_node = perceptions.ros.utils.SyncNode:main',

            # predictor nodes
            'yolov5_node = perceptions.ros.predictors.YOLOv5Node:main',
            'lidar_node = perceptions.ros.predictors.LidarNode:main'
        ],
    },
)
