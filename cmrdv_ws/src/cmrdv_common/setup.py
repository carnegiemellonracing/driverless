import os
import glob
from setuptools import setup, find_packages

package_name = 'cmrdv_common'
share_directory = os.path.join('share', package_name)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ankit',
    maintainer_email='akhandelwal2025@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "perceptions = cmrdv_common.heartbeat.heart:perceptions",
            "planning = cmrdv_common.heartbeat.heart:planning",
            "heartbeat = cmrdv_common.heartbeat.global:main", 
            "dim = cmrdv_common.heartbeat.heart:dim", 
            "dim_heartbeat = cmrdv_common.DIM.dim_heartbeat:main",
            "dim_request = cmrdv_common.DIM.dim_request:main",
            "sim_vis = cmrdv_common.simulator.SimVisNode:main"
        ],
    },
)
