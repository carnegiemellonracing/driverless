from setuptools import setup

package_name = 'perceptions'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
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
            'data_node = perceptions.utils.DataNode:main',
            'stereo_node = perceptions.predict.StereoNode:main',
            'lidar_node = perceptions.predict.LidarNode:main'
        ],
    },
)
