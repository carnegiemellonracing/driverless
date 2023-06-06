from setuptools import setup, find_packages

package_name = 'cmrdv_perceptions'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'stereo_vision = cmrdv_perceptions.StereoCamNode:main',
            'lidar = cmrdv_perceptions.LidarNode:main',
            'visualization = cmrdv_perceptions.CVVisNode:main',
            'numpy_test_pub = cmrdv_perceptions.numpy_test_pub:main',
            'numpy_test_sub = cmrdv_perceptions.numpy_test_sub:main'
        ],
    },
)
