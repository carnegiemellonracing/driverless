from setuptools import setup

package_name = 'cmrdv_planning'

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
    maintainer='ankit',
    maintainer_email='akhandelwal2025@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
            # 'midpoint=cmrdv_planning.midpoint:main',
            # 'SLAM=cmrdv_planning.SLAM:main',
            # 'optimizer=cmrdv_planning.optimizer:main',
            'SLAM_test=cmrdv_planning.SLAM_Validation:main'
        ],
    },
)
