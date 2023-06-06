from setuptools import setup

package_name = 'cmrdv_actuators'

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
            "fsm = cmrdv_actuators.fsm_node:main",
            "brake = cmrdv_actuators.brakes_node:main",
            "steering = cmrdv_actuators.steering_node:main"
        ],
    },
)
