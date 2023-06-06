from setuptools import setup

package_name = 'cmrdv_collection'

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
            'data_collection = cmrdv_collection.collection_node:main',
            'sim_pub = cmrdv_collection.sim_collection_node:main',
            'test_pts = cmrdv_collection.test_sub:main',
        ],
    },
)
