from setuptools import find_packages, setup

package_name = 'python_pcl_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aryalohia',
    maintainer_email='aryalohia00@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cmr_cpp_pipline_node = python_pcl_pipeline.cmr_cpp_pipeline:main',
        ],
    },
)
