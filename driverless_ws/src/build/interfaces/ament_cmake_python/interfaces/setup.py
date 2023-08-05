from setuptools import find_packages
from setuptools import setup

setup(
    name='interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('interfaces', 'interfaces.*')),
)
