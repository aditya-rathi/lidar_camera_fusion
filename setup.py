## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['models','models.correlation_package', 'lidar_camera_fusion'],
    package_dir={'': 'include'})

setup(**setup_args)