from setuptools import find_packages
from setuptools import setup


setup(
    name='morphing_agents',
    version='0.1',
    description='A Collecting Of Dynamic Morphology Agents',
    include_package_data=True,
    packages=find_packages(),
    install_requires=['gym', 'numpy', 'mujoco_py', 'robel'])
