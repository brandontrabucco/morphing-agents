from setuptools import setup


INSTALL_REQUIRES = [
    'gym',
    'numpy',
    'mujoco_py',
    'robel']


CLASSIFIERS = [
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8']


setup(
    name='morphing-agents',
    packages=['morphing_agents'],
    version='1.0',
    license='MIT',
    description='Collection Of Dynamic Morphology Agents For MuJoCo',
    author='Brandon Trabucco',
    author_email='brandon@btrabucco.com',
    url='https://github.com/brandontrabucco/morphing-agents',
    download_url='https://github.com/brandontrabucco/design-bench/archive/v1.tar.gz',
    keywords=['MuJoCo', 'Agents', 'Dynamic Morphology'],
    install_requires=INSTALL_REQUIRES,
    classifiers=CLASSIFIERS)
