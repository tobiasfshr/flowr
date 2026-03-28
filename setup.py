"""Package installation setup."""
from setuptools import setup, find_packages


setup(
    name='flowr',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    description="FlowR: Flowing from Sparse to Dense 3D Reconstructions",
    author='Tobias Fischer',
    author_email='tobias.fischer@inf.ethz.ch',
    license='Apache 2.0',
)
