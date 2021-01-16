import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

setup(
    name='vivarium-pymunk',
    version='0.0.1',
    packages=[
        'vivarium_pymunk',
        'vivarium_pymunk.processes',
        'vivarium_pymunk.composites',
        'vivarium_pymunk.experiments',
    ],
    author='Eran Agmon',
    author_email='eagmon@stanford.edu',
    url='https://github.com/vivarium-collective/vivarium-pymunk',
    license='MIT',
    entry_points={
        'console_scripts': []},
    short_description='a vivarium wrapper for the pymunk physics engine',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={},
    include_package_data=True,
    install_requires=[
        'vivarium-core',
        'pytest',
        'pymunk',
        'alphashape',
    ],
)
