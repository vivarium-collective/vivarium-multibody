import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

setup(
    name='vivarium-multibody',
    version='0.0.10',
    packages=[
        'vivarium_multibody',
        'vivarium_multibody.plots',
        'vivarium_multibody.library',
        'vivarium_multibody.processes',
        'vivarium_multibody.composites',
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
        'vivarium-core>=0.2.0',
        'pytest',
        'pymunk',
    ],
)
