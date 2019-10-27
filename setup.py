#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='research_seed',
    version='0.0.1',
    description='Describe Your Cool Project',
    url='https://github.com/toshas/GatedCRFLoss',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=requirements,
    packages=find_packages()
)
