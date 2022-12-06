#!/usr/bin/env python
from setuptools import setup


# see setup.cfg
setup(install_requires=open("requirements.txt", "r").readlines())
