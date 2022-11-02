#!/usr/bin/env python
import os
from setuptools import setup

# see setup.cfg
setup(install_requires=open("requirements.txt", "r").readlines())
