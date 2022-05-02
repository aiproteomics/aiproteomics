#!/usr/bin/env python
import os
from setuptools import setup
from sympy import re



# see setup.cfg
setup(install_requires=open("requirements.txt", "r").readlines())
