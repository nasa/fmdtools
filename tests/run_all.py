# -*- coding: utf-8 -*-
"""
Use this scripts to run all tests in the repository.
"""
import unittest
import os, sys

loader = unittest.TestLoader()
suite = loader.discover(os.getcwd())

runner = unittest.TextTestRunner()
runner.run(suite)

# Run defined tests in the example repositories 
loader = unittest.TestLoader()
suite = loader.discover(os.path.join('..', 'example_pump'))

runner = unittest.TextTestRunner()
runner.run(suite)

