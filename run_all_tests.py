# -*- coding: utf-8 -*-
"""
Use this scripts to run all tests in the repository.
"""
import unittest
import os
from tests import CommonTests

loader = unittest.TestLoader()
suite = loader.discover(os.getcwd())

runner = unittest.TextTestRunner()
runner.run(suite)