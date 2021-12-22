# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:16:22 2021

@author: dhulse
"""
import unittest
import os

loader = unittest.TestLoader()
suite = loader.discover(os.getcwd())

runner = unittest.TextTestRunner()
runner.run(suite)