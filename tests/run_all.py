# -*- coding: utf-8 -*-
"""
Use this script to run all tests in the repository.
"""
import unittest
import os, sys

if __name__=="__main__":
    import time
    t=time.time()
    loader = unittest.TestLoader()
    suite = loader.discover(os.getcwd())
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
    #runner.run(suite._tests[1])
    #runner.run(suite._tests[5])
    
    # Run defined tests in the example repositories 
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join('..', 'example_pump'))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
    t_end=time.time()-t
    print(t_end)

