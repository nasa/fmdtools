# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:13:21 2019

@author: Daniel Hulse
"""
import cProfile
import pstats

#%run -m cProfile -o speedtest quad_script.py

p=pstats.Stats('speedtest3')
p.sort_stats('tottime').print_stats(100)