# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:17:31 2020

@author: danie
"""

import sys

sys.path.append('../../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_mdl import *
from drone_opt import *
import time
import pandas as pd
import numpy as np



#explore_tradoffs(loc='rural')
plot_tradeoffs('grid_results_rural.csv')
