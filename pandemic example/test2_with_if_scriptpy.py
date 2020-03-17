# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:51:29 2020

@author: Daniel Hulse
"""

import sys
sys.path.append('../')
import pandas as pd
import faultprop as fp
import resultproc as rp
import csv
import numpy as np
from scipy.optimize import minimize
import csv
from test2_with_if import DiseaseModel

x0 = np.array([0.1,3,0.5,10,0.2,2])

dm1 = DiseaseModel(x0)
    
    
    

rp.show_graph(dm1.graph)

endresults, resgraph, mdlhist_nom = fp.run_nominal(dm1)

rp.plot_mdlhist(mdlhist_nom, fxnflows=['Campus'])

normal_state_table = rp.make_histtable(mdlhist_nom)
normal_state_table.to_csv('normal_state_table.csv')