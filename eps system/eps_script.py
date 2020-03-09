# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:46:24 2020

@author: hulsed
"""
import sys
sys.path.append('../')
from eps import EPS
import fmdtools.faultprop as fp
import fmdtools.resultproc as rp

mdl= EPS()
rp.show_bipartite(mdl.bipartite)
