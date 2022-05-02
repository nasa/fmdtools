# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:58:33 2022

@author: dhulse
"""
from ex_pump import Pump
import sys, os
sys.path.append(os.path.join('..'))


from fmdtools.modeldef import *
import fmdtools.resultdisp as rd
import fmdtools.faultsim.propagate as propagate


mdl = Pump()

# nominal, single-fault, mult-fault
endresults, resgraph, mdlhist=propagate.one_fault(mdl, 'ExportWater','block', time=20, staged=False, run_stochastic=True, modelparams={'seed':10})


# pickle, CSV, json
mdlhist_flattened = rd.process.flatten_hist(mdlhist)
rd.process.save_result(mdlhist, "single_fault.pkl")


mdlhist_saved = rd.process.load_result("single_fault.pkl")
mdlhist_saved_flattened = rd.process.flatten_hist(mdlhist_saved)

mdlhist_saved['faulty']['time'][0]=100
mdlhist['faulty']['time']==mdlhist_saved['faulty']['time']

# nominal approach

# resilience approach

# nested approach