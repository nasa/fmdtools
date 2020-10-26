# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:14:23 2020

@author: danie
"""

from drone_opt import *
import fmdtools.faultsim.propagate as propagate

mdl_split = x_to_mdl([1,1,55,2,0], loc='rural')
mdl_med = x_to_mdl([0,1,55,2,0], loc='rural')
mdl_med.params['respolicy']

#testing mechanical fault response
#endresults_med, resgraph, mdlhist_med = propagate.one_fault(mdl_med,'AffectDOF', 'RFmechbreak', time=6)
#rd.plot.mdlhistvals(mdlhist_med, fxnflowvals = {'RSig_Traj':'mode', 'Planpath':'mode', 'DOFs': ['planvel','uppwr', 'planpwr'], 'AffectDOF':['LRstab', 'FRstab']}, time=6)
#plot_faulttraj(mdlhist_med, mdl_med.params)
#plot_xy(mdlhist_med['faulty'], endresults_med)

#testing battery fault response
#endresults_med, resgraph, mdlhist_med = propagate.one_fault(mdl_med,'StoreEE', 'S1P1nocharge', time=4)
endresults_med, resgraph, mdlhist_med = propagate.one_fault(mdl_med,'StoreEE', 'S1P1nocharge', time=6)
rd.plot.mdlhistvals(mdlhist_med, fxnflowvals = {'RSig_Traj':'mode', 'Planpath':'mode', 'DOFs': 'planvel', 'StoreEE':'soc'}, time=4, legend=False)
plot_faulttraj(mdlhist_med, mdl_med.params)
plot_xy(mdlhist_med['faulty'], endresults_med)

#testing battery fault response
#endresults_med, resgraph, mdlhist_med = propagate.one_fault(mdl_med,'StoreEE', 'S1P1nocharge', time=4)
endresults_split, resgraph, mdlhist_split = propagate.one_fault(mdl_split,'StoreEE', 'S1P1nocharge', time=6)
rd.plot.mdlhistvals(mdlhist_split, fxnflowvals = {'RSig_Traj':'mode', 'Planpath':'mode', 'DOFs': 'planvel', 'StoreEE':'soc'}, time=4, legend=False)
plot_faulttraj(mdlhist_split, mdl_split.params)
plot_xy(mdlhist_split['faulty'], endresults_split)


fhist=mdlhist_med['faulty']
faulttime = sum([any([fhist['functions'][f]['faults'][t]!={'nom'} for f in fhist['functions']]) for t in range(len(fhist['time'])) if fhist['flows']['DOFs']['elev'][t]])

app_med = SampleApproach(mdl_med, faults='single-component', phases={'forward'})
endclasses_med, mdlhists = propagate.approach(mdl_med, app_med, staged=True)
simplefmea_med = rd.tabulate.simplefmea(endclasses_med)

app_split = SampleApproach(mdl_split, faults='single-component', phases={'forward'})
endclasses_split, mdlhists = propagate.approach(mdl_split, app_split, staged=True)
simplefmea_split = rd.tabulate.simplefmea(endclasses_split)