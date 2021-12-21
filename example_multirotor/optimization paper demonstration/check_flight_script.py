# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:23:55 2020

@author: Daniel Hulse
"""

from drone_opt import *
import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp.graph as graph

mdl_med = x_to_mdl([1,1,80,1,1])
calc_oper(mdl_med)
endresults_med, resgraph, mdlhist_med =propagate.nominal(mdl_med)
plot_xy(mdlhist_med, endresults_med, title='Flight plan at 80 m')
app_med = SampleApproach(mdl_med, faults='single-component', phases={'forward'})
faulttime_med = app_med.times
landtime_med = min([i for i,a in enumerate(mdlhist_med['functions']['Planpath']['mode']) if a=='taxi'])
rcost_med = calc_res(mdl_med)
rcost_med2 = x_to_rcost([1,1],[80],[1,1])

mdl_low = x_to_mdl([1,1,50,1,1])
calc_oper(mdl_low)
endresults_low, resgraph, mdlhist_low =propagate.nominal(mdl_low)
plot_xy(mdlhist_low, endresults_low, title='Flight plan at 50 m')
app_low = SampleApproach(mdl_low, faults='single-component', phases={'forward'})
faulttime_low = app_low.times
landtime_low = min([i for i,a in enumerate(mdlhist_low['functions']['Planpath']['mode']) if a=='taxi'])
rcost_low = calc_res(mdl_low)
rcost_low2 = x_to_rcost([1,1],[50],[1,1])

mdl_hi = x_to_mdl([1,1,180,1,1])
calc_oper(mdl_hi)
endresults_hi, resgraph, mdlhist_hi =propagate.nominal(mdl_hi)
plot_xy(mdlhist_hi, endresults_hi, title='Flight plan at 180 m')
app_hi = SampleApproach(mdl_hi, faults='single-component', phases={'forward'})
faulttime_hi = app_hi.times
landtime_hi = min([i for i,a in enumerate(mdlhist_hi['functions']['Planpath']['mode']) if a=='taxi'])
rcost_hi = calc_res(mdl_hi)
rcost_hi2 = x_to_rcost([1,1],[180],[1,1])

plot_xys({'80 m': mdlhist_med , '50 m':mdlhist_low , '180 m': mdlhist_hi}, {'80 m': endresults_med , '50 m':endresults_low , '180 m': endresults_hi})


