# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:08:05 2020

@author: Daniel Hulse
"""

import sys
sys.path.append('../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
from tank_model import Tank
from fmdtools.modeldef import SampleApproach


# Nominal Run - nothing happens
mdl = Tank()

endresults, resgraph, mdlhist = propagate.nominal(mdl)

rd.plot.mdlhistvals(mdlhist)
rd.graph.show(resgraph)

# Faulty Run - nothing happens b/c no fault
endresults, resgraph, mdlhist = propagate.one_fault(mdl,'Human','NotVisible', time=2)

rd.plot.mdlhistvals(mdlhist, fault='NotVisible', time=2)
rd.graph.show(resgraph,faultscen='NotVisible', time=2)

endresults, resgraph, mdlhist = propagate.one_fault(mdl,'Human','FalseReach', time=2, gtype='component')

rd.plot.mdlhistvals(mdlhist,fault='FalseReach',time=2)
rd.graph.show(resgraph,gtype='bipartite',faultscen='FalseReach', time=2)





#import matplotlib.pyplot as plt
#plt.figure()
#reshist, diff, summary = rp.compare_hist(mdlhist)
#rp.plot_resultsgraph_from(mdl,reshist,time=20)

endclasses, mdlhists = propagate.single_faults(mdl)


app_stuck = SampleApproach(mdl, faults=[('Import_Water', 'Stuck')])

endresults, resgraph, mdlhist = propagate.one_fault(mdl,'Import_Water','Stuck', time=2)

app_full = SampleApproach(mdl)
endclasses, mdlhists = propagate.approach(mdl, app_full)


