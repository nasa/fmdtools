# -*- coding: utf-8 -*-
"""
File name: quad_script.py
Author: Daniel Hulse
Created: June 2019
Description: I/O with the quadrotor model defined in quad_mdl.py
"""
import sys
sys.path.append('../')

import fmdtools.faultprop as fp
import fmdtools.resultproc as rp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from quad_mdl import *
import time

#scenlist=fp.listinitfaults(graph, mdl.times)
mdl = Quadrotor()


app = SampleApproach(mdl)
endclasses, mdlhists = fp.run_approach(mdl, app)
simplefmea = rp.make_simplefmea(endclasses)
summfmea = rp.make_summfmea(endclasses, app)

# =============================================================================
#endresults1, resgraph, flowhist3, ghist3=fp.runnominal(mdl, track={'DOFs','Dir1', 'Env1', 'Force_LG'})
#fp.showgraph(resgraph, showfaultlabels=False)
#fp.plotflowhist(flowhist3, 'N/A', time=0)
## 
## #Check various scenarios individually
## 
endresults, resgraph, mdlhist = fp.run_one_fault(mdl, 'DistEE', 'short', time=5, staged=True, gtype='bipartite')

rp.show_bipartite(resgraph, faultscen='DistEE short', time=5, showfaultlabels=False)
### 
rp.plot_mdlhist(mdlhist, 'StoreEE short', time=5) #, fxnflows=['StoreEE'])
### 
endresults, resgraph, mdlhist2=fp.run_one_fault(mdl, 'AffectDOF', 'RFshort', time=13, staged=True)
rp.show_graph(resgraph)
#fp.plotflowhist(flowhist2, 'RFshort', time=13)
#fp.plotghist(ghist2, 't=13 RFshort')
#
#xnom=flowhist2['nominal']['Env1']['x']
#ynom=flowhist2['nominal']['Env1']['y']
#znom=flowhist2['nominal']['Env1']['elev']
#
#x=flowhist2['faulty']['Env1']['x']
#y=flowhist2['faulty']['Env1']['y']
#z=flowhist2['faulty']['Env1']['elev']
#
#fig2 = plt.figure(2)
#ax2 = fig2.add_subplot(111, projection='3d')
#ax2.set_xlim3d(-100, 100)
#ax2.set_ylim3d(-100,100)
#ax2.set_zlim3d(0,100)
#ax2.plot(xnom,ynom,znom)
#ax2.plot(x,y,z)
#ax2.set_title('Fault response to RFpropbreak fault at t=13')
#ax2.legend(['Nominal Flightpath','Faulty Flighpath'], loc=4)
#
#plt.show()
#plt.close()

#resultstab=fp.runlist(mdl,staged=True)

#resultstab.write('tab4.ecsv', overwrite=True)


resultstab=fp.run_list(mdl, staged=True)

#t1=time.time()
#endclasses, mdlhists=fp.run_list(mdl, staged=True)
#simplefmea = rp.make_simplefmea(endclasses)
#t2=time.time()
#print(simplefmea)
#reshists, diffs, summaries = rp.compare_hists(mdlhists, returndiff=False)

#t3=time.time()
#t_running = t2-t1
#t_processing =t3-t2
#fullfmea = rp.make_fullfmea(endclasses, summaries)
#heatmap = rp.make_avgdegtimeheatmap(reshists)

#rp.show_bipartite(mdl.bipartite, heatmap=heatmap, scale=2)

#heatmap2 = rp.make_expdegtimeheatmap(reshists, endclasses)
#rp.show_bipartite(mdl.bipartite, heatmap=heatmap2, scale=2)



#print(t_reused)
#print(t_copied)
# based on this test, it appears reusing the model is actually slightly slower
# than copying. Not sure why. However, it's probably the case that execution is
# probably the biggest bottleneck