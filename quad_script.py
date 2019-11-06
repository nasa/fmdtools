# -*- coding: utf-8 -*-
"""
File name: quad_script.py
Author: Daniel Hulse
Created: June 2019
Description: I/O with the quadrotor model defined in quad_mdl.py
"""

import faultprop as fp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from quad_mdl import *
import time

#scenlist=fp.listinitfaults(graph, mdl.times)
mdl = quadrotor()


# =============================================================================
#endresults1, resgraph, flowhist3, ghist3=fp.runnominal(mdl, track={'DOFs','Dir1', 'Env1', 'Force_LG'})
#fp.showgraph(resgraph, showfaultlabels=False)
#fp.plotflowhist(flowhist3, 'N/A', time=0)
## 
## #Check various scenarios individually
## 
endresults, resgraph, mdlhist = fp.runonefault(mdl, 'DistEE', 'short', time=5, staged=True, gtype='bipartite')

fp.showbipartite(resgraph, faultscen='DistEE short', time=5, showfaultlabels=False)
### 
fp.plotmdlhist(mdlhist, 'StoreEE short', time=5) #, fxnflows=['StoreEE'])
### 
endresults, resgraph, mdlhist2=fp.runonefault(mdl, 'AffectDOF', 'RFshort', time=13, staged=True)
fp.showgraph(resgraph)
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


#Doing a time test
#t1=time.time()
#fullresults, resultstab=fp.proplist(mdl,reuse=True)
t2=time.time()

resultstab=fp.runlist(mdl, staged=True)
t3=time.time()
#
#t_reused=t2-t1
t_copied=t3-t2
#print(t_reused)
#print(t_copied)
# based on this test, it appears reusing the model is actually slightly slower
# than copying. Not sure why. However, it's probably the case that execution is
# probably the biggest bottleneck