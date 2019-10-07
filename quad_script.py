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

import quad_mdl as mdl

graph=mdl.initialize()

#scenlist=fp.listinitfaults(graph, mdl.times)

endresults1, resgraph, flowhist3, ghist3=fp.runnominal(mdl, track={'DOFs','Dir1', 'Env1', 'Force_LG'})
fp.showgraph(resgraph)
fp.plotflowhist(flowhist3, 'N/A', time=0)

#x=flowhist3['nominal']['Env1']['x']
#y=flowhist3['nominal']['Env1']['y']
#z=flowhist3['nominal']['Env1']['elev']
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlim3d(-100, 100)
#ax.set_ylim3d(-100,100)
#ax.set_zlim3d(0,100)
#ax.plot(x,y,z)
#plt.close()

#Check various scenarios individually

endresults, resgraph, flowhist, ghist=fp.proponefault(mdl, 'DistEE', 'short', time=5, track={'EE_1', 'Env1'})
fp.showgraph(resgraph)

fp.plotflowhist(flowhist, 'StoreEE short', time=5)

endresults, resgraph, flowhist2, ghist2=fp.proponefault(mdl, 'AffectDOF', 'RFshort', time=13, track={'DOFs', 'Env1', 'Dir1', 'Force_Air'}, gtrack=[10,13,20,40])
fp.showgraph(resgraph)
fp.plotflowhist(flowhist2, 'RFshort', time=13)
fp.plotghist(ghist2, 't=13 RFshort')

xnom=flowhist2['nominal']['Env1']['x']
ynom=flowhist2['nominal']['Env1']['y']
znom=flowhist2['nominal']['Env1']['elev']

x=flowhist2['faulty']['Env1']['x']
y=flowhist2['faulty']['Env1']['y']
z=flowhist2['faulty']['Env1']['elev']

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlim3d(-100, 100)
ax2.set_ylim3d(-100,100)
ax2.set_zlim3d(0,100)
ax2.plot(xnom,ynom,znom)
ax2.plot(x,y,z)
ax2.set_title('Fault response to RFpropbreak fault at t=13')
ax2.legend(['Nominal Flightpath','Faulty Flighpath'], loc=4)

plt.show()
plt.close()

#fullresults, resultstab=fp.proplist(mdl)
#print(resultstab)
# resultstab.write('tab.ecsv', overwrite=True)
