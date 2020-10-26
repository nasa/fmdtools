# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:57:40 2020

@author: Daniel Hulse
"""

from drone_opt import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#results = brute_search()

#resultstab, opthist = pd.DataFrame(results)
#resultstab.columns = resultstab.columns.to_flat_index()
#resultstab.to_csv('fullspace.csv')

#resultstab = pd.read_csv('fullspace.csv')



# TWO 3-D PARETO FRONTS:
#   - Design, Operational, Resilience Cost 
#   - Repair, Landing, Flight Cost (Safety - Viewed Value)

resultstab = pd.read_csv('fullspace.csv')
resultstab = resultstab.drop(labels='Unnamed: 0', axis=1)

pareto3 = get_3dpareto(resultstab, 0,1,2)

x = [x for x,y,z in pareto3.values()]
y = [y for x,y,z in pareto3.values()]
z = np.array([z for x,y,z in pareto3.values()])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
#ax.plot_trisurf(x,y,z)
ax.set_xlabel('DesignCost')
ax.set_ylabel('Operational Cost')
ax.set_zlabel('Resilience Cost')

resilcosts = resultstab[3:5]
flightcosts = resultstab[5:7].sum()
resilcosts = resilcosts.append(flightcosts, ignore_index=True)

pareto3 = get_3dpareto(resilcosts, 0,1,2)

x = [x for x,y,z in pareto3.values()]
y = [y for x,y,z in pareto3.values()]
z = np.array([z for x,y,z in pareto3.values()])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
#ax.plot_trisurf(x,y,z)
ax.set_xlabel('Repair Cost')
ax.set_ylabel('Landing Cost')
ax.set_zlabel('Flight Cost')




plt.figure()
pareto = get_2dpareto(resultstab, 5, 6)

x = [x for x,y in pareto.values()]
y = [y for x,y in pareto.values()]

plt.plot(x,y)
plt.xlabel('Safety Cost')
plt.ylabel('Lost Value')






