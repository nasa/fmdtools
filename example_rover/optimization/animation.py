# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:23:52 2022

@author: igirshfe
"""

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.animation import FuncAnimation, PillowWriter

FRIC_LB = 1
FRIC_UB = 20
DRIFT_LB = -0.5
DRIFT_UB = 0.5
TRANSFER_LB = 0
TRANSFER_UB = 1

SPECIES_SIZE = 6 #number of individuals in a species
NUM_SPECIES = 4 #number of species
NUM_HSTATES = 3 #number of health states

dfccea = pd.read_csv('ccea_species.csv')

print(dfccea.head())

gen = dfccea['']

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#ax = plt.axes(projection='3d')
#set axes ranges
ax.set_xlim(FRIC_LB,FRIC_UB)
ax.set_ylim(DRIFT_LB,DRIFT_UB)
ax.set_zlim(TRANSFER_LB,TRANSFER_UB)

#labels
ax.set_xlabel("Friction")
ax.set_ylabel("Drift")
ax.set_zlabel("Transfer")
ax.set_title("Generation {}".format(gen))

#define points

colors = iter(plt.cm.rainbow(np.linspace(0,1,NUM_SPECIES)))

for i,s in enumerate(species):
    c = next(colors)
    for j in range(SPECIES_SIZE):
        ax.scatter(s[j][0],s[j][1],s[j][2], c=c)
