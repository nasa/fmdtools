# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:57:40 2020

@author: Daniel Hulse
"""

from drone_opt import *
import pandas as pd
import matplotlib.pyplot as plt

#results = brute_search()

#resultstab = pd.DataFrame(results)

#resultstab.to_csv('fullspace.csv')
#resultstab.columns = resultstab.columns.to_flat_index()
#resultstab = pd.read_csv('fullspace.csv')

resultstab = pd.read_csv('fullspace.csv')
resultstab = resultstab.drop(labels='Unnamed: 0', axis=1)

pareto = dict()

for x in resultstab:
    if not any([(resultstab[x][0] >= resultstab[i][0] and resultstab[x][2] > resultstab[i][2]) or (resultstab[x][0] > resultstab[i][0] and resultstab[x][2] >= resultstab[i][2]) for i in resultstab]):
        pareto[x] = resultstab[x][1], resultstab[x][2]

pareto = dict(sorted(pareto.items(), key=lambda x: x[1]))

x = [x for x,y in pareto.values()]
y = [y for x,y in pareto.values()]

plt.plot(x,y)
plt.xlabel('Design Cost')
plt.ylabel('Resilience Cost')