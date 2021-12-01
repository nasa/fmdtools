# -*- coding: utf-8 -*-
"""
File name: pump_script.py
Author: Daniel Hulse
Created: October 2019
Description: A simple example of I/O using faultprop.py and the pump model in ex_pump.py
"""

#Using the model that was set up, we can now perform a few different operations

#First, import the fault propogation library as well as the model
#since the package is in a parallel location to examples
import sys, os
sys.path.insert(1,os.path.join(".."))

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
from ex_pump import * #required to import entire module
import time

mdl = Pump()

##NOMINAL PLOTS
#Before seeing how faults propogate, it's useful to see how the model performs
#in the nominal state to check to see that the model has been defined correctly.
# Some things worth checking:
#   -are all functions on the graph?
#   -are the functions connected with the correct flows?
#   -do any faults occur in the nominal state?
#   -do all the flow states proceed as desired over time?
endresults, resgraph, mdlhist=propagate.nominal(mdl, track='all')
#plot graph
rd.graph.show(resgraph)
#plot the flows over time
rd.plot.mdlhist(mdlhist, 'Nominal')
#we can also look at the value of states over time with a table
nominal_state_table = rd.tabulate.hist(mdlhist)
print(nominal_state_table)
#this table is a pandas dataframe we can export with:
# nominal_state_table.to_csv("filename.csv")

#plots can be made on the bipartite version of the graph
endresults_bip, resgraph_bip, mdlhist2=propagate.nominal(mdl, gtype='bipartite')
rd.graph.show(resgraph_bip,gtype='bipartite', scale=2)

##SINGLE-FAULT PLOTS
## SCENARIO 1
##We might further query the faults to see what happens to the various states
endresults, resgraph, mdlhist=propagate.one_fault(mdl, 'MoveWater', 'short', time=10, staged=True)
#Here we again make a plot of states--however, looking at this might not tell us what degraded/failed
short_state_table = rd.tabulate.hist(mdlhist)
print(short_state_table)
#short_state_table.to_csv("test.csv")

#We can further process this table to get tables that show what degraded over time
reshist,diff1, summary = rd.process.hist(mdlhist)
rd.graph.result_from(mdl, reshist, 50)
rd.graph.result_from(mdl, reshist, 50, gtype='normal')
# We can also look at heat maps of the effect of the flow over time
heatmaps = rd.process.heatmaps(reshist, diff1)
# this is the amount of time each are degraded
rd.graph.show(mdl.bipartite,gtype='bipartite', heatmap=heatmaps['degtime'], scale=2)
# or the number of faults in each function
rd.graph.show(mdl.bipartite,gtype='bipartite', heatmap=heatmaps['maxfaults'], scale=2)
# or the accumulated difference between the states of the nominal and this over time
# note that this only counts states, not faults
rd.graph.show(mdl.bipartite,gtype='bipartite', heatmap=heatmaps['intdiff'], scale=2)
# max faults is best displayed on the graph view
rd.graph.show(mdl.graph, heatmap=heatmaps['maxfaults'])

#summary gives a dict of which functions and flows degraded over time, while reshist
# gives a history of the processed results. We can view the summary in a table
summary_table = rd.tabulate.summary(reshist)
print(summary_table)

#If we want to know precisely what degraded, when:
short_state_table_processed = rd.tabulate.hist(reshist)
print(short_state_table_processed)
#We might also be interested in a simpler view with just the functions/flows that degraded
short_state_table_simple = rd.tabulate.deghist(reshist)
print(short_state_table_simple)
# As well as statistics
short_state_table_stats = rd.tabulate.stats(reshist)
print(short_state_table_stats)

#Here, we plot flows and functions of interest of the model
rd.graph.show(resgraph)
rd.plot.mdlhist(mdlhist, 'short', time=10, fxnflows=['Wat_1','Wat_2', 'EE_1', 'Sig_1'])

##in addition to these visualizations, we can also look at the final results 
##to see which specific faults were caused, as well as the flow states
print(endresults)
#
##we can also look at other faults
endresults, resgraph, mdlhist=propagate.one_fault(mdl, 'ExportWater', 'block', time=10, staged=True)
rd.graph.show(resgraph)
rd.plot.mdlhist(mdlhist, 'blockage', time=10, fxnflows=['Wat_1','Wat_2', 'EE_1'])
# we can also view the results as a bipartite graph
endresults, resgraph_bip2, mdlhist=propagate.one_fault(mdl, 'ExportWater', 'block', time=10, gtype='bipartite')
rd.graph.show(resgraph_bip2, faultscen='ExportWater block', time=10, scale=2)


#you can save to a csv this with:


#finally, to get the results of all of the scenarios, we can go through the list
#note that this will propogate faults based on the times vector put in the model,
# e.g. times=[0,3,15,55] will propogate the faults at the begining, end, and at
# t=15 and t=15
endclasses, mdlhists=propagate.single_faults(mdl, staged=True)
simplefmea = rd.tabulate.simplefmea(endclasses)
print(simplefmea)
reshists, diffs, summaries = rd.process.hists(mdlhists)
fullfmea = rd.tabulate.fullfmea(endclasses, summaries)
heatmap = rd.process.avgdegtimeheatmap(reshists)

rd.graph.show(mdl.bipartite,gtype='bipartite', heatmap=heatmap, scale=2)

heatmap2 = rd.process.expdegtimeheatmap(reshists, endclasses)
rd.graph.show(mdl.bipartite, gtype='bipartite', heatmap=heatmap2, scale=2)
#time test
a = time.time()
endresults, resgraph, mdlhist=propagate.one_fault(mdl, 'MoveWater', 'short', time=10, staged=False)
b = time.time()
print(b-a)

endresults, resgraph, mdlhist=propagate.one_fault(mdl, 'MoveWater', 'mech_break', time=0, staged=False)
rd.plot.mdlhist(mdlhist, 'mech_break', time=0, fxnflows=['Wat_1','Wat_2', 'EE_1', 'Sig_1'])

#resultstab.write('tab.ecsv', overwrite=True)
#rd.graph.result_from(mdl, reshist, 10, gtype='normal')