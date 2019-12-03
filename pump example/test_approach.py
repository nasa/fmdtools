# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:17:59 2019

@author: Daniel Hulse
"""
import sys
sys.path.append('../')

import fmdtools.faultprop as fp
import fmdtools.resultproc as rp
from ex_pump import * #required to import entire module
import time

mdl = Pump()

app_full = Approach(mdl, 'fullint')
app_center = Approach(mdl, 'center')
app_maxlike = Approach(mdl, 'maxlike')
app_multipt = Approach(mdl, 'multi-pt', numpts=3)
app_rand = Approach(mdl, 'randtimes')
app_arand = Approach(mdl, 'arandtimes')

app_short = Approach(mdl, 'multi-pt', faults=[('ImportEE', 'inf_v')])


# adding joint faults could look something like this:
class CustApproach(Approach):
    def __init__(self):
        #define joint fault scenarios
        jointfaultscens = {'on':(('ImportEE', 'inf_v'), ('ExportWater', 'block'))}
        super().__init__(mdl, 'maxlike')
        #self.addjointfaults(jointfaultscens) ???

def resilquant(approach, mdl):
    endclasses, mdlhists = fp.run_approach(mdl, approach)
    reshists, diffs, summaries = rp.compare_hists(mdlhists)
    
    simplefmea = rp.make_simplefmea(endclasses)
    util=sum(simplefmea['expected cost'])
    expdegtimes = rp.make_expdegtimeheatmap(reshists, endclasses)
    return util, expdegtimes


util_full, expdegtimes_full= resilquant(app_full, mdl)

util_center, expdegtimes_center = resilquant(app_center, mdl)    

perc_error = {i:(expdegtimes_full[i] - expdegtimes_center[i])/expdegtimes_full[i] for i in expdegtimes_full}

rp.show_bipartite(mdl.bipartite, heatmap=perc_error)

#note the percent error with/without the delay!!