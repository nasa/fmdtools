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

#mdl = Pump(params={'repair', 'ee', 'water', 'delay'})
mdl = Pump(params={'water'}) # should give identical utilities

app_full = SampleApproach(mdl, 'fullint')
app_center = SampleApproach(mdl, 'center')
app_maxlike = SampleApproach(mdl, 'maxlike')
app_multipt = SampleApproach(mdl, 'multi-pt', numpts=3)
app_rand = SampleApproach(mdl, 'randtimes')
app_arand = SampleApproach(mdl, 'arandtimes')

app_short = SampleApproach(mdl, 'multi-pt', faults=[('ImportEE', 'inf_v')])


# adding joint faults could look something like this:
class CustApproach(SampleApproach):
    def __init__(self):
        #define joint fault scenarios
        jointfaultscens = {'on':(('ImportEE', 'inf_v'), ('ExportWater', 'block'))}
        super().__init__(mdl, 'maxlike')
        #self.addjointfaults(jointfaultscens) ???

def resilquant(approach, mdl):
    endclasses, mdlhists = fp.run_approach(mdl, approach)
    reshists, diffs, summaries = rp.compare_hists(mdlhists)
    
    fmea = rp.make_phasefmea(endclasses, approach)
    util=sum(fmea['expected cost'])
    expdegtimes = rp.make_expdegtimeheatmap(reshists, endclasses)
    return util, expdegtimes, fmea


util_short, expdegtimes_short, fmea_short = resilquant(app_short, mdl)

util_center, expdegtimes_center, fmea_center = resilquant(app_center, mdl)


util_full, expdegtimes_full, fmea_full= resilquant(app_full, mdl)

#center_error = {i:(expdegtimes_full[i] - expdegtimes_center[i])/expdegtimes_full[i] for i in expdegtimes_full}

#rp.show_bipartite(mdl.bipartite, heatmap=center_error)

util_maxlike, expdegtimes_maxlike, fmea_maxlike = resilquant(app_maxlike, mdl)

#maxlike_error = {i:(expdegtimes_full[i] - expdegtimes_maxlike[i])/expdegtimes_full[i] for i in expdegtimes_full}

#rp.show_bipartite(mdl.bipartite, heatmap=maxlike_error)

#note the percent error with/without the delay!!

# sum(fmea_full[0:25]['expected cost']) - same as center
# sum(fmea_full[25:340]['expected cost']) - different from center! (because of blockage degradation) - should calc individual error if possible!
# sum(fmea_full[340:]['expected cost']) - same as center

# fmea_center[0:5]['expected cost']
# fmea_center[5:12]['expected cost']
# fmea_center[12:]['expected cost']