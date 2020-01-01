# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:17:59 2019

@author: Daniel Hulse
"""
import sys
sys.path.append('../')

import numpy as np
import quadpy
import fmdtools.faultprop as fp
import fmdtools.resultproc as rp
from ex_pump import * #required to import entire module
import time

#mdl = Pump(params={'repair', 'ee', 'water', 'delay'})
mdl = Pump(params={'water'}) # should give identical utilities
mdl = Pump()

app_quad = SampleApproach(mdl, defaultsamp={'samp':'quadrature', 'quad': quadpy.line_segment.gauss_patterson(1)})

app_full = SampleApproach(mdl, defaultsamp={'samp':'fullint'})
app_maxlike = SampleApproach(mdl, defaultsamp={'samp':'likeliest'})
app_multipt = SampleApproach(mdl, defaultsamp={'samp':'evenspacing', 'numpts':3})
app_center = SampleApproach(mdl, defaultsamp={'samp':'evenspacing', 'numpts':1})
app_rand = SampleApproach(mdl, defaultsamp={'samp':'randtimes', 'numpts':3})
app_symrand = SampleApproach(mdl, defaultsamp={'samp':'symrandtimes', 'numpts':3})

app_cust = SampleApproach(mdl, sampparams={(('ExportWater', 'block'), 'on'):{'samp':'fullint'}})

tab=rp.make_samptimetable(app_multipt.sampletimes)

app_short = SampleApproach(mdl, faults=[('ImportEE', 'inf_v')], defaultsamp={'samp':'evenspacing', 'numpts':3})


#newscenids = prune_app(app_full, mdl)

#endclasses, mdlhists = fp.run_approach(mdl, app_full)

#rp.plot_samplecosts(app_full, endclasses)


endclasses, mdlhists = fp.run_approach(mdl, app_quad)
rp.plot_samplecosts(app_quad, endclasses)

endclasses, mdlhists = fp.run_approach(mdl, app_full)
rp.plot_samplecosts(app_full, endclasses)

#endclasses, mdlhists = fp.run_approach(mdl, app_fullp)
#rp.plot_samplecosts(app_fullp, endclasses)


endclasses, mdlhists = fp.run_approach(mdl, app_maxlike)
rp.plot_samplecosts(app_maxlike, endclasses)




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
    
    fmea = rp.make_summfmea(endclasses, approach)
    fmea2 = rp.make_phasefmea(endclasses, approach)
    util=sum(fmea['expected cost'])
    expdegtimes = rp.make_expdegtimeheatmap(reshists, endclasses)
    return util, expdegtimes, fmea, fmea2

util_full, expdegtimes_full, fmea_full, f_f= resilquant(app_full, mdl)

#util_fullp, expdegtimes_fullp, fmea_fullp, f_fp= resilquant(app_fullp, mdl)

util_short, expdegtimes_short, fmea_short, _ = resilquant(app_short, mdl)

util_quad, expdegtimes_quad, fmea_quad, f_q= resilquant(app_quad, mdl)

util_cust, expdegtimes_cust, fmea_cust, f_c= resilquant(app_cust, mdl)

util_enter, expdegtimes_center, fmea_center, f_ce= resilquant(app_center, mdl)


center_error = {i:(expdegtimes_full[i] - expdegtimes_center[i])/expdegtimes_full[i] for i in expdegtimes_full}

rp.show_bipartite(mdl.bipartite, heatmap=center_error)

util_maxlike, expdegtimes_maxlike, fmea_maxlike, f_m = resilquant(app_maxlike, mdl)

maxlike_error = {i:(expdegtimes_full[i] - expdegtimes_maxlike[i])/expdegtimes_full[i] for i in expdegtimes_full}

rp.show_bipartite(mdl.bipartite, heatmap=maxlike_error)

util_multipt, expdegtimes_multipt, fmea_multipt, f_mi = resilquant(app_multipt, mdl)

def prune_app(app, mdl):
    endclasses, mdlhists = fp.run_approach(mdl, app)
    newscenids = dict.fromkeys(app.scenids.keys())
    
    for modeinphase in app.scenids:
        costs= np.array([endclasses[scen]['cost'] for scen in app.scenids[modeinphase]])
        fullint = np.mean(costs)
        errs = abs(fullint - costs)
        mins = np.where(errs == errs.min())[0]
        newscenids[modeinphase] =  [app.scenids[modeinphase][mins[int(len(mins)/2)]]]
    return newscenids


sampparams = {'scen':{'numpts', 'type'}}




#note the percent error with/without the delay!!

# sum(fmea_full[0:25]['expected cost']) - same as center
# sum(fmea_full[25:340]['expected cost']) - different from center! (because of blockage degradation) - should calc individual error if possible!
# sum(fmea_full[340:]['expected cost']) - same as center

# fmea_center[0:5]['expected cost']
# fmea_center[5:12]['expected cost']
# fmea_center[12:]['expected cost']