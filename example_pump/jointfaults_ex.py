# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:45:41 2019

This code tests some approaches to sampling joing fault scenarios

@author: Daniel Hulse
"""

import sys, os
sys.path.insert(0, os.path.join('..'))

import fmdtools.faultsim.propagate as prop
import fmdtools.resultdisp as rd
from ex_pump import * #required to import entire module

mdl = Pump()

app_jf1 = SampleApproach(mdl, jointfaults={'faults':2, 'pcond':0.1})
# if a function can have multiple modes injected at the same time
app_jf2 = SampleApproach(mdl, jointfaults={'faults':3, 'jointfuncs':True, 'pcond':0.1})

app_jf3 = SampleApproach(mdl, jointfaults={'faults':3, 'jointfuncs':True})

#note that faults above level 4 aren't made here because the rate is so low that it rounds to zero
app_jf5 = SampleApproach(mdl, jointfaults={'faults':5})

app_list = SampleApproach(mdl, jointfaults={'faults':[(('ImportEE', 'inf_v'),('ImportWater', 'no_wat'))], 'pcond':0.1})

endclasses, mdlhists = prop.approach(mdl, app_jf5)
fmea = rd.tabulate.phasefmea(endclasses, app_jf5).sort_values('expected cost', ascending=False)

fmea_small = rd.tabulate.summfmea(endclasses, app_jf5).sort_values('expected cost', ascending=False)

endclasses, mdlhists = prop.approach(mdl, app_jf2)

#rp.plot_samplecosts(app_jf2, endclasses, joint=True)
reshists, diffs, summaries = rd.process.hists(mdlhists)

mdlhist = {'nominal': mdlhists['nominal'], 'faulty':mdlhists['ImportEE: no_v, ImportWater: no_wat, MoveWater: mech_break, t=27']}
rd.plot.mdlhist(mdlhist, fault='IE:no_v, IW: no_w, MW: m_b', time=27)
