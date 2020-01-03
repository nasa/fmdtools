# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:45:41 2019

@author: Daniel Hulse
"""

import sys
sys.path.append('../')

import fmdtools.faultprop as fp
import fmdtools.resultproc as rp
from ex_pump import * #required to import entire module

#mdl = Pump(params={'repair', 'ee', 'water', 'delay'})
mdl = Pump(params={'water'}) # should give identical utilities
mdl = Pump()

app_jf1 = SampleApproach(mdl, jointfaults={'faults':2, 'pcond':0.1})
# if a function can have multiple modes injected at the same time
app_jf2 = SampleApproach(mdl, jointfaults={'faults':3, 'jointfuncs':True, 'pcond':0.1})

app_jf3 = SampleApproach(mdl, jointfaults={'faults':3, 'jointfuncs':True})

#note that faults above level 4 aren't made here because the rate is so low that it rounds to zero
app_jf5 = SampleApproach(mdl, jointfaults={'faults':5})

app_list = SampleApproach(mdl, jointfaults={'faults':[(('ImportEE', 'inf_v'),('ImportWater', 'no_wat'))], 'pcond':0.1})

endclasses, mdlhists = fp.run_approach(mdl, app_jf5)
fmea = rp.make_phasefmea(endclasses, app_jf5).sort_values('expected cost', ascending=False)

fmea_small = rp.make_summfmea(endclasses, app_jf5).sort_values('expected cost', ascending=False)

endclasses, mdlhists = fp.run_approach(mdl, app_jf2)

#rp.plot_samplecosts(app_jf2, endclasses, joint=True)
reshists, diffs, summaries = rp.compare_hists(mdlhists)

mdlhist = {'nominal': mdlhists['nominal'], 'faulty':mdlhists['ImportEE: no_v, ImportWater: no_wat, MoveWater: mech_break, t=27']}
rp.plot_mdlhist(mdlhist, fault='IE:no_v, IW: no_w, MW: m_b', time=27)
