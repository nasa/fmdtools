# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:26:37 2019

@author: hulsed
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

app_full_plin = SampleApproach(mdl, defaultsamp={'samp':'fullint'})

endclasses, mdlhists = fp.run_approach(mdl, app_full_plin)
fmea = rp.make_phasefmea(endclasses, app_full_plin)
rp.plot_samplecost(app_full_plin, endclasses, ('ExportWater','block'), samptype='fullint')

app_full_plin.prune_scenarios(endclasses,samptype='piecewise')
endclasses_plin, mdlhists_plin = fp.run_approach(mdl, app_full_plin)
fmea_plin = rp.make_phasefmea(endclasses_plin, app_full_plin)

rp.plot_samplecost(app_full_plin, endclasses_plin, ('ExportWater','block'), samptype='pruned piecewise-linear')

app_full_sing = SampleApproach(mdl, defaultsamp={'samp':'fullint'})
app_full_sing.prune_scenarios(endclasses,samptype='bestpt')
endclasses_sing, mdlhists_sing = fp.run_approach(mdl, app_full_sing)
fmea_sing = rp.make_phasefmea(endclasses_sing, app_full_sing)

rp.plot_samplecost(app_full_sing, endclasses_sing, ('ExportWater','block'))