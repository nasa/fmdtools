# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:26:37 2019

This script shows some approaches to creating the ideal quadrature based on the full silulation of sample points

@author: hulsed
"""
import sys
sys.path.append('../')

import numpy as np
import quadpy
import fmdtools.faultprop as fp
import fmdtools.resultdisp as rd
from ex_pump import * #required to import entire module
import time

#mdl = Pump(params={'repair', 'ee', 'water', 'delay'})
mdl = Pump(params={'cost':{'water'}, 'delay':10, 'units':'hrs'}) # should give identical utilities
mdl = Pump()

app_full_plin = SampleApproach(mdl, defaultsamp={'samp':'fullint'})

endclasses, mdlhists = fp.run_approach(mdl, app_full_plin)
fmea = rd.tabulate.phasefmea(endclasses, app_full_plin)
rd.plot.samplecost(app_full_plin, endclasses, ('ExportWater','block'), samptype='fullint')

app_full_plin.prune_scenarios(endclasses,samptype='piecewise')
endclasses_plin, mdlhists_plin = fp.run_approach(mdl, app_full_plin)
fmea_plin = rd.tabulate.phasefmea(endclasses_plin, app_full_plin)

rd.plot.samplecost(app_full_plin, endclasses_plin, ('ExportWater','block'), samptype='pruned piecewise-linear')

app_full_sing = SampleApproach(mdl, defaultsamp={'samp':'fullint'})
app_full_sing.prune_scenarios(endclasses,samptype='bestpt')
endclasses_sing, mdlhists_sing = fp.run_approach(mdl, app_full_sing)
fmea_sing = rd.tabulate.phasefmea(endclasses_sing, app_full_sing)

rd.plot.samplecost(app_full_sing, endclasses_sing, ('ExportWater','block'))
