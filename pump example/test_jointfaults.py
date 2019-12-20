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
app_jf5 = SampleApproach(mdl, jointfaults={'faults':5})

app_list = SampleApproach(mdl, jointfaults={'faults':[(('ImportEE', 'inf_v'),('ImportWater', 'no_wat'))], 'pcond':0.1})