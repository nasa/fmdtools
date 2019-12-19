# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:45:41 2019

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


app_jf1 = SampleApproach(mdl, jointfaults={'faults':2})
app_jf2 = SampleApproach(mdl, jointfaults={'faults':3})
app_jf5 = SampleApproach(mdl, jointfaults={'faults':5})