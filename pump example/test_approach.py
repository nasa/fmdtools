# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:17:59 2019

@author: Daniel Hulse
"""
import sys
sys.path.append('../')

import fmdkit.faultprop as fp
import fmdkit.resultproc as rp
from ex_pump import * #required to import entire module
import time

mdl = Pump()

app_full = Approach(mdl, 'fullint')
app_center = Approach(mdl, 'center')
app_maxlike = Approach(mdl, 'maxlike')
app_multipt = Approach(mdl, 'multi-pt')
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