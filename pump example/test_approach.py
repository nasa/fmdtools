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
app_arand = Approach(mdl, 'arand')