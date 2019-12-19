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

app_fullp = SampleApproach(mdl, defaultsamp={'samp':'fullint'})

endclasses, mdlhists = fp.run_approach(mdl, app_fullp)

app_fullp.prune_scenarios2(endclasses)