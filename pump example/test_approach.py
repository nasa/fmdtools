# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:17:59 2019

@author: Daniel Hulse
"""
import sys
sys.path.append('../')

import fmdtools.faultprop as fp
import fmdtools.resultproc as rp
from ex_pump import * #required to import entire module
import time

mdl = Pump()

app_full = Approach(mdl, 'fullint')
app_center = Approach(mdl, 'center')
app_maxlike = Approach(mdl, 'maxlike')
app_multipt = Approach(mdl, 'multi-pt', numpts=3)
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
        
endclasses_full, mdlhists_full = fp.run_approach(mdl, app_full)

simplefmea_full = rp.make_simplefmea(endclasses_full)
print(simplefmea_full)
util_full=sum(simplefmea_full['expected cost'])


endclasses_center, mdlhists_center = fp.run_approach(mdl, app_center)

simplefmea_center = rp.make_simplefmea(endclasses_center)
print(simplefmea_center)
util_center=sum(simplefmea_center['expected cost'])

endclasses_mp, mdlhists_mp = fp.run_approach(mdl, app_multipt)
simplefmea_mp = rp.make_simplefmea(endclasses_mp)
print(simplefmea_mp)
util_mp=sum(simplefmea_mp['expected cost'])

endclasses_arand, mdlhists_arand = fp.run_approach(mdl, app_arand)
simplefmea_arand = rp.make_simplefmea(endclasses_arand)
print(simplefmea_arand)
util_arand=sum(simplefmea_arand['expected cost'])

endclasses_ml, mdlhists_ml = fp.run_approach(mdl, app_maxlike)
simplefmea_ml = rp.make_simplefmea(endclasses_ml)
print(simplefmea_ml)
util_ml=sum(simplefmea_ml['expected cost'])

#check first phase - no error
simplefmea_full[0:25]['expected cost']
simplefmea_center[0:5]['expected cost']
#second phase - no error
simplefmea_full[25:340]['expected cost']
simplefmea_center[5:12]['expected cost']
#last phase
simplefmea_full[340:]['expected cost']
simplefmea_center[12:]['expected cost']
#importwater no_wat
#note: rates are fine