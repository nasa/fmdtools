# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:59:41 2023

@author: dhulse
"""

from examples.pump.ex_pump import MoveWat
from fmdtools.sim.propagate import nominal, one_fault, approach, single_faults, nominal_approach, nested_approach
from fmdtools.sim.approach import SampleApproach, NominalApproach
from fmdtools.analyze import plot


class MoveWatDynamic(MoveWat):
    default_sp = {'times': (0, 50)}

    def static_loading(self, time):
        # Signal Inputs
        if time < 5:
            self.sig_in.s.power = 0.0
        elif time < 50:
            self.sig_in.s.power = 1.0
        else:
            self.sig_in.s.power = 0.0
        # EE Inputs
        self.ee_in.s.voltage = 500.0
        # Water Input
        self.wat_out.s.level = 1.0
        # Water Output
        self.wat_out.s.area = 1.0


a = MoveWatDynamic()

result, mdlhist = nominal(a, track='all', disturbances={10: {"wat_in.s.level": 0.0}})
plot.hist(mdlhist, 'flows.sig_in.s.power', 'flows.wat_out.s.flowrate',
          'flows.wat_in.s.level', 'flows.wat_in.s.flowrate')

app = SampleApproach(a)
results, mdlhists = approach(a, app)

result, mdlhist = nominal(a, track='all')

plot.hist(mdlhist, 'flows.sig_in.s.power', 'flows.wat_out.s.flowrate')

result, mdlhist = one_fault(a, "short", time=10, track='all')

plot.hist(mdlhist, 'flows.sig_in.s.power', 'flows.wat_out.s.flowrate')

results, mdlhists = single_faults(a)

nomapp = NominalApproach()
nomapp.add_seed_replicates("seed_range", 10)

results, mdlhists = nominal_approach(a, nomapp)

results, mdlhists, apps = nested_approach(a, nomapp)
