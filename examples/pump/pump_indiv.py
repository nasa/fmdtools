# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:59:41 2023

@author: dhulse
"""

from examples.pump.ex_pump import MoveWat
import fmdtools.sim.propagate as prop
from fmdtools.sim.sample import FaultSample, FaultDomain, ParameterSample


class MoveWatDynamic(MoveWat):

    __slots__ = ()
    default_sp = {'end_time': 50}

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


a = MoveWatDynamic(track='all')

result, mdlhist = prop.nominal(a, track='all',
                               disturbances={10: {"wat_in.s.level": 0.0}})
mdlhist.plot_line('flows.sig_in.s.power', 'flows.wat_out.s.flowrate',
                  'flows.wat_in.s.level', 'flows.wat_in.s.flowrate')

fd = FaultDomain(a)
fd.add_all()
fs = FaultSample(fd)
fs.add_fault_phases()

results, mdlhists = prop.fault_sample(a, fs)

result, mdlhist = prop.nominal(a, track='all')

mdlhist.plot_line('flows.sig_in.s.power', 'flows.wat_out.s.flowrate')

result, mdlhist = prop.one_fault(a, "short", time=10, track='all')

mdlhist.plot_line('flows.sig_in.s.power', 'flows.wat_out.s.flowrate')

results, mdlhists = prop.single_faults(a)


ps = ParameterSample()
ps.add_variable_replicates([], replicates=10)


results, mdlhists = prop.parameter_sample(a, ps)

fds = {"fd": (('all',), {})}
fss = {"fs": (('fault_phases', ["fd"]), {})}
results, mdlhists, apps = prop.nested_sample(a, ps, faultdomains=fds, faultsamples=fss)
