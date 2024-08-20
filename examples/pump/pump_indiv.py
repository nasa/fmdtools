#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Individual-function pump model.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from examples.pump.ex_pump import MoveWat

import fmdtools.sim.propagate as prop
from fmdtools.sim.sample import FaultSample, FaultDomain, ParameterSample


class MoveWatDynamic(MoveWat):

    __slots__ = ()
    default_sp = {'end_time': 50}

    def static_loading(self, time):
        """Simulate how the outside system interacts with the function."""
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


if __name__ == "__main__":
    a = MoveWatDynamic(track='all', s={'eff': 1.0})

    result, mdlhist = prop.nominal(a,
                                   disturbances={10: {"wat_in.s.level": 0.0}})
    # mdlhist.plot_line('flows.sig_in.s.power', 'flows.wat_out.s.flowrate',
    #                   'flows.wat_in.s.level', 'flows.wat_in.s.flowrate')

    fd = FaultDomain(a)
    fd.add_all()
    fs = FaultSample(fd)
    fs.add_fault_phases()

    results, mdlhists = prop.fault_sample(a, fs)

    result, mdlhist = prop.nominal(a, track='all')

    # mdlhist.plot_line('flows.sig_in.s.power', 'flows.wat_out.s.flowrate')

    result, mdlhist = prop.one_fault(a, "short", time=10, track='all')

    # mdlhist.plot_line('flows.sig_in.s.power', 'flows.wat_out.s.flowrate')

    results, mdlhists = prop.single_faults(a)

    ps = ParameterSample()
    ps.add_variable_replicates([], replicates=10)


    results, mdlhists = prop.parameter_sample(a, ps)

    fds = {"fd": (('all',), {})}
    fss = {"fs": (('fault_phases', ["fd"]), {})}
    results, mdlhists, apps = prop.nested_sample(a, ps,
                                                 faultdomains=fds, faultsamples=fss)

