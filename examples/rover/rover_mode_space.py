#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script exploring possible modes for the rover.

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

from examples.rover.rover_model import Rover, RoverParam

import fmdtools.sim.propagate as prop
from fmdtools.sim.sample import FaultDomain, FaultSample
from fmdtools.analyze.phases import from_hist

import multiprocessing as mp


set_ranges = {"s.friction": {1.5, 3.0, 10.0},
              "s.transfer": {0.5, 0.0},
              "s.drift": {-0.2, 0.2}}

dist_ranges = ranges = {"s.friction": (0.0, 20, 10),
                        "s.transfer": (1.0, 0.0, 10),
                        "s.drift": (-0.5, 0.5, 10)}

p_test = {'ground': {'linetype': 'turn',
                     'buffer_on': 0.5,
                     'buffer_poor': 0.8,
                     'buffer_near': 1.0}}

if __name__ == "__main__":
    mdl = Rover()
    p = RoverParam()
    endresults, mdlhist = prop.nominal(mdl)
    pm = from_hist(mdlhist)
    mdl_test = Rover(p=p_test, sp={'end_condition': ''})

    # range modes
    fd_range = FaultDomain(mdl_test)
    fd_range.add_fault_space('drive', 'custom', dist_ranges)

    fs_range = FaultSample(fd_range, pm['plan_path'])
    fs_range.add_fault_phases('drive')
    results_range, mdlhists_range = prop.fault_sample(mdl_test, fs_range, pool=mp.Pool(4),
                                                      staged=True)

    # set modes
    pd_id = FaultDomain(mdl_test)
    pd_id.add_fault_space('drive', 'custom', set_ranges)

    ps_id = FaultSample(pd_id, phasemap=pm['plan_path'])
    ps_id.add_fault_phases('drive', args=(3,))

    ec_id, hist_id = prop.fault_sample(mdl_test, ps_id, pool=mp.Pool(4), staged=True)

    #  manual modes
    endresults, mdlhist = prop.one_fault(mdl_test, "drive", "elec_open",
                                         time=1, staged=False)

    # custom fault
    f_kw = {'disturbances': {'s.friction': 1.0, 's.transfer': 0.0, 's.drift': 1.0}}

    endresults, reshist = prop.one_fault(mdl, "drive", "custom", time=15,
                                         staged=True, f_kw=f_kw)

    line_dist = endresults.classify.tend.line_dist
    end_loc = (reshist.faulty.flows.pos.s.x[-1], reshist.faulty.flows.pos.s.y[-1])
