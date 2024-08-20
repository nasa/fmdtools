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

if __name__ == "__main__":
    mdl = Rover()
    p = RoverParam()
    endresults, mdlhist = prop.nominal(mdl)
    pm = from_hist(mdlhist)

    # range modes
    mdl_range = Rover(p={"drive_modes": {"mode_args": 'range-manual-all'},
                     'ground':{'linetype': 'turn', 'buffer_on': 0.5, 'buffer_poor': 0.8, 'buffer_near': 1.0}},
                  sp={'end_condition': ''})
    fd_range = FaultDomain(mdl_range)
    fd_range.add_all_fxnclass_modes('Drive')
    # fd_range.add_fault('drive', 'hmode_5')
    # fd_range.add_fault('drive', 'hmode_995')
    # fd_range.add_fault('drive', 'elec_open')
    # faults = [('drive', 'hmode_'+str(i)) for i in range(800)]
    # fd_range.add_faults(*faults)
    fs_range = FaultSample(fd_range, pm['plan_path'])
    fs_range.add_fault_phases('drive')
    results_range, mdlhists_range = prop.fault_sample(mdl_range, fs_range, pool=mp.Pool(4),
                                                      staged=True)

    # set modes
    mdl_id = Rover(p={"drive_modes": {"mode_args": "set"}})
    pd_id = FaultDomain(mdl_id)
    pd_id.add_all_fxn_modes('drive')

    ps_id = FaultSample(pd_id, phasemap=pm['plan_path'])
    ps_id.add_fault_phases('drive', args=(3,))

    ec_id, hist_id = prop.fault_sample(mdl, ps_id, pool=mp.Pool(4), staged=True)
    #  manual modes
    mdl_id = Rover(p={"drive_modes": {"mode_args": "manual"}})
    endresults, mdlhist = prop.one_fault(mdl_id, "drive", "elec_open",
                                         time=1, staged=False)

    # 100 drive faults
    mdl_100 = Rover(p={"drive_modes": {"mode_args": 100}})
    endresults, mdlhist = prop.one_fault(mdl_100, "drive", "hmode_34",
                                         time=1, staged=False)

    # custom fault
    x = [1.0, 0.0, 1.0]
    mdl = Rover(p={"drive_modes":
                   {"custom_fault":
                    {"friction": x[0], "drift": x[1], "transfer": x[2]}}})

    _, mdlhist = prop.nominal(mdl)

    endresults, reshist = prop.one_fault(mdl, "drive", "custom_fault",
                                         time=15, staged=True)

    line_dist = endresults.endclass.line_dist
    end_loc = (reshist.faulty.flows.pos.s.x[-1], reshist.faulty.flows.pos.s.y[-1])
