# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:30:00 2023

@author: dhulse
"""
from examples.rover.rover_model import Rover, RoverParam
import fmdtools.sim.propagate as prop
from fmdtools.sim.sample import FaultDomain, FaultSample
from fmdtools.analyze.phases import from_hist

if __name__ == "__main__":
    mdl = Rover()
    p = RoverParam()
    endresults, mdlhist = prop.nominal(mdl)
    pm = from_hist(mdlhist)

    # set modes
    mdl_id = Rover(p={"drive_modes": {"mode_args": "set"}})
    pd_id = FaultDomain(mdl_id)
    pd_id.add_all_fxn_modes('drive')

    ps_id = FaultSample(pd_id, phasemap=pm['plan_path'])
    ps_id.add_fault_phases('drive', args=(3,))

    ec_id, hist_id = prop.fault_sample(mdl, ps_id)

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
