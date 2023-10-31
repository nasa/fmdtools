# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:30:00 2023

@author: dhulse
"""
from rover_model import Rover, RoverParam
import fmdtools.sim.propagate as prop
from fmdtools.sim.approach import SampleApproach


mdl = Rover()
p = RoverParam()
endresults, mdlhist = prop.nominal(mdl)
phases, modephases = mdlhist.get_modephases()

mdl_id = Rover(p={"drive_modes": {"mode_args": "set"}})
app_id = SampleApproach(
    mdl_id,
    faults="drive",
    phases={"drive": phases["avionics"]["drive"]},
    defaultsamp={"samp": "evenspacing", "numpts": 3},
)
endclasses_id, mdlhists_id = prop.approach(mdl_id, app_id)  # pool=mp.Pool(4))

# behave_endclasses_nested, behave_mdlhists_nested = prop.nested_approach(mdl, behave_nomapp, pool=mp.Pool(5), faults='drive')


# endresults,  mdlhist = prop.one_fault(mdl, 'drive','hmode_34', time=1, staged=False)
# an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift']})

# an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift', 'transfer']})

mdl = Rover(p=p.copy_with_vals(drive_modes={"mode_args": "manual"}))
endresults, mdlhist = prop.one_fault(
    mdl, "drive", "elec_open", time=1, staged=False
)

mdl = Rover(p=p.copy_with_vals(drive_modes={"mode_args": 100}))
endresults, mdlhist = prop.one_fault(mdl, "drive", "hmode_34", time=1, staged=False)

x = [1.0, 0.0, 1.0]
mdl = Rover(
    p=p.copy_with_vals(
        drive_modes={
            "custom_fault": {"friction": x[0], "drift": x[1], "transfer": x[2]}
        }
    )
)

_, mdlhist = prop.nominal(mdl)

endresults, reshist = prop.one_fault(
    mdl, "drive", "custom_fault", time=15, staged=True
)

line_dist = endresults.endclass.line_dist
end_loc = (reshist.faulty.flows.ground.s.x[-1], reshist.faulty.flows.ground.s.y[-1])
