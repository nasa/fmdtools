# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:05:01 2023

@author: dhulse
"""
from pymoo.optimize import minimize
from fmdtools.sim import search
from fmdtools.sim.sample import SampleApproach
from fmdtools.define.role.parameter import SimParam
from examples.rover.rover_model import Rover

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
import numpy as np
import multiprocess as mp

if __name__ == "__main__":
    mdl = Rover(sp=SimParam(times=(0, 100), phases=(("start", 0, 30), ("end", 31, 60))))
    track = {"functions": {"Environment": "in_bound"}, "flows": {"ground": "all"}}
    rover_prob = search.ProblemInterface(
        "rover_problem", mdl, pool=mp.Pool(5), staged=True, track=track
    )
    app_drive = SampleApproach(
        mdl,
        faults="drive",
        phases={"global": [0, 39]},
        defaultsamp={"samp": "evenspacing", "numpts": 3},
    )
    rover_prob.add_simulation("drive_faults", "multi", app_drive.scenlist)
    rover_prob.add_variables(
        "drive_faults",
        ("cor_f", (-10, 100)),
        ("cor_d", (-100, 100)),
        ("cor_t", (-10, 100)),
        vartype="param",
    )
    rover_prob.add_objectives(
        "drive_faults", end_dist="end_dist", tot_deviation="tot_deviation"
    )
    
    
    pymoo_prob = rover_prob.to_pymoo_problem(objectives="end_dist")
    algorithm = PatternSearch(x0=np.array([0.0, 0.0, 0.0]))
    # res = minimize(pymoo_prob, algorithm, verbose=True)
