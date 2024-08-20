#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rover optimization script.

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

from fmdtools.define.block.base import SimParam
from fmdtools.sim import search
from fmdtools.sim.sample import FaultDomain, FaultSample, ParameterDomain
import fmdtools.sim.propagate as prop
import fmdtools.analyze as an

from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
import numpy as np
import multiprocess as mp

if __name__ == "__main__":
    mdl = Rover(sp=SimParam(end_time=100, phases=(("start", 0, 30), ("end", 31, 60))))
    track = {"functions": {"Environment": "in_bound"}, "flows": {"ground": "all"}}
    # rover_prob = search.ProblemInterface(
    #    "rover_problem", mdl, pool=mp.Pool(5), staged=True, track=track
    # )
    endresults, mdlhist = prop.nominal(mdl)
    phasemap = an.phases.from_hist(mdlhist)
    fault_domain = FaultDomain(mdl)
    fault_domain.add_all_fxnclass_modes("Drive")
    fault_sample = FaultSample(fault_domain, phasemap["plan_path"])
    fault_sample.add_fault_phases("drive", args=(3,))
    pd = ParameterDomain(RoverParam)
    pd.add_variables(
        "correction.cor_f",
        "correction.cor_d",
        "correction.cor_t",
        lims={
            "correction.cor_f": (-10, 100),
            "correction.cor_d": (-100, 100),
            "correction.cor_t": (-10, 100),
        },
    )
    rover_prob = search.ParameterSimProblem(mdl, pd, "fault_sample", fault_sample)

    rover_prob.add_result_objective("f1", "endclass.end_dist")
    rover_prob.add_result_objective("f2", "endclass.tot_deviation")
    
    #define pmoo problem
    
    n_var = len(rover_prob.variables)
    
    x_low = np.array([])
    x_up = np.array([])
    for bound in rover_prob.parameterdomain.variables.values():
        x_low = np.append(x_low, bound[0])
        x_up = np.append(x_up, bound[1])
    
    obj = [lambda x: rover_prob.f1(*x)]
    pymoo_prob = FunctionalProblem(n_var, obj, xl = x_low, xu= x_up)
    algorithm = PatternSearch(x0=np.array([0.0, 0.0, 0.0]))
    # res = minimize(pymoo_prob, algorithm, verbose=True)
