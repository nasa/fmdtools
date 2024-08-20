# -*- coding: utf-8 -*-
"""
Functions/classes for optimizing the drone defined in drone_mdl_opt.py.

Used/tested in opt_drone_rural to demonstrate fmdtools.sim.search.

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

from examples.multirotor.drone_mdl_rural import Drone, DroneParam, ResPolicy
from examples.multirotor.drone_mdl_rural import DronePhysicalParameters
from examples.multirotor.drone_mdl_dynamic import DroneEnvironmentGridParam

from fmdtools.sim.sample import FaultDomain, FaultSample
from fmdtools.sim import propagate as prop
from fmdtools.analyze.phases import from_hist

import numpy as np

# environmental parameters/constants
target = [0, 150, 160, 160]
safe = [0, 50, 10, 10]
start = [0.0, 0.0, 10, 10]

# variable options/costs
bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
batcostdict = {'monolithic': 0, 'series-split': 300,
               'parallel-split': 300, 'split-both': 600}

linarchs = ['quad', 'hex', 'oct']
linecostdict = {'quad': 0, 'hex': 1000, 'oct': 2000}

respols = ['continue', 'to_home', 'to_nearest', 'emland']

# template model
def_mdl = Drone()

# fault modes to optimize over
fd = FaultDomain(def_mdl)
fd.add_singlecomp_modes("store_ee")
fs = FaultSample(fd)
fs.add_fault_phases("move")


def plan_flight(z):
    """Plan a flightplan that covers an area at the given height z."""
    sq = rect(target[0:2], target[2], target[3])
    landing = start

    flightplan = {0: landing, 1: landing[0:2]+[z]}

    width, height = z, z
    # x,y, z
    startpt = [sq[0][0]+width/2, sq[0][1]+height/2, z]
    endpt = [sq[1][0]-width/2, sq[1][1]+height/2, z]

    num_rows = int(np.ceil((sq[2][1]-sq[0][1])/width))

    leftpts = [[startpt[0], startpt[1] + r*width] for r in range(num_rows)]
    rightpts = [[endpt[0], endpt[1] + r*width] for r in range(num_rows)]
    leftpts.sort(reverse=True)
    rightpts.sort(reverse=True)

    vec1 = leftpts
    vec2 = rightpts
    vec = []
    n = 2
    newplan = {}
    while len(vec1+vec2+vec) > 0:
        newplan[n] = vec1.pop()+[z]
        n += 1
        if len(vec1) < len(vec2) or n == 0:
            vec = vec2
            vec2 = vec1
            vec1 = vec

    flightplan.update(newplan)
    flightplan.update(
        {max(flightplan)+1: flightplan[1], max(flightplan)+2: flightplan[0]})
    return (tuple(tuple(v) for v in flightplan.values()), )


def rect(center, xw, yw):
    """
    Create list of corner coordinates for a rect, given a center, xwidth, and ywidth.

    Parameters
    ----------
    center : list
        Center of rectangle [x, y].
    xw : float
        x-width
        DESCRIPTION.
    yw : float
        y-width

    Returns
    -------
    square : list
        Outer points of the square [ll, lr, ur, ul].
    """
    square = [[center[0]-xw/2, center[1]-yw/2],
              [center[0]+xw/2, center[1]-yw/2],
              [center[0]+xw/2, center[1]+yw/2],
              [center[0]-xw/2, center[1]+yw/2]]
    return square


"""Manually-constructed optimization functions:"""


def x_to_dcost(xdes):
    """Calculate overall design cost from cost dictionaries."""
    descost = batcostdict[bats[int(xdes[0])]] + linecostdict[linarchs[int(xdes[1])]]
    return descost


def cd(*xd):
    """Calculate design cost (where xd is a tuple)."""
    return x_to_dcost(xd)


def xd_paramfunc(*xdes):
    """Transform x_des into corresponding bats/linearchs."""
    return bats[int(xdes[0])], linarchs[int(xdes[1])]


def calc_oper(mdl):
    """Calculate operational objective, constraints, and phasedict for a given model."""
    endresults_nom, mdlhist = prop.nominal(mdl)
    opercost = endresults_nom.endclass['expected_cost']
    g_soc = 20 - mdlhist.fxns.store_ee.s.soc[-1]
    # g_faults = any(endresults_nom['faults'])
    g_max_height = sum([i for i in mdlhist.flows.dofs.s.z-122 if i > 0])

    phasemaps = from_hist(mdlhist)
    return opercost, g_soc, g_max_height, phasemaps


def x_to_ocost(xdes, xoper, loc='rural'):
    """Calculate operational cost (obj, const, phases) at x_des, x_oper, and loc."""
    fp = plan_flight(xoper[0])[0]
    phys_p = DronePhysicalParameters(bat=bats[xdes[0]], linearch=linarchs[xdes[1]])
    params = DroneParam(respolicy=ResPolicy(bat='continue', line='continue'),
                        flightplan=fp,
                        env_param=DroneEnvironmentGridParam(loc=loc),
                        phys_param=phys_p)
    mdl = Drone(p=params)
    return calc_oper(mdl)


def xr_paramfunc(*x_res):
    """Convert x_res into corresponding bat/line resilience policy."""
    return respols[int(x_res[0])], respols[int(x_res[1])]


def calc_res(mdl, fullcosts=False, faultmodes='all', include_nominal=True,
             pool=False, phases={}, staged=True):
    """Calculate resilience cost for a given model."""
    fs = FaultSample(fd, phasemap=phases['plan_path'])
    fs.add_fault_phases("move")

    result, mdlhists = prop.fault_sample(mdl, fs, staged=staged, pool=pool,
                                         showprogress=False)
    rescost = result.total('expected_cost')-(not include_nominal) * \
        result.nominal.endclass['expected_cost']
    return rescost


def x_to_rcost(xdes, xoper, xres, loc='rural', fullcosts=False, faultmodes='all',
               include_nominal=False, pool=False, phases={}, staged=True):
    """Calculate resilience cost at xdes, xoper, xres variables."""
    fp = plan_flight(xoper[0])[0]
    phys_p = DronePhysicalParameters(bat=bats[xdes[0]], linearch=linarchs[xdes[1]])
    params = DroneParam(phys_param=phys_p,
                        respolicy=ResPolicy(*xr_paramfunc(*xres)),
                        flightplan=fp,
                        env_param=DroneEnvironmentGridParam(loc=loc))
    mdl = Drone(p=params)
    if not phases:
        _, _, _, phases = calc_oper(mdl)
    return calc_res(mdl,
                    fullcosts=fullcosts,
                    faultmodes=faultmodes,
                    include_nominal=include_nominal,
                    pool=pool,
                    phases=phases,
                    staged=staged)


if __name__ == "__main__":
    a = 1
