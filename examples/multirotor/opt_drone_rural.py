# -*- coding: utf-8 -*-
"""
Functions/classes for optimizing the drone defined in drone_mdl_opt.py.

TODO: Adapt to new sample/optimization methods.
"""

from fmdtools.sim.search import ProblemInterface
from drone_mdl_rural import Drone, DroneParam
from fmdtools.sim.sample import FaultSample
import numpy as np


respols = ['continue', 'to_home', 'to_nearest', 'emland']

target = [0, 150, 160, 160]
safe = [0, 50, 10, 10]
start = [0.0, 0.0, 10, 10]
def_mdl = Drone()

def plan_flight(z):
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
    return {'flightplan': tuple(tuple(v) for v in flightplan.values())}

# creates list of corner coordinates for a square, given a center, xwidth, and ywidth
def rect(center, xw, yw):
    """
    creates list of corner coordinates for a rect, given a center, xwidth, and ywidth

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

# Optimization Functions
bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
linarchs = ['quad', 'hex', 'oct']
batcostdict = {'monolithic': 0, 'series-split': 300,
               'parallel-split': 300, 'split-both': 600}
linecostdict = {'quad': 0, 'hex': 1000, 'oct': 2000}


def x_to_dcost(xdes):
    descost = batcostdict[bats[int(xdes[0])]] + linecostdict[linarchs[int(xdes[1])]]
    return descost


def xd_paramfunc(xdes):
    return {'bat': bats[int(xdes[0])], 'linearch': linarchs[int(xdes[1])]}


opt_prob = ProblemInterface("drone_problem", def_mdl)
opt_prob.add_simulation("dcost", "external", x_to_dcost)
opt_prob.add_objectives("dcost", cd="cd")
opt_prob.add_variables("dcost", ('batteryarch', (0, 3)), ('linearch', (0, 3)))

opt_prob.add_simulation("ocost", "single", {}, staged=False,
                        upstream_sims={"dcost": {'paramfunc': xd_paramfunc}})
opt_prob.add_objectives("ocost", co="expected cost")
opt_prob.add_constraints("ocost", g_soc=("store_ee.s.soc", "vars", "end",("greater", 20)),
                                  g_max_height=("dofs.s.z", "vars", "all", ("less", 122)),
                                  g_faults=("repcost", "endclass", "end", ("less", 0.1)))
opt_prob.add_variables("ocost", "height", vartype=plan_flight)
#opt_prob.cd([2,2])
#opt_prob.co([10])

respols = ['continue', 'to_home', 'to_nearest', 'emland']


def spec_respol(bat, line):
    return {'respolicy': ResPolicy(bat=respols[int(bat)], line=respols[int(line)])}


app = SampleApproach(def_mdl,  phases={'move'},
                     faults=('single-component', 'store_ee'))
opt_prob.add_simulation("rcost", "multi", app.scenlist, include_nominal=False,
                        upstream_sims={'ocost': {'phases': {
                            'plan_path': 'move'}, 'pass_mdl': []}},
                        app_args={'faults': ('single-component', 'store_ee')},
                        staged=True)
opt_prob.add_objectives("rcost", cr="expected cost")
opt_prob.add_variables("rcost", "bat", "line", vartype=spec_respol)

#opt_prob.cr([1,0])

#an.plot.mdlhists(opt_prob._sims['rcost']['mdlhists']['store_ee lowcharge, t=7.0'], fxnflowvals={'dofs'}, time_slice=6)
#an.plot.mdlhists(opt_prob._sims['rcost']['mdlhists']['store_ee lowcharge, t=7.0'], fxnflowvals={'store_ee'}, time_slice=6)
#(variablename, objtype (optional), t (optional))


def calc_oper(mdl):
    endresults_nom, mdlhist = propagate.nominal(mdl)
    opercost = endresults_nom.endclass['expected cost']
    g_soc = 20 - mdlhist.fxns.store_ee.s.soc[-1]
    #g_faults = any(endresults_nom['faults'])
    g_max_height = sum([i for i in mdlhist.flows.dofs.s.z-122 if i > 0])

    phases, modephases = mdlhist.get_modephases()
    return opercost, g_soc, g_max_height, phases


def x_to_ocost(xdes, xoper, loc='rural'):
    fp = plan_flight(xoper[0], def_mdl)
    params = DroneParam(bat=bats[xdes[0]], linearch=linarchs[xdes[1]], respolicy=ResPolicy(
        bat='continue', line='continue'))
    mdl = Drone(params=params)
    return calc_oper(mdl)


def calc_res(mdl, fullcosts=False, faultmodes='all', include_nominal=True,
             pool=False, phases={}, staged=True):
    #app = SampleApproach(mdl, faults=('single-component', faultmodes), phases={'forward'})
    app = SampleApproach(mdl, faults=('single-component', 'store_ee'),
                         phases={'move': phases['plan_path']['move']})
    result, mdlhists = propagate.fault_sample(
        mdl, app, staged=staged, pool=pool, showprogress=False)  # , staged=False)
    rescost = result.total('expected cost')-(not include_nominal) * \
        result.nominal.endclass['expected cost']
    #an.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'dofs'}, time_slice=6)
    #an.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=7.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'store_ee'}, time_slice=6)
    #an.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'plan_path'}, time_slice=6)
    #an.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'rsig_traj', 'hsig_bat','hsig_dofs'})
    #[ec['expected cost'] for ec in endclasses.values()]
    #[ec['endclass']['expected cost'] for ec in opt_prob._sims['rcost']['results'].values()]
    #plot_faulttraj({'nominal':mdlhists['nominal'], 'faulty':mdlhists['store_ee lowcharge, t=7.0']}, mdl.params, title='Fault response to store_ee lowcharge, t=6.0')
    #phases, modephases = an.process.modephases(mdlhists['nominal'])
    #an.plot.phases({p:ph for p,ph in phases.items() if p=='plan_path'}, modephases)
    return rescost


def x_to_rcost(xdes, xoper, xres, loc='rural', fullcosts=False, faultmodes='all',
               include_nominal=False, pool=False, phases={}, staged=True):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    # start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0, 0.0, 10, 10]

    fp = plan_flight(xoper[0])

    params = DroneParam(bat=bats[xdes[0]], linearch=linarchs[xdes[1]], respolicy=ResPolicy(
        bat=respols[xres[0]], line=respols[xres[1]]))
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
    import fmdtools.sim.propagate as prop
    import matplotlib.pyplot as plt
    from fmdtools.analyze import show


    #opt_prob.add_combined_objective("total_cost", 'cd', 'co', 'cr')
    #opt_prob.total_cost([1,1],[100],[1,1])
    #opt_prob.total_cost([1,1,100,1,1])
    #opt_prob.time_sims([1,1,100,1,1])

    opt_prob.cr([2, 2, 100, 0, 0])

    x_to_rcost([2, 2], [100], [0, 0], faultmodes='store_ee')
    x_to_rcost([0, 0], [100], [0, 0], faultmodes='store_ee')
    opt_prob.show_architecture()

    opt_prob.update_sim_options("ocost", track={"functions":{"plan_path":"all"}, "flows":{"dofs":"all"}})
    opt_prob.update_sim_options("rcost", log_iter_hist=True, pool=mp.Pool(4), track={"functions":{"store_ee":"faults"}, "flows":{"dofs":"all"}})

    opt_prob.total_cost([1,1,120,1,1])
    opt_prob.total_cost([1,1,60,1,1])

    #opt_prob.time_sims([1, 1, 100, 1, 1])

    #opt_prob.iter_hist

    #plt.show()
