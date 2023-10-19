# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:53:34 2023

@author: dhulse
"""

import unittest
from examples.tank.tank_optimization_model import Tank
from examples.tank.tank_opt import x_to_rcost_leg, x_to_totcost_leg, x_to_descost

# from examples.tank.tank_optimization_model import Tank as Tank2
from examples.tank.tank_optimization_model import make_tankparam
from tests.common import CommonTests
import multiprocessing as mp
from fmdtools.sim.search import ProblemInterface
from fmdtools.sim.sample import SampleApproach


class TankOptTests(unittest.TestCase, CommonTests):

    def setUp(self):
        self.mdl = Tank()

    def test_same_rcost(self):

        kwarg_options = [dict(staged=True),
                         dict(staged=False),
                         dict(staged=True, pool=mp.Pool(4)),
                         dict(staged=False, pool=mp.Pool(4))]

        res_vars_i = {(v[0], v[1], v[2], v[3]): 1 for v in self.mdl.p.faultpolicy}

        for kwarg in kwarg_options:
            prob = ProblemInterface("res_problem", self.mdl, **kwarg)

            prob.add_simulation("des_cost", "external", x_to_descost)
            prob.add_objectives("des_cost", cd="cd")
            prob.add_variables("des_cost", 'capacity', 'turnup')

            app = SampleApproach(self.mdl)
            prob.add_simulation("res_sim", "multi", app.scenlist, include_nominal=True,
                                upstream_sims={"des_cost": {'p':
                                                            {"capacity": "capacity",
                                                             "turnup": "turnup"}}})
            res_vars = [(var, None) for var in res_vars_i.keys()]
            prob.add_variables("res_sim", *res_vars, vartype=make_tankparam)
            prob.add_objectives("res_sim", cost="expected cost", objtype="endclass")
            prob.add_combined_objective('tot_cost', 'cd', 'cost')

            for des_var in [[15, 0.5], [22, 0.1], [18, 0]]:
                rvar = [*res_vars_i.values()][:27]
                lvar = [*res_vars_i.values()][27:]
                prob.clear()
                prob.update_sim_vars("res_sim", new_p={'capacity': des_var[0],
                                                       'turnup': des_var[1]})
                inter_cost = prob.cost([*res_vars_i.values()])
                func_cost = x_to_rcost_leg(rvar, lvar, des_var)
                self.assertAlmostEqual(inter_cost, func_cost)

                inter_totcost = prob.tot_cost(des_var, [*res_vars_i.values()])
                func_totcost = x_to_totcost_leg(des_var, rvar, lvar)
                self.assertAlmostEqual(inter_totcost, func_totcost)


if __name__ == '__main__':

    unittest.main()
