# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:44:01 2022

@author: dhulse
"""
import unittest
from fmdtools.sim import propagate
from fmdtools.define.rand import get_pdf_for_rand
from fmdtools.define.model import Model

import numpy as np
from tests.common import CommonTests

from fmdtools.define.state import State
class LocationState(State):
    x: float=0.0
    y: float=0.0

from fmdtools.define.flow import CommsFlow, MultiFlow
class Communications(CommsFlow):
    _init_s = LocationState
class Location(MultiFlow):
    _init_s = LocationState

from fmdtools.define.parameter import Parameter
class MoveParam(Parameter):
    x_up: float=0.0
    y_up: float=0.0

from fmdtools.define.block import FxnBlock
class Mover(FxnBlock):
    _init_p = MoveParam
    _init_communications = Communications
    _init_location = Location 
    def __init__(self, name='mover', flows={}, **kwargs):
        super().__init__(name=name, flows=flows, **kwargs)
        self.internal_info = self.communications.create_comms(name)
        self.loc = self.location.create_local(name)
    def dynamic_behavior(self, time):
        #move
        self.loc.s.inc(x=self.p.x_up, y=self.p.y_up)
        # the inbox should be cleared each timestep to allow new messages
        self.internal_info.clear_inbox()
    def behavior(self, time):
        #recieve messages
        self.internal_info.receive()
        #communicate
        if self.p.x_up==0.0:  
            self.internal_info.s.y=self.loc.s.y
            self.internal_info.send("all", "local", "y")
        elif self.p.y_up==0.0:   
            self.internal_info.s.x=self.loc.s.x
            self.internal_info.send("all", "local", "x")
    def find_classification(self, scen, fxnhist):
        return {"last_x": self.loc.s.x, "min_x": fxnhist.faulty.location.get(self.name).x}
        
class Coordinator(FxnBlock):
    _init_communications = Communications
    def __init__(self, name='coordinator', flows={}, **kwargs):
        super().__init__(name=name, flows=flows, **kwargs)
        self.coord_view= self.communications.create_comms(name, ports=["mover_1", "mover_2"])
    def dynamic_behavior(self, time):
        self.coord_view.clear_inbox()
    def behavior(self, time):
        self.coord_view.receive()
        self.coord_view.update("local", "mover_1", "y")
        self.coord_view.update("local", "mover_2", "x")

class TestModel(Model):
    default_sp = dict(times=(0,10))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_flow("communications", Communications)
        self.add_flow("location",       Location)
        self.add_fxn("mover_1",     Mover, "communications", "location", p = {"x_up":1.0})
        self.add_fxn("mover_2",     Mover, "communications", "location", p = {"y_up":1.0})
        
        self.add_fxn("coordinator", Coordinator, "communications")
        
        self.build()

class define_Tests(unittest.TestCase, CommonTests):
    def test_pdf_translation_options(self):
        """
        Test for getting the probability of a pdf using get_pdf_for_rand. 
        Ensures that arguments are passed to scipy and a probability is returned
        
        Also spot-tests values for certain distributions (integers, random, normal, choice)
        """
        rands = {'integers':(4,), 'random':(None,), 'shuffle':([1,2],), 'permutation':([1,2],), 'permuted':([1,2],), 'choice':([1,2,3],)}
        same_funcs = {'beta':(1,2), 'dirichlet':([0.5],), 'f':(1,2), 'gamma':(1,), 'laplace':(0,1), 'logistic':(0,1),\
                      'multivariate_normal':(0,1), 'pareto':(1,), 'uniform':(0,1), 'wald':(0,1)}
        same_funcs_pmf = {'multinomial':(2, [0.2]), 'poisson':(1,), 'zipf':(2,)}
        different_funcs_pmf = {'binomial':(2,0.5), 'geometric':(0.5,), 'logseries':(0.5,),\
                               'multivariate_hypergeometric':(8,2), 'negative_binomial':(0.5,2)}
        different_funcs = {'chisquare':(3,), 'gumbel':(1,1), 'noncentral_chisquare':(3,0.1),\
                           'noncentral_f':(2,3,0.1), 'normal':(1,1), 'power':(1.0,), 'standard_cauchy':(),\
                           'standard_gamma':(0.5,), 'standard_normal':(), 'weibull':(0.5,)}
        randnames = {**rands, **same_funcs, **same_funcs_pmf, **different_funcs}
        
        x=1        
        expected_values = {'integers': 0.25, 'random':1, 'shuffle':0.5, 'permuted':0.5, 'choice':1/3,\
                           'normal': 0.398942, 'pareto': 1.0, 'poisson': 0.368, 'power': 1.0, 'standard_normal':0.241971}
        
        for randname in randnames:
            p_d = get_pdf_for_rand(x, randname, randnames[randname])
            self.assertLessEqual(p_d[0], 1.0)       #checks to see that probability is 0<x<1
            self.assertGreaterEqual(p_d[0], 0.0)    #note that some densities may be higher than this under some values, this is mainly a check 
            if randname in expected_values: # spot tests for common distributions
                self.assertAlmostEqual(p_d[0], expected_values[randname], 3)
            self.assertIsInstance(p_d, np.ndarray)
    def test_multiflows(self):
        mdl = TestModel()
        endresults, mdlhist = propagate.nominal(mdl)
        # check that location copied such that the global version aren't modified but the local ones are
        np.testing.assert_array_equal(mdlhist.flows.location.s.x, np.zeros(11))
        np.testing.assert_array_equal(mdlhist.flows.location.s.y, np.zeros(11))
        np.testing.assert_array_equal(mdlhist.flows.location.mover_1.s.x, [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist.flows.location.mover_1.s.y, np.zeros(11))
        np.testing.assert_array_equal(mdlhist.flows.location.mover_2.s.x, np.zeros(11))
        np.testing.assert_array_equal(mdlhist.flows.location.mover_2.s.y, [i for i in range(11)])
        # check that communications combined such that both Movers have iterating x-y values
        np.testing.assert_array_equal(mdlhist.flows.communications.mover_1.s.x, [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist.flows.communications.mover_1.s.y, [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist.flows.communications.mover_2.s.x, [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist.flows.communications.mover_2.s.y, [i for i in range(11)])
        # check that coordinator parses communiations from each Mover
        np.testing.assert_array_equal(mdlhist.flows.communications.coordinator.mover_1.s.x, [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist.flows.communications.coordinator.mover_2.s.y, [i for i in range(11)])
        
        #tests that copying works
        mdl.flows["communications"].mover_1.s.x=25
        mdl.flows["communications"].mover_1.send(["mover_2", "coordinator"])
        self.assertEqual(mdl.flows["communications"].fxns["coordinator"]["in"], {"mover_1":()})
        
        
        mdl.flows["communications"].coordinator.receive()
        self.assertEqual(mdl.flows["communications"].fxns["mover_1"]["out"].s.x, 25)
        self.assertEqual(mdl.flows["communications"].fxns["coordinator"]["internal"].mover_1.s.x, 25)
        self.assertEqual(mdl.flows["communications"].fxns["mover_2"]["in"], {"mover_1":()})
        
        # copies should keep in/out dicts in place
        mdl2 = mdl.copy()
        self.assertEqual(mdl2.flows["communications"].fxns["mover_1"]["out"].s.x, 25)
        self.assertEqual(mdl2.flows["communications"].fxns["mover_2"]["in"], {"mover_1":()})
        self.assertEqual(mdl.flows["communications"].fxns["coordinator"]["internal"].mover_1.s.x, 25)
        
        


if __name__ == '__main__':
    unittest.main()
    
    mdl = TestModel()
    mdl.flows["communications"].mover_1.s.x=25
    mdl.flows["communications"].mover_1.send("mover_2")
    
    endclass, mdlhist = propagate.nominal(mdl)