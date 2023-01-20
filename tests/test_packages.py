# Communications-*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:44:01 2022

@author: dhulse
"""
import unittest
import sys, os
sys.path.insert(1, os.path.join('..'))
from fmdtools.faultsim import propagate
import fmdtools.resultdisp as rd
from fmdtools.modeldef import *
import numpy as np
from CommonTests import CommonTests


class Mover(FxnBlock):
    def __init__(self, name, flows, params):
        self.set_atts(**params)
        FxnBlock.__init__(self, name,flows, comms={"Communications":"internal_info"},local={"Location":"loc"})
        self.internal_info.x = 20.0
        self.internal_info.y = 20.0

    def dynamic_behavior(self, time):
        #move
        self.loc.inc(x=self.x_up, y=self.y_up)
        # the inbox should be cleared each timestep to allow new messages
        
        #recieve messages
        self.internal_info.receive()
        self.internal_info.clear_inbox()
        #communicate
        if self.internal_info.y == 20 and self.internal_info.x == 20: self.internal_info.send('all', 'local', 'x','y')
        if self.x_up==0.0: 
            self.internal_info.y=self.loc.y
            self.internal_info.send("all", "local", 'y')
        elif self.y_up==0.0:   
            self.internal_info.x=self.loc.x
            self.internal_info.send("all", "local", "x")
    def find_classification(self, scen, fxnhist):
        return {"last_x": self.loc.x, "min_x": fxnhist["faulty"]["Location"][self.name]["x"]}
class Mover2(Mover, FxnBlock):
    def __init__(self, name, flows, params):
        """Same as mover 1, just with a non-standard initialiation for the local/comms flows"""
        self.set_atts(**params)
        super().__init__(name,flows, params)
        self.internal_info = self.Communications.create_comms(name)
        self.loc = self.Location.create_local(name)
class Coordinator(FxnBlock):
    def __init__(self, name, flows):
        FxnBlock.__init__(self, name,flows)
        self.coord_view= self.Communications.create_comms(name, ports=["Mover_1", "Mover_2"])
        # self.coord_view.x = 20.0
        # self.coord_view.y = 20.0
    def dynamic_behavior(self, time):
        self.coord_view.receive()
        self.coord_view.clear_inbox()
        self.coord_view.update("local", "Mover_1", "y")
        self.coord_view.update("local", "Mover_2", "x")
        
class TestModel(Model):
    def __init__(self, params={}, modelparams={'times':[0,10], 'tstep':1}, valparams={}):
        super().__init__(params=params, modelparams=modelparams, valparams=valparams)
        
        self.add_flow("Communications",{"x":0.0, "y":1.0}, fclass=CommsFlow)
        self.add_flow("Location", {"x":0.0, "y":0.0}, fclass=MultiFlow)

        self.add_fxn("Mover_1", ["Communications", "Location"], fclass=Mover, fparams= {"x_up":0.0, "y_up":1.0})
        self.add_fxn("Mover_2", ["Communications", "Location"], fclass=Mover2, fparams= {"x_up":1.0, "y_up":0.0})
        
        self.add_fxn("Coordinator", ["Communications"], fclass=Coordinator)
        
        self.build_model()

class modeldef_Tests(unittest.TestCase, CommonTests):
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
        np.testing.assert_array_equal(mdlhist["flows"]["Location"]["x"], np.zeros(11))
        np.testing.assert_array_equal(mdlhist["flows"]["Location"]["y"], np.zeros(11))
        np.testing.assert_array_equal(mdlhist["flows"]["Location"]["Mover_1"]["x"], np.zeros(11))
        np.testing.assert_array_equal(mdlhist["flows"]["Location"]["Mover_1"]["y"], [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist["flows"]["Location"]["Mover_2"]["y"], np.zeros(11))
        np.testing.assert_array_equal(mdlhist["flows"]["Location"]["Mover_2"]["x"], [i for i in range(11)])
        # check that communications combined such that both Movers have iterating x-y values
        np.testing.assert_array_equal(mdlhist["flows"]["Communications"]["Mover_1"]["x"], [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist["flows"]["Communications"]["Mover_1"]["y"], [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist["flows"]["Communications"]["Mover_2"]["x"], [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist["flows"]["Communications"]["Mover_2"]["y"], [i for i in range(11)])
        # check that coordinator parses communiations from each Mover
        np.testing.assert_array_equal(mdlhist["flows"]["Communications"]["Coordinator"]["x"], [i for i in range(11)])
        np.testing.assert_array_equal(mdlhist["flows"]["Communications"]["Coordinator"]["y"], [i for i in range(11)])
        
        #tests that copying works
        mdl.flows["Communications"].Mover_1.x=25
        mdl.flows["Communications"].Mover_1.send(["Mover_2", "Coordinator"])
        self.assertEqual(mdl.flows["Communications"].fxns["Coordinator"]["in"], {"Mover_1":()})
        
        
        mdl.flows["Communications"].Coordinator.receive()
        self.assertEqual(mdl.flows["Communications"].fxns["Mover_1"]["out"].x, 25)
        self.assertEqual(mdl.flows["Communications"].fxns["Coordinator"]["internal"].Mover_1.x, 25)
        self.assertEqual(mdl.flows["Communications"].fxns["Mover_2"]["in"], {"Mover_1":()})
        
        # copies should keep in/out dicts in place
        mdl2 = mdl.copy()
        self.assertEqual(mdl2.flows["Communications"].fxns["Mover_1"]["out"].x, 25)
        self.assertEqual(mdl2.flows["Communications"].fxns["Mover_2"]["in"], {"Mover_1":()})
        self.assertEqual(mdl.flows["Communications"].fxns["Coordinator"]["internal"].Mover_1.x, 25)
        
        


if __name__ == '__main__':
 #   unittest.main()
    
    mdl = TestModel()
    mdl.flows["Communications"].Mover_1.x=25
    mdl.flows["Communications"].Mover_1.send("Mover_2")
    
    endclass, mdlhist = propagate.nominal(mdl)