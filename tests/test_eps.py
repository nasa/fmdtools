# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:32:00 2021

@author: dhulse
"""
import unittest
import sys, os
sys.path.insert(1, os.path.join('..'))
from example_eps.eps import EPS
from fmdtools.faultsim import propagate
import fmdtools.resultdisp as rd
from fmdtools.modeldef import SampleApproach
import numpy as np

class epsTests(unittest.TestCase):
    def setUp(self):
        self.mdl = EPS()
    def test_backward_fault_prop_1(self):
        """ Tests that defined fault cases that require reverse propagation propagatate
        backwards through the graph as expected - distributor short leads to empty battery
        """
        endresults, resgraph, mdlhist = propagate.one_fault(self.mdl, 'Distribute_EE', 'short')
        self.assertEqual(endresults['faults']['Store_EE'], ['no_storage'])
    def test_backward_fault_prop_2(self):
        """ Tests that defined fault cases that require reverse propagation propagatate
        backwards through the graph as expected - motor short leads to distributor short
        """
        endresults, resgraph, mdlhist = propagate.one_fault(self.mdl, 'EE_to_ME', 'short')
        self.assertEqual(endresults['faults']['Store_EE'], ['no_storage'])
        self.assertEqual(endresults['faults']['Distribute_EE'], ['short'])
    def test_all_faults(self):
        """ Some basic tests for propagating lists of faults in the model--
        that histories have length 1, endresults have >0 costs, and total costs are higher
        than repairs"""
        mdl=self.mdl
        endclasses, reshists = propagate.single_faults(mdl, showprogress=False)
        actual_num_faults = np.sum([len(f.faultmodes) for f in mdl.fxns.values()])
        self.assertEqual(len(endclasses), actual_num_faults)
        hist_len_is_1 = all([all([all([len(j)==1 for j in i.values()]) for i in f['flows'].values()]) for  f in reshists.values()])
        self.assertTrue(hist_len_is_1)                  # all histories have length 1
        all_have_costs = all([e['cost'] for e in endclasses.values()])
        self.assertTrue(all_have_costs)                 # all endresults have positive costs
        repcosts = np.sum([ np.sum([m['rcost'] for m in f.faultmodes.values()]) for f in mdl.fxns.values()])
        total_simcosts = np.sum([e['cost'] for e in endclasses.values()])
        self.assertGreater(total_simcosts, repcosts)    # fault costs higher than if it was just repairs
    def test_fault_app(self):
        """ Tests that the expected number of scenarios are generated for a given approach """
        actual_num_faults = int(np.sum([len(f.faultmodes) for f in self.mdl.fxns.values()]))
        for num_joint in [2,3,actual_num_faults]:
            approach = SampleApproach(self.mdl, jointfaults={'faults':num_joint, 'jointfuncs':True, 'pcond':1.0, 'inclusive':False})
            self.assertEqual(len(approach.scenlist), np.math.comb(actual_num_faults,num_joint)) # tests the length
            endclasses, reshists = propagate.approach(self.mdl, approach, showprogress=False)

if __name__ == '__main__':
    unittest.main()
    
    
    
    mdl = EPS()
    
    approach = SampleApproach(mdl)
    
    #endresults, resgraph, mdlhist = propagate.one_fault(mdl, 'Distribute_EE', 'short')
    
    #mdl = EPS()
    #endresults, resgraph, mdlhist = propagate.one_fault(mdl, 'EE_to_ME', 'short')
    
    #mdl_nom = EPS()
    #endresults_nom, resgraph_nom, mdlhist_nom = propagate.nominal(mdl_nom)
    
    #endclasses, reshists = propagate.single_faults(mdl)