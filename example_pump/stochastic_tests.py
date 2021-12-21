
import unittest


from pump_stochastic import Pump
from fmdtools.faultsim import propagate

class StochasticVerification(unittest.TestCase):
    #def setUp(self):
    def test_run_safety(self):
        
        mdl = Pump(modelparams = {'phases':{'start':[0,5], 'on':[5, 50], 'end':[50,55]}, 'times':[0,20, 55], 'tstep':1,'seed':10})
        seed0 = mdl.seed
        endresults_1, resgraph_1, mdlhist_1=propagate.nominal(mdl, run_stochastic=True)
        seed1 = mdl.seed
        endresults_2, resgraph_2, mdlhist_2=propagate.nominal(mdl, run_stochastic=True)
        seed2 = mdl.seed
        self.assertEqual(seed0, seed1,seed2)
        self.assertTrue(all(mdlhist_1['functions']['MoveWater']['eff']==mdlhist_2['functions']['MoveWater']['eff']))
    def test_run_safety_noseed(self):
        
        mdl = Pump(modelparams = {'phases':{'start':[0,5], 'on':[5, 50], 'end':[50,55]}, 'times':[0,20, 55], 'tstep':1,'seed':None})
        seed0 = mdl.seed
        endresults_1, resgraph_1, mdlhist_1=propagate.nominal(mdl, run_stochastic=True)
        seed1 = mdl.seed
        endresults_2, resgraph_2, mdlhist_2=propagate.nominal(mdl, run_stochastic=True)
        seed2 = mdl.seed
        self.assertEqual(seed0, seed1,seed2)
        self.assertTrue(all(mdlhist_1['functions']['MoveWater']['eff']==mdlhist_2['functions']['MoveWater']['eff']))
    def test_run_approach(self):
        a=1
    def test_run_faults(self):
        b=1
    def test_run_nested(self):
        c=1
    
    

if __name__ == '__main__':
    unittest.main()
