# -*- coding: utf-8 -*-
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


if __name__ == '__main__':
    unittest.main()