#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some integration tests for fmdtools packages.

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

from fmdtools.define.container.rand import get_prob_for_rand

import unittest


class define_Tests(unittest.TestCase):
    def test_pdf_translation_options(self):
        """
        Test for getting the probability of a pdf using get_prob_for_rand.
        Ensures that arguments are passed to scipy and a probability is returned

        Also spot-tests values for certain distributions
        (integers, random, normal, choice)
        """
        rands = {'integers': (4,),
                 'random': (None,),
                 'shuffle': ([1, 2],),
                 'permutation': ([1, 2],),
                 'permuted': (),
                 'choice': ([1, 2, 3],)}
        same_funcs = {'beta': (1, 2),
                      'dirichlet': ([0.5],),
                      'f': (1, 2),
                      'gamma': (1,),
                      'laplace': (0, 1),
                      'logistic': (0, 1),
                      'multivariate_normal': (0, 1),
                      'pareto': (1,),
                      'uniform': (0, 1),
                      'wald': (0, 1)}
        same_funcs_pmf = {'multinomial': (2, [0.2]),
                          'poisson': (1,),
                          'zipf': (2,)}
        different_funcs_pmf = {'binomial': (2, 0.5),
                               'geometric': (0.5,),
                               'logseries': (0.5,),
                               'multivariate_hypergeometric': (8, 2),
                               'negative_binomial': (0.5, 2)}
        different_funcs = {'chisquare': (3,),
                           'gumbel': (1, 1),
                           'noncentral_chisquare': (3, 0.1),
                           'noncentral_f': (2, 3, 0.1),
                           'normal': (1, 1),
                           'power': (1.0,),
                           'standard_cauchy': (),
                           'standard_gamma': (0.5,),
                           'standard_normal': (),
                           'weibull': (0.5,)}
        randnames = {**rands, **same_funcs, **same_funcs_pmf, **different_funcs}

        x = 1
        expected_values = {'integers': 0.25,
                           'random': 0.0,
                           'shuffle': 0.5,
                           'permuted': 1.0,
                           'choice': 1/3,
                           'normal': 0.398942,
                           'pareto': 1.0,
                           'poisson': 0.368,
                           'power': 1.0,
                           'standard_normal': 0.241971}

        for randname in randnames:
            try:
                p_d = get_prob_for_rand(x, randname, *randnames[randname])
                self.assertLessEqual(p_d, 1.0)  # checks to see that probability is 0<x<1
                # note that some densities may be higher than this under some values
                # this is mainly a check
                self.assertGreaterEqual(p_d, 0.0)
                # spot tests for common distributions
                if randname in expected_values:
                    self.assertAlmostEqual(p_d, expected_values[randname], 3)
            except AssertionError as e:
                raise AssertionError("Incorrect values from: "+randname) from e
            except TypeError as e:
                raise TypeError("Incorrect calculation from: "+randname) from e
            except ValueError as e:
                raise ValueError("Incorrect calculation from: "+randname) from e


if __name__ == '__main__':
    unittest.main()
