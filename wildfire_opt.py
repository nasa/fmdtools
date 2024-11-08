# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:18:57 2024

@author: dhulse
"""

from wildfiresim import WildfireSim, WildFireSimParameter, def_p, ps
from fmdtools.sim.search import ParameterSimProblem
from fmdtools.sim.sample import ParameterDomain, ParameterSample
import numpy as np

light_mdl = WildfireSim(p=def_p) #, track=None)

pd = ParameterDomain(WildFireSimParameter.from_base_loc)
pd.add_variable('x', var_lim = (0, 45))
pd.add_variable('y', var_lim = (0, 45))
pd.add_constant('num_strikes', 4)

psp = ParameterSimProblem(light_mdl, pd, 'parameter_sample', ps)
psp.add_result_objective('perc_burned', 'fxns.firepropagation.s.perc_burned', metric=np.mean)

psp.perc_burned(10, 10)

