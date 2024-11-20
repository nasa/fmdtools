# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:18:57 2024

@author: dhulse
"""

from wildfiresim import WildfireSim, WildFireSimParameter, def_p, ps
from fireenvironment import show_properties
from fmdtools.sim.search import ParameterSimProblem
from fmdtools.sim.sample import ParameterDomain

import autograd.numpy as np
from autograd import grad, jacobian
from matplotlib import pyplot as plt

from scipy.optimize._linesearch import line_search_armijo
from scipy.optimize import minimize


# def plot_path(fun, path=[], scale=10):
#     S, T, U = prepare_grid(fun, scale)
#     fig = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(111, aspect="equal")
#     c = ax.contourf(
#         S, T, U, cmap=plt.cm.hot,
#         levels=np.linspace(np.min(U), np.max(U), 101))
#     plt.colorbar(c)
#     if path:
#         path = np.asarray(path)
#         ax.plot(path[:, 0], path[:, 1], "-o")
#         ax.set_xlabel("$x_1$")
#         ax.set_ylabel("$x_2$")
#     return fig, ax



# def prepare_grid(fun, scale):
#     s = t = np.linspace(-scale, scale, 101)
#     S, T = np.meshgrid(s, t)
#     U = np.empty_like(S)
#     for i in range(S.shape[0]):
#         for j in range(S.shape[1]):
#             U[i, j] = fun(np.array([S[i, j], T[i, j]]))
#     return S, T, U



# def bfgs(x0, fun, grad, grad_threshold=1e-30, max_iter=100, n_linesearch=10):
#     x = np.copy(x0)
#     path = [np.copy(x0)]

#     n_dims = x.shape[0]
#     B = np.eye(n_dims)

#     old_f = None

#     g = grad(x)
#     for _ in range(max_iter):
#         p = np.linalg.solve(B, -g)

#         alpha, n_feval, old_f = line_search_armijo(fun, x, p, g, old_f)
#         if alpha is None:  # found no solution that satisfies the conditions
#             alpha = 1.0

#         s = alpha * p
#         x += s

#         prev_g = g
#         g = grad(x)
#         if np.linalg.norm(g) <= grad_threshold:
#             break

#         if prev_g is not None:
#             y = g - prev_g
#             B += (np.outer(y, y) / np.dot(y, s) -
#                   B.dot(s[:, np.newaxis]).dot(s[np.newaxis, :]).dot(B) / s.dot(B).dot(s))

#         path.append(np.copy(x))

#     return x, path

# test of bfgs code
# approx_grad = grad(perc_burned)
# x_best, path = bfgs([10.0, 10.0], perc_burned, approx_grad)

def create_problem():
    light_mdl = WildfireSim(p=def_p, track=None)
    pd = ParameterDomain(WildFireSimParameter.from_base_loc)
    pd.add_variable('x', var_lim = (0, 45))
    pd.add_variable('y', var_lim = (0, 45))
    pd.add_constant('num_strikes', 4)

    psp = ParameterSimProblem(light_mdl, pd, 'parameter_sample', ps)
    psp.add_result_objective('perc_burned', 'fxns.firepropagation.s.perc_burned', metric=np.mean)
    return psp


def show_scen_strikes(ps, fig=None, ax=None):
    props = {'strike': {"color": "yellow", "alpha": 0.5}}
    for scen in ps.scenarios():
        mdl = WildfireSim(p={**def_p, 'num_strikes': 4}, r=scen.r, track=None)
        fig, ax = mdl.flows['fireenvironment'].c.show(properties=props, fig=fig, ax=ax)
    return fig, ax

# psp.perc_burned(10, 10)


# adaptation function for autograd/bfgs:
# def perc_burned(x):
#     if isinstance(x, np.ndarray) or isinstance(x, list):
#         return psp.perc_burned(*x)
#     else:
#         return psp.perc_burned(*[*x._value])


if __name__ == "__main__":


    om = OptimMap()
    fig, ax = om.show(properties={"fval": {}})
    
    
    initial_simplex = [[10.0, 10.0],[0.0, 40.0],[40.0, 40.0]]
    res = minimize(perc_burned, [10.0, 10.0], method="Nelder-Mead",
                   bounds=((-1.0, 47.), (-1.0, 47.)),
                   options=dict(initial_simplex=initial_simplex,
                                maxiter=10,
                                disp=True))
    
    fig, ax = psp.mdl.flows['fireenvironment'].c.show(properties=show_properties)
    fig, ax = psp.iter_hist.plot_trajectories("variables.x", "variables.y",
                                              fig=fig, ax=ax,
                                              time_ticks=0.1, time_groups=['nominal'])
    psp.iter_hist.plot_line("objectives.perc_burned")