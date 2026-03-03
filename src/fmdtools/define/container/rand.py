#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Rand` class and other methods defining random properties used in blocks.

Has public Classes and Functions:

- :class:`Rand`: Superclass for Block random properties.
- :func:`get_pfunc_for_dist`: Gets the corresponding probability mass/density
  for outcome x for probability distributions with name 'randname' in numpy.
- :func:`get_prob_for_rand`: Gets the corresponding probability mass/density for random
  sample x from 'randname' function in numpy.
- :func:`calc_prob_for_integers`: Calculate probability for random.integers.
- :func:`calc_prob_density_for_random`: Calc probability density for random.random.
- :func:`calc_prob_for_choice`: Calculate probability for random.choice.
- :func:`calc_prob_for_shuffle_permutation`: Calculate probability for random.shuffle
  and random.permutation.
- :func:`calc_prob_for_permuted`: Calculate probability for random.permuted.
- :func:`array

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

from fmdtools.define.container.base import BaseContainer
from fmdtools.define.container.state import State
from fmdtools.define.base import round_float, array_x, unpack_x, is_iter

from scipy import stats, special
from recordclass import astuple
import numpy as np
import math


class Rand(BaseContainer):
    """
    Class for defining and interacting with random states of the model.

    Attributes
    ----------
    rng : np.random.default_rng
        random number generator
    probs : list
        probability of the given states
    seed : int
        state for the random number generator
    run_stochastic : bool
        Whether the rand is to be updated/called
    track_pdf : bool
        Whether a pdf is to be tracked and returned from the Rand.

    Examples
    --------
    Rand is meant to be extended in model definition with random states, e.g.:

    >>> class RandState(State):
    ...     noise: np.float64=1.0
    >>> class ExampleRand(Rand):
    ...     s: RandState = RandState()

    Which enables the use of set_rand_state, update_stochastic_states, etc for updating
    these states with methods called from the rng when run_stochastic=True.

    >>> exr = ExampleRand(run_stochastic=True, track_pdf=True)
    >>> exr.set_rand_state('noise', 'normal', 1.0, 1.0)
    >>> exr.s
    RandState(noise=1.3047170797544314)
    >>> exr.probs
    [np.float64(0.3808442490605113)]

    Checking copy:

    >>> exr2 = exr.copy()
    >>> exr2.s
    RandState(noise=1.3047170797544314)
    >>> exr2.run_stochastic
    np.True_
    >>> exr2.rng.bit_generator.state['state']['state'] == exr.rng.bit_generator.state['state']['state']
    True

    More state setting:
    >>> exr.set_rand_state('noise', 'normal', 1.0, 1.0)
    >>> exr.probs
    [np.float64(0.3808442490605113), np.float64(0.23230084450139615)]
    >>> exr.return_probdens()
    np.float64(0.08847044068025682)
    >>> exr2.probs
    [np.float64(0.3808442490605113)]
    """

    rolename = "r"
    rng: np.random._generator.Generator = np.random.default_rng()
    probs: list = list()
    probdens: np.float64 = np.float64(1.0)
    seed: np.int64 = np.int64(42)
    run_stochastic: np.bool_ = np.bool_(False)
    track_pdf: np.bool_ = np.bool_(False)
    default_track = ('s', 'probdens')

    def __init__(self, *args, seed=42, s_kwargs={}, **kwargs):

        args = self.get_true_fields(*args,
                                    seed=seed,
                                    rng=np.random.default_rng(seed),
                                    **kwargs)
        super().__init__(*args)
        if 's' in self.__fields__:
            self.s = self.s.__class__()
            self.s.set_atts(**s_kwargs)
        if self.seed is None:
            raise Exception("Invalid seed: None")

    def create_repr(self, fields=["seed", "s"], **kwargs):
        """Limit default repr to relevant fields."""
        return super().create_repr(fields=fields, **kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Rand

    def get_rand_states(self, auto_update_only=False):
        """
        Get the randomly-assigned states associated with the Rand at self.s.

        Parameters
        ----------
        auto_update_only : bool, optional
            Whether to only get auto-updated states. The default is False.

        Returns
        -------
        rand_states : dict
            States in self.s

        Examples
        --------
        >>> ExampleRand().get_rand_states()
        {'noise': np.float64(1.0)}
        >>> ExampleRand().get_rand_states(auto_update_only=True)
        {}
        """
        rand_states = self.s.asdict()
        if auto_update_only:
            rand_states = {state: vals for state,
                           vals in rand_states.items()
                           if hasattr(self.s, state+"_update")}
        return rand_states

    def set_rand_state(self, statename, methodname, *args):
        """
        Update the given random state with a given method and arguments.

        (if in run_stochastic mode)

        Parameters
        ----------
        statename : str
            name of the random state defined
        methodname :
            str name of the numpy method to call in the rng
        *args : args
            arguments for the numpy method
        """
        if getattr(self, 'run_stochastic', True):
            gen_method = getattr(self.rng, methodname)
            newvalue = gen_method(*args)
            if isinstance(newvalue, np.ndarray) and type(self.s[statename]) not in [list, np.array]:
                raise Exception("Random method for " + statename + " in " +
                                str(self.__class__) + " returned array when it should" +
                                " be a float/int--check args")
                newvalue = newvalue[0]
            self.s.set_field(statename, newvalue)
            if self.track_pdf:
                value_pds = get_prob_for_rand(newvalue, methodname, *args)
                self.probs.append(value_pds)

    def return_mutables(self):
        """Get mutable rand states."""
        rs = tuple([*self.rng.bit_generator.state['state'].values()])
        if 's' in self.__fields__:
            return rs + astuple(self.s)
        else:
            return rs

    def return_probdens(self):
        """Return probability density/mass corresponding to random sim."""
        if self.probs:
            return as_prob(self.probs)
        else:
            return 1.0

    def update_stochastic_states(self):
        """Update the defined stochastic states defined to auto-update."""
        if hasattr(self, 's'):
            if self.track_pdf:
                self.probs.clear()
            for state in self.s.__fields__:
                if hasattr(self.s, state+"_update"):
                    self.set_rand_state(state, getattr(self.s, state+'_update')[0],
                                        *getattr(self.s, state+'_update')[1])

    def reset(self):
        """Reset Rand to the initial state."""
        self.probs.clear()
        if 's' in self.__fields__:
            self.s.reset()
        self.rng = np.random.default_rng(self.seed)

    def update_seed(self, seed):
        """Update the random seed to the given value."""
        self.seed = seed
        BitGen = type(self.rng.bit_generator)
        self.rng.bit_generator.state = BitGen(seed).state

    def set_rng(self, other_rng):
        """Set the state of the rng in the Rand to the same state as other_rng."""
        self.rng = create_matching_rng(other_rng, init_seed=self.seed)

    def set_field(self, fieldname, value, as_copy=True):
        """Extend BaseContainer.assign to accomodate the rng."""
        if fieldname == 'rng':
            value = create_matching_rng(value, init_seed=self.seed)
            super(BaseContainer, self).__setattr__(fieldname, value)
        else:
            BaseContainer.set_field(self, fieldname, value, as_copy=as_copy)

    def init_hist_att(self, hist, att, timerange, track, str_size='<U20'):
        """Add field 'att' to history. Accommodates track_pdf option."""
        if self.track_pdf and att == 'track_pdf':
            hist.init_att('probdens', self.return_probdens(),
                          timerange=timerange, track='all')
        else:
            BaseContainer.init_hist_att(self, hist, att, timerange, track, str_size)


def create_matching_rng(other_rng, init_seed=42):
    """Create an rng matching the state of other_rng."""
    rng = np.random.default_rng(init_seed)
    rng.bit_generator.__setstate__(other_rng.bit_generator.__getstate__())
    return rng


def calc_prob_for_integers(x, *args):
    """
    Get probability for np.default_rng.integers.

    Examples
    --------
    >>> calc_prob_for_integers([0], 2)
    np.float64(0.5)
    >>> calc_prob_for_integers([0, 1], 0, 2)
    np.float64(0.25)
    >>> calc_prob_for_integers([0, 1, 2], 0, 2)
    np.float64(0.0)
    """
    if len(args) == 1:
        xmin, xmax = 0, args[0]
    elif len(args) == 2:
        xmin, xmax = args
    else:
        raise Exception("Invalid args: "+str(args))
    if xmin <= np.min(x) and np.max(x) < xmax:
        return np.prod([1/(xmax-xmin) for x in x])
    else:
        return np.float64(0.0)


def calc_prob_density_for_random(x):
    """
    Get probability density for np.default_rng.random.

    Examples
    --------
    >>> calc_prob_density_for_random([0.5])
    np.float64(1.0)
    >>> calc_prob_density_for_random([0.5, 0.1, 0.9, 0.5])
    np.float64(0.25)
    >>> calc_prob_density_for_random([0.5, 0.1, 0.9, 0.5, 1.1])
    np.float64(0.0)
    """
    if 0 <= np.min(x) and np.max(x) < 1.0:
        return np.float64(1/len(x))
    else:
        return np.float64(0.0)


def calc_prob_for_choice(x, options=[], size=1, replace=True, p=None):
    """
    Get probability corresponding to a call to np.choice.

    Examples
    --------
    >>> calc_prob_for_choice([1], [1,2])
    np.float64(0.5)
    >>> calc_prob_for_choice([1,2], [1,2], replace=False)
    np.float64(0.5)
    >>> calc_prob_for_choice([1,2], [1,2,3], p=[0.1, 0.1, 0.9])
    np.float64(0.01)
    """
    if isinstance(options, int):
        options = [*np.arange(options)]

    if replace:
        if p is None:
            p = [1/len(options) for i in options]
        return round_float(np.prod([p[options.index(i)] for i in x]), res=1e-6)
    else:
        if p is not None:
            raise Exception("Cannot calculate probabilities with replacement.")
        for j in x:
            if j not in options:
                raise Exception(str(j)+" not in options: "+str(options))
        perms = special.perm(len(options), len(x))
        if perms == 0.0:
            raise Exception("Too many draws from sample with replacement")
        return round_float(1/perms, res=1e-6)


def calc_prob_for_shuffle_permutation(x, options, *args, check_valid=True):
    """
    Get probability corresponding to rng.shuffle and rng.permutation.

    Examples
    --------
    >>> calc_prob_for_shuffle_permutation([1,2], [1,2])
    np.float64(0.5)
    >>> calc_prob_for_shuffle_permutation([2,1,3], [1,2,3])
    np.float64(0.16666666666666666)
    """
    if is_iter(options):
        options = np.array(options)
    else:
        options = np.arange(options)
    if check_valid and not set(unpack_x(options)).issuperset(set(unpack_x(x))):
        return np.float64(0.0)
    return np.float64(1/math.factorial(options.size))


def calc_prob_for_permuted(x, axis=None):
    """
    Get probability corresponding to rng.permuted.

    Examples
    --------
    >>> calc_prob_for_permuted(np.array([[1,2], [3,4]]))
    np.float64(0.041666666666666664)
    >>> calc_prob_for_permuted(np.array([[1,2], [3,4], [5,6]]), 0)
    np.float64(0.16666666666666666)
    >>> calc_prob_for_permuted(np.array([[1,2], [3,4], [5,6]]), 1)
    np.float64(0.5)
    """
    if axis is not None:
        return calc_prob_for_shuffle_permutation(x, x.shape[axis], check_valid=False)
    else:
        return calc_prob_for_shuffle_permutation(x, x)


def as_prob(pd):
    """Return array output of probabilities as single joint probability."""
    if is_iter(pd):
        return np.prod(pd)
    else:
        return pd


def get_scipy_pdf(randname, *args, **kwargs):
    """Get callable for scipy pdf function with given name and arguments."""
    def scipy_pdf(*x):
        return as_prob(getattr(stats, randname).pdf(array_x(x), *args, **kwargs))
    return scipy_pdf


def get_scipy_pmf(randname, *args, **kwargs):
    """Get callable for scipy pmf function with given name and arguments."""
    def scipy_pmf(*x):
        return as_prob(getattr(stats, randname).pmf(array_x(x), *args, **kwargs))
    return scipy_pmf


def get_custom_pfunc(func_handle, *args, **kwargs):
    """Get callable for calc_func pdf/pmf function with provided arguments."""
    def custom_pfunc(*x):
        return as_prob(func_handle(array_x(x), *args, **kwargs))
    return custom_pfunc


def get_exp_ray_pdf(randname, *args):
    """
    Get callable for scipy exponential and rayleigh pdf with numpy.random arguments.

    Examples
    --------
    >>> get_exp_ray_pdf("rayleigh", 2)(2.0)
    np.float64(0.3032653298563167)
    >>> get_exp_ray_pdf("rayleigh", 2, 2)(2.0)
    np.float64(0.0)
    >>> get_exp_ray_pdf("exponential", 1)(0.0)
    np.float64(1.0)
    >>> get_exp_ray_pdf("exponential", 1, -1.0)(0.0)
    np.float64(0.36787944117144233)
    """
    if randname == 'exponential':
        randname = "expon"
    if len(args) == 0:
        return get_scipy_pdf(randname, *args)
    elif len(args) == 1:
        return get_scipy_pdf(randname, scale=args[0])
    elif len(args) == 2:
        return get_scipy_pdf(randname, loc=args[1], scale=args[0])
    else:
        raise Exception("Too many arguments for "+randname+" distribution")


def get_hypergeometric_pmf(*args):
    """
    Get callable for scipy hypergeomeric pmf with numpy.random arguments.

    Examples
    --------
    >>> get_hypergeometric_pmf(50, 450, 100)(10)
    np.float64(0.14736784420411747)
    """
    n_pop = args[0]+args[1]
    n_good = args[0]
    n_sample = args[2]
    return get_scipy_pmf("hypergeom", n_pop, n_good, n_sample)


def get_lognormal_pdf(*args):
    """
    Get callable for scipy lognormal pdf with numpy.random arguments.

    Examples
    --------
    >>> get_lognormal_pdf(0, .25)(1.0)
    np.float64(1.5957691216057308)
    """
    s = args[1]
    scale = np.exp(args[0])
    return get_scipy_pdf("lognorm", s, scale=scale)


def get_standard_t_pdf(*args):
    """
    Get callable for scipy multivariate_t dist for a numpy.random.standard_t call.

    Note: doesn't support len(x)>1.

    Examples
    --------
    >>> get_standard_t_pdf(1)([0.0])
    np.float64(0.31830988618379075)
    """
    return get_scipy_pdf("multivariate_t", df=args[0])


def get_triangular_pdf(*args):
    """
    Get callable for scipy.triang corresponding to a numpy.random.triangular call.

    Examples
    --------
    >>> get_triangular_pdf(0,1,2)(0.0)
    np.float64(0.0)
    >>> get_triangular_pdf(0,1,2)(1.0)
    np.float64(1.0)
    >>> get_triangular_pdf(0,1,2)(1.5)
    np.float64(0.5)
    >>> get_triangular_pdf(0,1,2)(0.5, 0.5)
    np.float64(0.25)
    """
    left, mode, right = args[:3]
    loc = left
    scale = right-loc
    c = (mode-loc)/scale
    return get_scipy_pdf("triang", c, loc, scale)


def get_vonmises_pdf(*args):
    """Get callable for scipy.vonmises corresponding to numpy.random arguments."""
    return get_scipy_pdf(args[1], args[0])


def get_pfunc_for_dist(randname, *args):
    """
    Get the probability mass/density function corresponding to a numpy random draw.

    Uses a call to scipy.stats when available (with the correct arguments), otherwise
    uses a custom function provided in this module.

    Parameters
    ----------
    randname : str
        Name of numpy.random distribution
    args : tuple
        Arguments sent to numpy.random distribution

    Returns
    -------
    pfunc : callable
        pdf/pmf for the draw.
    """
    same_funcs = ['beta', 'dirichlet', 'f', 'gamma', 'laplace',
                  'logistic', 'multivariate_normal', 'pareto', 'uniform', 'wald']
    same_funcs_pmf = ['multinomial', 'poisson', 'zipf']
    different_funcs_pmf = {'binomial': 'binom',
                           'geometric': 'geom',
                           'logseries': 'logser',
                           'multivariate_hypergeometric': 'multivariate_hypergeom',
                           'negative_binomial': 'nbinom'}

    different_funcs = {'chisquare': 'chi2',
                       'gumbel': 'gumbel_r',
                       'noncentral_chisquare': 'ncx2',
                       'noncentral_f': 'ncf',
                       'normal': 'norm',
                       'power': 'powerlaw',
                       'standard_cauchy': 'cauchy',
                       'standard_gamma': 'gamma',
                       'standard_normal': 'norm',
                       'weibull': 'weibull_min'}
    match randname:
        case str if randname in same_funcs:
            return get_scipy_pdf(randname, *args)
        case str if randname in same_funcs_pmf:
            return get_scipy_pmf(randname, *args)
        case str if randname in different_funcs:
            return get_scipy_pdf(different_funcs[randname], *args)
        case str if randname in different_funcs_pmf:
            return get_scipy_pmf(different_funcs_pmf[randname], *args)
        case str if randname in ['exponential', 'rayleigh']:
            return get_exp_ray_pdf(randname, *args)
        case 'hypergeometric':
            return get_hypergeometric_pmf(*args)
        case 'lognormal':
            return get_lognormal_pdf(*args)
        case 'standard_t':
            return get_standard_t_pdf(*args)
        case 'triangular':
            return get_triangular_pdf(*args)
        case 'vonmises':
            return get_vonmises_pdf(*args)
        case 'integers':
            return get_custom_pfunc(calc_prob_for_integers, *args)
        case 'random':
            return get_custom_pfunc(calc_prob_density_for_random)
        case 'bytes':
            raise Exception("Not able to calculate probability density for bytes")
        case 'choice':
            return get_custom_pfunc(calc_prob_for_choice, *args)
        case str if randname in ['shuffle', 'permutation']:
            return get_custom_pfunc(calc_prob_for_shuffle_permutation, *args)
        case 'permuted':
            return get_custom_pfunc(calc_prob_for_permuted, *args)
        case _:
            raise Exception("Invalid randname distribution: " + randname +
                            ". Ensure that it is a part of numpy.random/scipy.stats")


def get_prob_for_rand(x, randname, *args):
    """
    Get the probability density/mass for random sample x.

    Pulled from 'randname' function in numpy. Calls get_pfunc_for_dist when scipy has
    a corresponding distribution function, otherwise calls custom functions
    to calculate the probabilities/probability densities.

    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of numpy.random distribution
    *args : tuple
        Arguments sent to numpy.random distribution

    Returns
    -------
    prob: float of probability density or mass, depending on function

    Examples
    --------
    >>> get_prob_for_rand(0, "normal", 0, 1)
    np.float64(0.3989422804014327)
    >>> get_prob_for_rand([0,0], "normal", 0, 1)
    np.float64(0.15915494309189535)
    >>> get_prob_for_rand(2, "integers", 4)
    np.float64(0.25)
    """
    pfunc = get_pfunc_for_dist(randname, *args)
    return pfunc(x)


class RandState(State):
    """Example random state for testing and docs."""

    noise: np.float64 = np.float64(1.0)


class ExampleRand(Rand):
    """Example Rand for testing and docs."""

    s: RandState = RandState()


if __name__ == "__main__":
    exr = ExampleRand()
    import doctest
    doctest.testmod(verbose=True)
