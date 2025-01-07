#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Rand` class and other methods defining random properties used in blocks.

Has Classes and Functions:

- :class:`Rand`: Superclass for Block random properties.

- :func:`get_pdf_for_rand`: Gets the corresponding probability mass/density for random
  sample x from 'randname' function in numpy.

- :func:`get_scipy_pdf_helper`: Gets probability mass/density for the outcome x from the
  distribution "randname". Used as a helper function in determining stochastic model
  state probability

- :func:`get_pdf_for_dist`: Gets the corresponding probability mass/density (from scipy)
  for outcome x for probability distributions with name 'randname' in numpy.

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
from fmdtools.define.base import is_iter, round_float
from fmdtools.analyze.common import is_numeric

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

    Examples
    --------
    Rand is meant to be extended in model definition with random states, e.g.:

    >>> class RandState(State):
    ...     noise: float=1.0
    >>> class ExampleRand(Rand):
    ...     s: RandState = RandState()

    Which enables the use of set_rand_state, update_stochastic_states, etc for updating
    these states with methods called from the rng when run_stochastic=True.

    >>> exr = ExampleRand(run_stochastic=True)
    >>> exr.set_rand_state('noise', 'normal', 1.0, 1.0)
    >>> exr.s
    RandState(noise=1.3047170797544314)

    Checking copy:

    >>> exr2 = exr.copy()
    >>> exr2.s
    RandState(noise=1.3047170797544314)
    >>> exr2.run_stochastic
    True
    >>> exr2.rng.__getstate__()['state'] == exr.rng.__getstate__()['state']
    True
    """

    rolename = "r"
    rng: np.random._generator.Generator = np.random.default_rng()
    probs: list = list()
    probdens: float = 1.0
    seed: int = 42
    run_stochastic: bool = False
    default_track = ('s', 'probdens')

    def __init__(self, *args, seed=42, run_stochastic=False, probs=list(), s_kwargs={}):
        args = self.get_true_fields(*args,
                                    seed=seed,
                                    run_stochastic=run_stochastic,
                                    probs=probs,
                                    rng=np.random.default_rng(seed))
        super().__init__(*args)
        if 's' in self.__fields__:
            self.s = self.s.__class__()
            self.s.set_atts(**s_kwargs)
        if self.seed is None:
            raise Exception("Invalid seed: None")

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
        {'noise': 1.0}
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
            setattr(self.s, statename, newvalue)
            if self.run_stochastic == 'track_pdf':
                value_pds = get_pdf_for_rand(newvalue, methodname, *args)
                self.probs.append(value_pds)

    def return_mutables(self):
        if 's' in self.__fields__:
            return astuple(self.s)
        else:
            return ()

    def return_probdens(self):
        if self.probs:
            state_pd = np.prod(self.probs)
        else:
            state_pd = 1.0
        return state_pd

    def update_stochastic_states(self):
        """Update the defined stochastic states defined to auto-update."""
        if hasattr(self, 's'):
            if self.run_stochastic == 'track_pdf':
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
        self.rng = np.random.default_rng(self.seed)
        self.rng.__setstate__(other_rng.__getstate__())

    def set_field(self, fieldname, value, as_copy=True):
        """Extend BaseContainer.assign to accomodate the rng."""
        if fieldname == 'rng':
            self.set_rng(value)
        else:
            BaseContainer.set_field(self, fieldname, value, as_copy=as_copy)

    def init_hist_att(self, hist, att, timerange, track, str_size='<U20'):
        """Add field 'att' to history. Accommodates track_pdf option."""
        if self.run_stochastic == 'track_pdf' and att == 'track_pdf':
            hist.init_att('probdens', self.return_probdens(),
                          timerange=timerange, track='all')
        else:
            BaseContainer.init_hist_att(self, hist, att, timerange, track, str_size)


def get_prob_for_integers(x, *args):
    """
    Get probability for np.default_rng.integers.

    Examples
    --------
    >>> get_prob_for_integers([0], 2)
    0.5
    >>> get_prob_for_integers([0, 1], 0, 2)
    0.25
    >>> get_prob_for_integers([0, 1, 2], 0, 2)
    0.0
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
        return 0.0


def get_prob_density_for_random(x):
    """
    Get probability density for np.default_rng.random.

    Examples
    --------
    >>> get_prob_density_for_random([0.5])
    1.0
    >>> get_prob_density_for_random([0.5, 0.1, 0.9, 0.5])
    0.25
    >>> get_prob_density_for_random([0.5, 0.1, 0.9, 0.5, 1.1])
    0.0
    """
    if 0 <= np.min(x) and np.max(x) < 1.0:
        return 1/len(x)
    else:
        return 0.0


def get_prob_for_choice(x, options=[], size=1, replace=True, p=None):
    """
    Get probability corresponding to a call to np.choice.

    Examples
    --------
    >>> get_prob_for_choice([1], [1,2])
    0.5
    >>> get_prob_for_choice([1,2], [1,2], replace=False)
    0.5
    >>> get_prob_for_choice([1,2], [1,2,3], p=[0.1, 0.1, 0.9])
    0.01
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


def get_prob_for_shuffle_permutation(options):
    """
    Get probability corresponding to rng.shuffle and rng.permutation.

    Examples
    --------
    >>> get_prob_for_shuffle_permutation([1,2])
    0.5
    >>> get_prob_for_shuffle_permutation([1,2,3])
    0.16666666666666666
    """
    if not is_iter(options):
        options = np.arange(options)
    else:
        options = np.array(options)
    return 1/math.factorial(options.size)


def get_prob_for_permuted(x, axis=None):
    """
    Get probability corresponding to rng.permuted.

    Examples
    --------
    >>> get_prob_for_permuted(np.array([[1,2], [3,4]]))
    0.041666666666666664
    >>> get_prob_for_permuted(np.array([[1,2], [3,4], [5,6]]), 0)
    0.16666666666666666
    >>> get_prob_for_permuted(np.array([[1,2], [3,4], [5,6]]), 1)
    0.5
    """
    if axis is not None:
        return get_prob_for_shuffle_permutation(x.shape[axis])
    else:
        return get_prob_for_shuffle_permutation(x)


def get_pdf_for_rand(x, randname, *args):
    """
    Get the probability density/mass function for random sample x.

    Pulled from 'randname' function in numpy. Calls get_pdf_for_dist when scipy has
    a corresponding distribution function, otherwise calls custom functions
    to calculate the probabilities/probability densities.

    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of numpy.random distribution
    args : tuple
        Arguments sent to numpy.random distribution

    Returns
    -------
    prob: float of probability density or mass, depending on function

    Examples
    --------
    >>> get_pdf_for_rand(0, "normal", 0,1)
    0.3989422804014327
    >>> get_pdf_for_rand(2, "integers", 4)
    0.25
    """
    if is_iter(x):
        x = np.array(x)
    elif is_numeric(x):
        x = np.array([x])
    else:
        raise Exception("Invalid x '"+str(x)+"' of type "+str(type(x)))

    if randname == 'integers':
        pd = get_prob_for_integers(x, *args)
    elif randname == 'random':
        pd = get_prob_density_for_random(x)
    elif randname == 'bytes':
        raise Exception("Not able to calculate probability density for bytes")
    elif randname == 'choice':
        pd = get_prob_for_choice(x)
    elif randname in ['shuffle', 'permutation']:
        pd = get_prob_for_shuffle_permutation(*args)
    elif randname == 'permuted':
        pd = get_prob_for_permuted(*args)
    else:
        pd = get_pdf_for_dist(x, randname, args)

    if is_iter(pd):
        return np.prod(pd)
    else:
        return pd


def get_scipy_pdf_helper(x, randname, args, pmf=False):
    """
    Get probability mass/density for the outcome x.

    Pulled from the distribution "randname" in scipy with arguments "args".

    Used as a helper function in determining stochastic model state probability

    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of scipy.stats probability distribution
    args : tuple
        Arguments to send to scipy.stats.randname.pdf
    pmf : Bool, optional
        Whether the distribution uses a probability mass function instead of a pdf.
        The default is False.

    Returns
    -------
    prob: float/array of probability densities

    """
    if randname == 'dirichlet':
        a = 1
    if pmf:
        return getattr(stats, randname).pmf(x, *args)
    else:
        return getattr(stats, randname).pdf(x, *args)


# note: when python 3.10 releases, this should become match/case
def get_pdf_for_dist(x, randname, args):
    """
    Get the scipy probability mass/density for outcome x from numpy random draw.

    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of numpy.random distribution
    args : tuple
        Arguments sent to numpy.random distribution

    Returns
    -------
    prob: float/array of probability densities
    """
    if type(x) in [np.ndarray, list] and len(x) > 1 and len(args) > 0:
        args = args[:-1]

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
    if randname in same_funcs:
        return get_scipy_pdf_helper(x, randname, args)
    elif randname in same_funcs_pmf:
        return get_scipy_pdf_helper(x, randname, args, pmf=True)
    elif randname in different_funcs:
        return get_scipy_pdf_helper(x, different_funcs[randname], args)
    elif randname in different_funcs_pmf:
        return get_scipy_pdf_helper(x, different_funcs_pmf[randname], args, pmf=True)
    elif randname in ['exponential', 'rayleigh']:
        if len(args) == 0:
            return getattr(stats, randname).pdf(x)
        elif len(args) == 1:
            return getattr(stats, randname).pdf(x, scale=args[0])
        elif len(args) == 2:
            return getattr(stats, randname).pdf(x, loc=args[1], scale=args[0])
        else:
            raise Exception("Too many arguments for "+randname+" distribution")
    elif randname == 'hypergeometric':
        n_pop = args[0]+args[1]
        n_good = args[0]
        n_sample = args[2]
        return stats.hypergeom.pmf(x, n_pop, n_good, n_sample)
    elif randname == 'lognormal':
        s = args[1]
        scale = np.exp(args[0])
        return stats.lognormal.pdf(x, s, scale=scale)
    elif randname == 'standard_t':
        return stats.multivariate_t.pdf(x, df=args[0])
    elif randname == 'triangular':
        left, mode, right = args[:3]
        loc = left
        scale = right-loc
        c = (mode-loc)/scale
        return stats.triang.pdf(x, c, loc, scale)
    elif randname == 'vonmises':
        return stats.vonmises.pdf(x, args[1], args[0])
    else:
        raise Exception("Invalid randname distribution: " + randname +
                        ". Ensure that it is a part of numpy.random/scipy.stats")


class RandState(State):
    """Example random state for testing and docs."""

    noise: float = 1.0


class ExampleRand(Rand):
    """Example Rand for testing and docs."""

    s: RandState = RandState()


if __name__ == "__main__":
    exr = ExampleRand()
    import doctest
    doctest.testmod(verbose=True)
