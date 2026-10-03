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

from scipy import stats
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

    Checking json import/export:
    >>> exrj = exr.tojson()
    >>> exr3 = ExampleRand.fromjson(exrj)
    >>> exr3.rng.bit_generator.state['state']['state'] == exr.rng.bit_generator.state['state']['state']
    True
    """

    rolename = "r"
    rng: np.random._generator.Generator = np.random.default_rng(42)
    seed: int = 42
    state: int = 0
    inc: int = 0
    has_uint32: int = 0
    uinteger: int = 0
    probs: list = list()
    probdens: np.float64 = np.float64(1.0)
    run_stochastic: np.bool_ = np.bool_(False)
    track_pdf: np.bool_ = np.bool_(False)
    default_track = ('s', 'probdens')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = self.create_rng()

    def create_repr(self, fields=["seed", "s"], **kwargs):
        """Limit default repr to relevant fields."""
        return super().create_repr(fields=fields, **kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Rand

    def store_rng_state(self, rng):
        """Update state from the rng."""
        st = rng.bit_generator.__getstate__()
        self.assign(st[0]['state'])
        self.assign(st[0], 'has_uint32', 'uinteger')
        self.seed = st[1].entropy

    def is_set(self):
        return not (self.state == 0 and self.inc == 0)

    def create_rng(self):
        """Update the state of another rng."""
        rng = np.random.default_rng(self.seed)
        if self.is_set():
            rng.bit_generator.__setstate__(self.gen_state())
        self.store_rng_state(rng)
        return rng

    def gen_state(self):
        """Generate state for rng."""
        return {'bit_generator': 'PCG64',
                'state': {'state': self.state, 'inc': self.inc},
                'has_uint32': self.has_uint32,
                'uinteger': self.uinteger}

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

        Array draws are accepted for list- or ndarray-valued states. Scalar states
        still reject array results so an incorrect draw size is not hidden.

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
            if (isinstance(newvalue, np.ndarray)
                    and not isinstance(self.s[statename], (list, np.ndarray))):
                raise Exception("Random method for " + statename + " in " +
                                str(self.__class__) + " returned array when it should" +
                                " be a float/int--check args")
                newvalue = newvalue[0]
            self.s.set_field(statename, newvalue)
            self.store_rng_state(self.rng)
            if self.track_pdf:
                value_pds = get_prob_for_rand(newvalue, methodname, *args)
                self.probs.append(value_pds)

    def return_mutables(self):
        """Get mutable rand states."""
        self.store_rng_state(self.rng)
        rs = (self.seed, self.state, self.inc)
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

    def update_seed(self, seed, state=[]):
        """Update the random seed to the given value."""
        self.seed = seed
        BitGen = type(self.rng.bit_generator)
        st = BitGen(seed).state
        if state:
            st['state'] = state
        self.rng.bit_generator.__setstate__(st)
        self.store_rng_state(self.rng)

    def set_field(self, fieldname, value, as_copy=True):
        """Extend BaseContainer.assign to accomodate the rng."""
        if fieldname == 'rng':
            self.store_rng_state(value)
            value = self.create_rng()
        BaseContainer.set_field(self, fieldname, value, as_copy=as_copy)

    def init_hist_att(self, hist, att, timerange, track, str_size='<U20'):
        """Add field 'att' to history. Accommodates track_pdf option."""
        if self.track_pdf and att == 'track_pdf':
            hist.init_att('probdens', self.return_probdens(),
                          timerange=timerange, track='all')
        else:
            BaseContainer.init_hist_att(self, hist, att, timerange, track, str_size)

    def asdict(self, *fields, exclude=['rng'], **kwargs):
        """Represent as a dict (exclude rng for json)."""
        self.store_rng_state(self.rng)
        return super().asdict(*fields, exclude=exclude, **kwargs)



def calc_prob_for_integers(x, low, high=None, size=None, dtype=np.int64,
                           endpoint=False):
    """
    Get the joint mass of independent np.default_rng.integers draws.

    Bounds broadcast against the supplied values. Size and dtype describe
    generation, not the mass of those values. Empty draws have joint mass one.

    Examples
    --------
    >>> calc_prob_for_integers([0], 2)
    np.float64(0.5)
    >>> calc_prob_for_integers([0, 1], 0, 2)
    np.float64(0.25)
    >>> calc_prob_for_integers([0, 1, 2], 0, 2)
    np.float64(0.0)
    """
    del size, dtype
    if high is None:
        low, high = 0, low
    # Python integer arithmetic preserves uint64 endpoints and interval widths.
    low = np.asarray(low).astype(object)
    high = np.asarray(high).astype(object) + int(endpoint)
    if np.any(high <= low):
        raise ValueError("Integer bounds must define a nonempty interval.")
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        if not np.all(np.isfinite(x) & (x == np.floor(x))):
            return np.float64(0.0)
    x, low, high = np.broadcast_arrays(x.astype(object), low, high)
    if np.any((x < low) | (x >= high)):
        return np.float64(0.0)
    return np.prod(np.asarray(1.0 / (high - low), dtype=np.float64))


def calc_prob_density_for_random(x):
    """
    Get the joint density of independent unit-uniform draws from rng.random.

    Each draw has density one on [0, 1), so their joint density is one when
    every value is in that interval and zero otherwise. An empty draw has
    the empty-product density of one.

    Examples
    --------
    >>> calc_prob_density_for_random([0.5])
    np.float64(1.0)
    >>> calc_prob_density_for_random([0.5, 0.1, 0.9, 0.5])
    np.float64(1.0)
    >>> calc_prob_density_for_random([0.5, 0.1, 0.9, 0.5, 1.1])
    np.float64(0.0)
    """
    x = np.asarray(x)
    return np.float64(np.all((0.0 <= x) & (x < 1.0)))


def calc_prob_for_choice(x, options=[], size=1, replace=True, p=None):
    """
    Get the ordered joint mass of choices from one-dimensional options.

    Scalars and arrays are evaluated using all supplied values. Sampling size
    does not change their mass. Uniform sampling without replacement is
    supported; weighted sampling without replacement remains unsupported.
    The existing six-decimal probability rounding is retained.

    Examples
    --------
    >>> calc_prob_for_choice([1], [1,2])
    np.float64(0.5)
    >>> calc_prob_for_choice([1,2], [1,2], replace=False)
    np.float64(0.5)
    >>> calc_prob_for_choice([1,2], [1,2,3], p=[0.1, 0.1, 0.8])
    np.float64(0.01)
    """
    if isinstance(options, (int, np.integer)):
        options = np.arange(options)
    options = np.asarray(options)
    if options.ndim != 1:
        raise ValueError("Choice probability requires one-dimensional options.")
    draws = np.asarray(x).ravel()
    if not replace and p is not None:
        raise Exception("Cannot calculate weighted probabilities without replacement.")
    if not replace and draws.size > options.size:
        raise ValueError("Too many draws from sample without replacement.")
    if not draws.size:
        return np.float64(1.0)
    if not options.size:
        raise ValueError("Choice options must not be empty for a nonempty draw.")
    if p is None:
        probabilities = np.full(options.size, 1.0 / options.size)
    else:
        probabilities = np.asarray(p)
        if probabilities.shape != options.shape:
            raise ValueError("Choice probabilities must match the options.")

    remaining = np.ones(options.size, dtype=bool)
    mass = 1.0
    for i, draw in enumerate(draws):
        matches = np.flatnonzero(options == draw)
        if replace:
            # Repeated option values contribute all of their probability mass.
            mass *= np.sum(probabilities[matches])
        else:
            available = matches[remaining[matches]]
            if not available.size:
                return np.float64(0.0)
            mass *= available.size / (options.size - i)
            remaining[available[0]] = False
    return round_float(mass, res=1e-6)


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


def get_location_scale_pdf(randname, loc=0.0, scale=1.0, size=None):
    """
    Get a location-scale density without passing NumPy's size to SciPy.

    The sample size controls generation, not the density of supplied values.
    Randname is the corresponding SciPy distribution name.

    Examples
    --------
    >>> get_location_scale_pdf("laplace", 0.0, 2.0, (2, 3))(0.0)
    np.float64(0.25)
    """
    return get_scipy_pdf(randname, loc=loc, scale=scale)


def get_exp_ray_pdf(randname, scale=1.0, size=None):
    """
    Get exponential or Rayleigh density with NumPy's scale and sample size.

    Both generators use a zero location. Size controls generation, not the
    density of the supplied samples.

    Examples
    --------
    >>> get_exp_ray_pdf("rayleigh", 2)(2.0)
    np.float64(0.3032653298563167)
    >>> get_exp_ray_pdf("rayleigh", 2, 2)(2.0)
    np.float64(0.3032653298563167)
    >>> get_exp_ray_pdf("exponential", 1)(0.0)
    np.float64(1.0)
    >>> get_exp_ray_pdf("exponential", 1, (2, 3))(0.0)
    np.float64(1.0)
    """
    if randname == 'exponential':
        randname = "expon"
    return get_scipy_pdf(randname, scale=scale)


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


def get_uniform_pdf(low=0.0, high=1.0, size=None):
    """Get a uniform density using NumPy's lower and upper bounds.

    NumPy specifies endpoints; SciPy specifies location and interval width.
    The optional draw size does not change the density of the supplied values.

    Examples
    --------
    >>> get_uniform_pdf(5.0, 6.0)(5.5)
    np.float64(1.0)
    >>> get_uniform_pdf(5.0, 6.0)(7.0)
    np.float64(0.0)
    """
    return get_scipy_pdf("uniform", loc=low,
                         scale=np.asarray(high) - np.asarray(low))


def get_pareto_pdf(a, size=None):
    """
    Get the Lomax density corresponding to NumPy's Pareto II draws.

    NumPy's samples start at zero, unlike SciPy's Pareto I distribution.
    The optional draw size does not change the density of the supplied values.

    Examples
    --------
    >>> get_pareto_pdf(3.0)(0.0)
    np.float64(3.0)
    >>> get_pareto_pdf(3.0)(1.0)
    np.float64(0.1875)
    """
    return get_scipy_pdf("lomax", a)


def get_lognormal_pdf(mean=0.0, sigma=1.0, size=None):
    """
    Get a lognormal density using NumPy's defaults for the underlying normal.

    Size controls sample generation and does not change the density.

    Examples
    --------
    >>> get_lognormal_pdf(0, .25)(1.0)
    np.float64(1.5957691216057308)
    >>> get_lognormal_pdf()(1.0)
    np.float64(0.3989422804014327)
    """
    return get_scipy_pdf("lognorm", sigma, scale=np.exp(mean))


def get_gamma_pdf(shape, scale=1.0, size=None):
    """
    Get the joint Gamma density using NumPy's shape and scale parameters.

    Size controls sample generation and is not a density parameter.

    Examples
    --------
    >>> get_gamma_pdf(1.0, 2.0)(0.0)
    np.float64(0.5)
    """
    return get_scipy_pdf("gamma", shape, scale=scale)


def get_standard_gamma_pdf(shape, size=None, dtype=np.float64, out=None):
    """
    Get the joint density of NumPy standard Gamma draws (unit scale).

    Size, dtype, and out control generation, not the density of supplied values.

    Examples
    --------
    >>> get_standard_gamma_pdf(1.0, 3)([0.0, 0.0, 0.0])
    np.float64(1.0)
    """
    return get_gamma_pdf(shape)


def get_standard_normal_pdf(size=None, dtype=np.float64, out=None):
    """Get standard normal density, accepting generation-only size/dtype/out.

    The density has zero location and unit scale for every requested draw shape.
    The output buffer is not modified when evaluating the density.
    """
    return get_scipy_pdf("norm")


def get_standard_cauchy_pdf(size=None):
    """Get standard Cauchy density without using sample size as location."""
    return get_scipy_pdf("cauchy")


def get_standard_t_pdf(df, size=None):
    """
    Get the joint density of independent numpy.random.standard_t draws.

    Use univariate Student t factors, including broadcast degrees of freedom.
    The optional draw size does not change the density of the supplied values.

    Examples
    --------
    >>> get_standard_t_pdf(1)([0.0])
    np.float64(0.31830988618379075)
    """
    return get_scipy_pdf("t", df=df)


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


def get_vonmises_pdf(mu, kappa, size=None):
    """Get a von Mises density using NumPy's mean and concentration arguments.

    SciPy uses kappa as a shape parameter and mu as its circular location.
    The optional draw size does not change the density of the supplied values.

    Examples
    --------
    >>> bool(np.isclose(get_vonmises_pdf(0.0, 0.0)(0.0), 1 / (2 * np.pi)))
    True
    """
    return get_scipy_pdf("vonmises", kappa, loc=mu)


def get_wald_pdf(mean, scale, size=None):
    """
    Get the inverse Gaussian density using NumPy's Wald parameters.

    SciPy's inverse Gaussian uses mean/scale as its shape and scale as its
    scale parameter, with zero location. Size only controls sample generation.

    Examples
    --------
    >>> bool(np.isclose(get_wald_pdf(2.0, 3.0)(2.0), np.sqrt(3 / (16 * np.pi))))
    True
    """
    return get_scipy_pdf("invgauss", np.asarray(mean) / np.asarray(scale),
                         scale=scale)


def get_multinomial_pmf(n, pvals, size=None):
    """Get the joint mass of multinomial count vectors on the last axis.

    NumPy's optional size controls generation and is not a PMF parameter.
    Leading axes are independent draws; n and pvals retain their broadcasting.
    """
    return get_scipy_pmf("multinomial", n, pvals)


def get_dirichlet_pdf(alpha, size=None):
    """Get the joint density of complete Dirichlet vectors on the last axis.

    NumPy places components last; SciPy's PDF expects them first. Leading axes
    represent independent draws, and size is only a generation argument.
    """
    distribution = stats.dirichlet(alpha)
    num_components = len(distribution.alpha)

    def dirichlet_pdf(*x):
        values = array_x(x)
        if values.ndim == 0 or values.shape[-1] != num_components:
            raise ValueError("Dirichlet samples must contain all components on "
                             "their last axis.")
        if not values.size:
            return np.float64(1.0)
        return as_prob(distribution.pdf(values.reshape(-1, num_components).T))

    return dirichlet_pdf


def get_pfunc_for_dist(randname, *args):
    """
    Get the probability mass/density function corresponding to a numpy random draw.

    Uses a call to scipy.stats when available (with the correct arguments), otherwise
    uses a custom function provided in this module.

    Univariate discrete and shape-family sampling sizes are accepted without
    shifting the support or changing distribution parameters.
    Poisson's omitted rate uses NumPy's default of one.

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
    shape_funcs = {'beta': ('beta', 2), 'f': ('f', 2),
                   'chisquare': ('chi2', 1),
                   'noncentral_chisquare': ('ncx2', 2),
                   'noncentral_f': ('ncf', 3),
                   'power': ('powerlaw', 1), 'weibull': ('weibull_min', 1)}
    same_funcs = ['multivariate_normal']
    discrete_funcs = {'poisson': ('poisson', 1), 'zipf': ('zipf', 1),
                      'binomial': ('binom', 2), 'geometric': ('geom', 1),
                      'logseries': ('logser', 1), 'negative_binomial': ('nbinom', 2)}
    location_scale_funcs = {'normal': 'norm', 'laplace': 'laplace',
                            'logistic': 'logistic', 'gumbel': 'gumbel_r'}
    different_funcs_pmf = {'multivariate_hypergeometric': 'multivariate_hypergeom'}

    match randname:
        case str if randname in location_scale_funcs:
            return get_location_scale_pdf(location_scale_funcs[randname], *args)
        case str if randname in same_funcs:
            return get_scipy_pdf(randname, *args)
        case 'multinomial':
            return get_multinomial_pmf(*args)
        case str if randname in discrete_funcs:
            scipy_name, num_params = discrete_funcs[randname]
            if randname == 'poisson' and not args:
                args = (1.0,)
            if len(args) not in (num_params, num_params + 1):
                raise TypeError(randname + " requires " + str(num_params) +
                                " distribution parameter(s) and an optional size.")
            # NumPy's optional size must not become SciPy's location parameter.
            return get_scipy_pmf(scipy_name, *args[:num_params])
        case str if randname in shape_funcs:
            scipy_name, num_params = shape_funcs[randname]
            if len(args) not in (num_params, num_params + 1):
                raise TypeError(randname + " requires " + str(num_params) +
                                " shape parameter(s) and an optional size.")
            return get_scipy_pdf(scipy_name, *args[:num_params])
        case str if randname in different_funcs_pmf:
            return get_scipy_pmf(different_funcs_pmf[randname], *args)
        case str if randname in ['exponential', 'rayleigh']:
            return get_exp_ray_pdf(randname, *args)
        case 'hypergeometric':
            return get_hypergeometric_pmf(*args)
        case 'uniform':
            return get_uniform_pdf(*args)
        case 'pareto':
            return get_pareto_pdf(*args)
        case 'lognormal':
            return get_lognormal_pdf(*args)
        case 'gamma':
            return get_gamma_pdf(*args)
        case 'standard_gamma':
            return get_standard_gamma_pdf(*args)
        case 'standard_normal':
            return get_standard_normal_pdf(*args)
        case 'standard_cauchy':
            return get_standard_cauchy_pdf(*args)
        case 'standard_t':
            return get_standard_t_pdf(*args)
        case 'triangular':
            return get_triangular_pdf(*args)
        case 'vonmises':
            return get_vonmises_pdf(*args)
        case 'wald':
            return get_wald_pdf(*args)
        case 'dirichlet':
            return get_dirichlet_pdf(*args)
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
    >>> bool(np.isclose(get_prob_for_rand([0, 0], "binomial", 1, 0.5, 2), 0.25))
    True
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
