# -*- coding: utf-8 -*-
"""
Description: A module for defining randon properties for use in blocks.

Has Classes and Functions:

- :class:`Rand`: Superclass for Block random properties.

- :func:`get_pdf_for_rand`: Gets the corresponding probability mass/density for
random sample x from 'randname' function in numpy.

- :func:`get_scipy_pdf_helper`: Gets probability mass/density for the outcome x from the
distribution "randname". Used as a helper function in determining stochastic model state
probability

- :func:`get_pdf_for_dist`: Gets the corresponding probability mass/density (from scipy)
for outcome x for probability distributions with name 'randname' in numpy.
"""
from scipy import stats
from recordclass import dataobject, asdict, astuple
import numpy as np
import math
from fmdtools.define.container.base import BaseContainer
import copy

from fmdtools.analyze.history import History, init_hist_iter


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

    Rand is meant to be extended in model definition with random states, e.g.:

    class RandState(State):
        noise: float=1.0
    class ExampleRand(Rand):
        s = RandState()
        run_stochastic: bool = True

    Which enables the use of set_rand, update_stochastic_states, etc for updating
    these states with methods called from the rng.
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
                                    rng=np.random.default_rng(self.seed))
        super().__init__(*args)
        if 's' in self.__fields__:
            self.s = self.s.__class__()
            self.s.set_atts(**s_kwargs)
        if self.seed == None:
            raise Exception("Invalid seed: None")

    def get_rand_states(self, auto_update_only=False):
        rand_states = asdict(self.s)
        if auto_update_only:
            rand_states = {state: vals for state,
                           vals in rand_states if hasattr(self.s, state+"_update")}
        return rand_states

    def return_mutables(self):
        if 's' in self.__fields__:
            return astuple(self.s)
        else:
            return ()

    def set_rand(self, statename, methodname, *args):
        """
        Update the given random state with a given method and arguments
        (if in run_stochastic mode)

        Parameters
        ----------
        statename : str
            name of the random state defined in assoc_rand_state(s)
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
                value_pds = get_pdf_for_rand(newvalue, methodname, args)
                self.probs.append(value_pds)

    def return_probdens(self):
        if self.probs:
            state_pd = np.prod(self.probs)
        else:
            state_pd = 1.0
        return state_pd

    def update_stochastic_states(self):
        """Updates the defined stochastic states defined to auto-update."""
        if hasattr(self, 's'):
            if self.run_stochastic == 'track_pdf':
                self.probs.clear()
            for state in self.s.__fields__:
                if hasattr(self.s, state+"_update"):
                    self.set_rand(state, getattr(self.s, state+'_update')[0],
                                  *getattr(self.s, state+'_update')[1])

    def reset(self):
        """Resets Rand to the initial state."""
        self.probs.clear()
        if 's' in self.__fields__:
            self.s.reset()
        self.rng = np.random.default_rng(self.seed)

    def update_seed(self, seed):
        """Updates the random seed to the given value"""
        self.seed = seed
        BitGen = type(self.rng.bit_generator)
        self.rng.bit_generator.state = BitGen(seed).state

    def assign(self, other_rand):
        if hasattr(self, 's'):
            self.s.assign(other_rand.s)
        self.seed = other_rand.seed
        self.rng = np.random.default_rng(self.seed)
        self.rng.__setstate__(other_rand.rng.__getstate__())
        self.probs = copy.copy(other_rand.probs)
        self.run_stochastic = other_rand.run_stochastic

    def to_default(self, *statenames):
        """Resets given random states to their default values"""
        for statename in statenames:
            default = self.s.__default_vals__[self.s.__fields__.index(statename)]
            self.s[statename] = default

    def create_hist(self, timerange, track):
        """
        Creates a History corresponding to Rand

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'.
                The default is None.

        Returns
        -------
        hist : History
            History of fields specified in track.
        """
        h = History()
        track = self.get_track(track)
        if self.run_stochastic == 'track_pdf' and 'track_pdf' in track:
            h.init_att('probdens', self.return_probdens(),
                       timerange=timerange, track='all')
        if 's' in track and hasattr(self, 's'):
            h['s'] = init_hist_iter('s', self.s, timerange=timerange, track=track)
        return h

    def copy(self):
        """Copy the rand."""
        cop = self.__class__().__init__()
        cop.assign(self)
        return cop


def get_pdf_for_rand(x, randname, args):
    """
    Gets the corresponding probability mass/density for
    for random sample x from 'randname' function in numpy.

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
    if type(x) not in [np.ndarray, list]:
        x = [x]
    if randname == 'integers':
        if len(args) == 1:
            pd = [1/args[0] for x in x]
        elif len(args) >= 2:
            pd = [1/(args[1]-args[0]) for x in x]
    elif randname == 'random':
        pd = [1 for x in x]
    elif randname == 'bytes':
        raise Exception("Not able to calculate pdf for bytes")
    elif randname == 'choice':
        if type(args[0]) == int:
            options = [*np.arange(args[0])]
        else:
            options = args[0]
        if len(args) == 4:
            p = args[3]
        else:
            p = [1/len(options) for i in options]
        pd = [p[options.index(i)] for i in x]
    elif randname in ['shuffle', 'permutation']:
        pd = [1/math.factorial(len(args[0]))]
    elif randname == 'permuted':
        if len(args) > 1 and type(args[0]) == np.ndarray:
            pd = [1/math.factorial(args[0].shape(args[1]))]
        else:
            pd = [1/math.factorial(len(args[0]))]
    else:
        pd = get_pdf_for_dist(x, randname, args)
    if type(pd) == list:
        pd = np.array(pd)
    elif type(pd) != np.ndarray:
        pd = np.array([pd])
    return pd


def get_scipy_pdf_helper(x, randname, args, pmf=False):
    """
    Gets probability mass/density for the outcome x from the distribution "randname"
    with arguments "args".

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
    Gets the corresponding probability mass/density (from scipy) for outcome x
    for probability distributions with name 'randname' in numpy.

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
