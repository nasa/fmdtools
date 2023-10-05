# -*- coding: utf-8 -*-
"""
Description: A module for defining ways of sampling model faults/parameters.

Has classes and functions:
    - :class:`NominalApproach`:     Class for defining a set of nominal scenarios (i.e., parameters) to simulate over.
    - :class:`SampleApproach`:      Class for defining a set of fault scenarios (modes and times) to sample over.
    - :func:`find_overlap_n`:       Function for finding overlap between given intervals. Used to sample joint fault modes.

"""

import numpy as np
from collections.abc import Hashable
from operator import itemgetter
from fmdtools.define.common import t_key, eq_units
from fmdtools.sim.scenario import Scenario, SingleFaultScenario, JointFaultScenario, NominalScenario, ParamScenario, Injection
import itertools
import copy
from warnings import warn

class PhaseMap(object):
    """
    Mapping of phases to times used to create scenario samples.

    Phases and modephases may be generated from Result.get_phases.

    Parameters
    ----------
    phases : dict
        Phases the mode will be injected during. Used to determine opportunity
        factor defined by the dict in fault.phases.
        Has structure {'phase1': [starttime, endtime]}. The default is {}.
    modephases : dict, optional
        Modes that the phases occur in. Used to determine opportunity vector defined
        by the dict in fault.phases (if .phases maps to modes of occurence an not
        phases).
        Has structure {'on': {'on1', 'on2', 'on3'}}. The default is {}.
    dt: float
        Timestep defining phases.
    """

    def __init__(self, phases, modephases={}, dt=1.0):
        self.phases = phases
        self.modephases = modephases
        self.dt = dt

    def find_phase(self, time, dt=1.0):
        """
        Find the phase that a time occurs in.

        Parameters
        ----------
        time : float
            Occurence time.

        Returns
        -------
        phase : str
            Name of the phase time occurs in.
        """
        for phase, times in self.phases.items():
            if times[0] <= time < times[1]+dt:
                return phase
        raise Exception("time "+str(time)+" not in phases: "+str(self.phases))

    def find_modephase(self, phase):
        """
        Find the mode in modephases that a given phase occurs in.

        Parameters
        ----------
        phase : str
            Name of the phase (e.g., 'on1').

        Returns
        -------
        mode : str
            Name of the corresponding mode (e.g., 'on').

        Examples
        --------
        >>> pm = PhaseMap({}, {"on": {"on0", "on1", "on2"}})
        >>> pm.find_modephase("on1")
        'on'
        """
        for mode, mode_phases in self.modephases.items():
            if phase in mode_phases:
                return mode
        raise Exception("Phase "+phase+" not in modephases: "+str(self.modephases))

    def find_base_phase(self, time):
        """
        Find the phase or modephase (if provided) that the time occurs in.

        Parameters
        ----------
        time : float
            Time to check.

        Returns
        -------
        phase : str
            Phase or modephase the time occurs in.
        """
        phase = self.find_phase(time)
        if self.modephases:
            phase = self.find_modephase(phase)
        return phase

    def calc_samples_in_phases(self, *times):
        """
        Calculate the number of times the provided times show up in phases/modephases.

        Parameters
        ----------
        *times : float
            Times to check

        Returns
        -------
        phase_times : TYPE
            DESCRIPTION.

        Examples
        --------
        >>> pm = PhaseMap(phases={'on':[0, 3], 'off': [4, 5]})
        >>> pm.calc_samples_in_phases(1,2,3,4,5)
        {'on': 3, 'off': 2}
        >>> pm = PhaseMap({'on':[0, 3], 'off': [4, 5]}, {'oper': {'on', 'off'}})
        >>> pm.calc_samples_in_phases(1,2,3,4,5)
        {'oper': 5}
        """
        if self.modephases:
            phase_times = {ph: 0 for ph in self.modephases}
        else:
            phase_times = {ph: 0 for ph in self.phases}
        for time in times:
            phase = self.find_phase(time)
            if self.modephases:
                phase = self.find_modephase(phase)
            phase_times[phase] += 1
        return phase_times

    def calc_phase_time(self, phase):
        """
        Calculate the length of a phase.

        Parameters
        ----------
        phase : str
            phase to calculate.
        phases : dict
            dict of phases and time intervals.
        dt : float, optional
            Timestep length. The default is 1.0.

        Returns
        -------
        phase_time : float
            Time of the phase


        Examples
        --------
        >>> pm = PhaseMap({"on": [0, 4], "off": [5, 10]})
        >>> pm.calc_phase_time("on")
        5.0
        """
        phasetimes = self.phases[phase]
        phase_time = phasetimes[1] - phasetimes[0] + self.dt
        return phase_time

    def calc_modephase_time(self, modephase):
        """
        Calculate the amount of time in a mode, given that mode maps to multiple phases.

        Parameters
        ----------
        modephase : str
            Name of the mode to check.
        phases : dict
            Dict mapping phases to times.
        modephases : dict
            Dict mapping modes to phases
        dt : float, optional
            Timestep. The default is 1.0.

        Returns
        -------
        modephase_time : float
            Amount of time in the modephase

        Examples
        --------
        >>> pm = PhaseMap({"on1": [0, 1], "on2": [2, 3]}, {"on": {"on1", "on2"}})
        >>> pm.calc_modephase_time("on")
        4.0
        """
        modephase_time = sum([self.calc_phase_time(mode_phase)
                              for mode_phase in self.modephases[modephase]])
        return modephase_time

    def calc_scen_exposure_time(self, time):
        """
        Calculate the time for the phase/modephase at the given time.

        Parameters
        ----------
        time : float
            Time within the phase.

        Returns
        -------
        exposure_time : float
            Exposure time of the given phasemap.
        """
        phase = self.find_phase(time)
        if self.modephases:
            phase = self.find_modephase(phase)
            return self.calc_modephase_time(phase)
        else:
            return self.calc_phase_time(phase)

    def get_phase_times(self, phase):
        """
        Get the set of discrete times in the interval for a phase.

        Parameters
        ----------
        phase : str
            Name of a phase in phases or modephases.

        Returns
        -------
        all_times : list
            List of times corresponding to the phase

        Examples
        --------
        >>> pm = PhaseMap({"on1": [0, 1], "on2": [2, 3]}, {"on": {"on1", "on2"}})
        >>> pm.get_phase_times('on1')
        [0.0, 1.0]
        >>> pm.get_phase_times('on2')
        [2.0, 3.0]
        >>> pm.get_phase_times('on')
        [0.0, 1.0, 2.0, 3.0]
        """
        if phase in self.modephases:
            phases = self.modephases[phase]
            intervals = [self.phases[ph] for ph in phases]
        elif phase in self.phases:
            intervals = [self.phases[phase]]
        int_times = [gen_interval_times(i, self.dt) for i in intervals]
        all_times = list(set(np.concatenate(int_times)))
        all_times.sort()
        return all_times

    def get_sample_times(self, *phases_to_sample):
        """
        Get the times to sample for the given phases.

        Parameters
        ----------
        *phases_to_sample : str
            Phases to sample. If none are provided, the full set of phases or modephases
            is used.

        Returns
        -------
        sampletimes : dict
            dict of times to sample with structure {'phase1': [t0, t1, t2], ...}

        Examples
        --------
        >>> pm = PhaseMap({"on1": [0, 4], "on2": [5, 6]}, {"on": {"on1", "on2"}})
        >>> pm.get_sample_times("on1")
        {'on1': [0.0, 1.0, 2.0, 3.0, 4.0]}
        >>> pm.get_sample_times("on")
        {'on': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        >>> pm.get_sample_times("on1", "on2")
        {'on1': [0.0, 1.0, 2.0, 3.0, 4.0], 'on2': [5.0, 6.0]}
        """
        if not phases_to_sample:
            if self.modephases:
                phases_to_sample = tuple(self.modephases)
            elif self.phases:
                phases_to_sample = tuple(self.phases)
        sampletimes = {}
        for phase in phases_to_sample:
            sampletimes[phase] = self.get_phase_times(phase)
        return sampletimes


def gen_interval_times(interval, dt):
    """Generate the times in a given interval given the timestep dt."""
    return np.arange(interval[0], interval[-1] + dt, dt)



class NominalApproach():
    """
    Class for defining sets of nominal simulations.

    To explain, a given system
    may have a number of input situations (missions, terrain, etc) which the
    user may want to simulate to ensure the system operates as desired. This
    class (in conjunction with propagate.nominal_approach()) can be used to
    perform these simulations.

    Attributes
    ----------
    scenarios : dict
        scenarios to inject based on the approach
    num_scenarios : int
        number of scenarios in the approach
    ranges : dict
        dict of the parameters defined in each method for the approach
    """

    def __init__(self):
        """Instantiate NominalApproach (simulation p are defined using methods)."""
        self.scenarios = {}
        self.num_scenarios = 0
        self.ranges = {}

    def __repr__(self):
        all_range_str=""
        for r, rangedict in self.ranges.items():
            rangestr = '\n-'+r+' ('+str(len(rangedict['scenarios']))+' scenarios)'
            rangedict = {k:v for k,v in rangedict.items() if k not in {'scenarios', 'levels','num_pts'}}
            if 'seeds' in rangedict: rangedict['seeds'] = len(rangedict['seeds'])
            subrangestr = "\n----"+"\n----".join([k+': '+str(v) for k,v in rangedict.items()])
            all_range_str=all_range_str+rangestr+subrangestr
        #rangestr = "\n- "+"\n- ".join([k+": "+str(len(v['scenarios']))+' scenarios' for k,v in self.ranges.items()])
        return "NominalApproach ("+str(self.num_scenarios)+" scenarios) with ranges:"+all_range_str

    def add_seed_replicates(self, rangeid, seeds):
        """
        Generates an approach with different seeds to use for the model's internal stochastic behaviors

        Parameters
        ----------
        rangeid : str
            Name for the set of replicates
        seeds : int/list
            Number of seeds (if an int) or a list of seeds to use.
        """
        if type(seeds)==int: seeds = np.random.SeedSequence.generate_state(np.random.SeedSequence(),seeds)
        self.ranges[rangeid] = {'seeds':seeds, 'scenarios':[]}
        for i in range(len(seeds)):
            self.num_scenarios+=1
            scenname = rangeid+'_'+str(self.num_scenarios)
            self.scenarios[scenname]= NominalScenario(rangeid=rangeid, r={'seed':int(seeds[i])}, prob=1/len(seeds), name=scenname)
            self.ranges[rangeid]['scenarios'].append(scenname)

    def add_param_replicates(self,paramfunc, rangeid, replicates, *args, ind_seeds=True, **kwargs):
        """
        Adds a set of repeated scenarios to the approach. For use in (external) random scenario generation.

        Parameters
        ----------
        paramfunc : method
            Python method which generates a set of model parameters given the input arguments.
            method should have form: method(fixedarg, fixedarg..., inputarg=X, inputarg=X)
        rangeid : str
            Name for the set of replicates
        replicates : int
            Number of replicates to use
        *args : any
            arguments to send to paramfunc
        ind_seeds : Bool/list
            Whether the models should be run with different seeds (rather than the same seed). Default is True
            When a list is provided, these seeds are are used. Must be of length replicates.
        **kwargs : any
            keyword arguments to send to paramfunc
        """
        if ind_seeds==True:         seeds = np.random.SeedSequence.generate_state(np.random.SeedSequence(),replicates)
        elif type(ind_seeds)==list: 
            if len(ind_seeds)!=replicates: raise Exception("list ind_seeds must be of length replicates")
            else:                   seeds=ind_seeds
        else:                       seeds = [None for i in range(replicates)]
        self.ranges[rangeid] = {'fixedargs':args, 'inputranges':kwargs, 'scenarios':[], 'num_pts' : replicates, 'paramfunc':paramfunc}
        for i in range(replicates):
            self.num_scenarios+=1
            p = paramfunc(*args, **kwargs)
            scenname = rangeid+'_'+str(self.num_scenarios)
            self.scenarios[scenname]=ParamScenario(name=scenname,
                                                   rangeid=rangeid,
                                                   p=p,
                                                   r={'seed':int(seeds[i])},
                                                   paramfunc=paramfunc,
                                                   fixedargs=args,
                                                   prob=1/replicates)
            self.ranges[rangeid]['scenarios'].append(scenname)

    def get_param_scens(self, rangeid, *level_params):
        """
        Returns the scenarios of a range associated with given parameter ranges

        Parameters
        ----------
        rangeid : str
            Range id to check
        level_params : str (multiple)
            Level parameters iterate over

        Returns
        -------
        param_scens : dict
            The scenarios associated with each level of parameter (or joint parameters)
        """
        inputranges = {param:self.ranges[rangeid]['inputranges'][param] for param in level_params}
        partialspace= self.range_to_space(inputranges)
        partialspace = [tuple([a if isinstance(a, Hashable) else str(a) for a in p]) for p in partialspace]
        param_scens = {(p if len(p)>1 else p[0]):set() for p in partialspace}
        full_indices = list(self.ranges[rangeid]['inputranges'].keys())
        inds = [full_indices.index(param) for param in level_params]
        
        for xvals, scenarios in self.ranges[rangeid]['levels'].items():
            new_index = itemgetter(*inds)(xvals)
            if type(scenarios)==str: scenarios = [scenarios]
            param_scens[new_index].update(scenarios)
        return param_scens

    def range_to_space(self,inputranges):
        ranges = (np.arange(*arg) if type(arg)==tuple else tuple(arg) for k,arg in inputranges.items())
        space = [x for x in itertools.product(*ranges)]
        return space

    def add_param_ranges(self,paramfunc, rangeid, *args, replicates=1, seeds='shared',set_args={}, **kwargs):
        """
        Adds a set of scenarios to the approach.

        Parameters
        ----------
        paramfunc : method
            Python method which generates a set of model parameters given the input arguments.
            method should have form: method(fixedarg, fixedarg..., inputarg=X, inputarg=X)
        rangeid : str
            Name for the range being used. Default is 'nominal'
        *args: specifies values for positional args of paramfunc.
            May be given as a fixed float/int/dict/str defining a set value for positional arguments
        replicates : int
            Number of points to take over each range (for random parameters). Default is 1.
        seeds : str/list
            Options for seeding models/replicates: (Default is 'shared')
                - 'shared' creates random seeds and shares them between parameters and models
                - 'independent' creates separate random seeds for models and parameter generation
                - 'keep_model' uses the seed provided in the model for all of the model
            When a list is provided, these seeds are are used (and shared). Must be of length replicates.
        set_args : dict
            Dictionary of lists of values for each param e.g., {'param1':[value1, value2, value3]}
        **kwargs : specifies range for keyword args of paramfunc
            May be given as a fixed float/int/dict/str (k=value) defining a set value for the range (if not the default) or
            as a tuple k=(start, end, step) for the range, or
        """
        inputranges = {ind:rangespec for ind,rangespec in enumerate(args) if type(rangespec)==tuple}
        fixedkwargs = {k:v for k,v in kwargs.items() if not type(v)==tuple}
        inputranges = {k:v for k,v in kwargs.items() if type(v)==tuple}
        inputranges.update(set_args)
        fullspace = self.range_to_space(inputranges)
        inputnames = list(inputranges.keys())  
        
        if type(seeds)==list: 
            if len(seeds)!=replicates: raise Exception("list seeds must be of length replicates")
        else: seedstr=seeds;  seeds=np.random.SeedSequence.generate_state(np.random.SeedSequence(),replicates)
        if seedstr=='shared':         mdlseeds=seeds
        elif seedstr=='independent':  mdlseeds=np.random.SeedSequence.generate_state(np.random.SeedSequence(),replicates)
        elif seedstr=='keep_model':   mdlseeds= [None for i in range(replicates)]
        
        self.ranges[rangeid] = {'fixedargs':args, 'fixedkwargs':fixedkwargs, 'inputranges':inputranges, 'scenarios':[], 'num_pts' : len(fullspace), 'levels':{}, 'replicates':replicates, 'paramfunc':paramfunc}
        for xvals in fullspace:
            inputparams = {**{name:xvals[i] for i,name in enumerate(inputnames)}, **fixedkwargs}
            level_key = tuple([x if isinstance(x,Hashable) else str(x) for x in xvals])
            if replicates>1:    self.ranges[rangeid]['levels'][level_key]=[]
            for i in range(replicates):
                np.random.seed(seeds[i])
                self.num_scenarios+=1
                p = paramfunc(*args, **inputparams)
                scenname = rangeid+'_'+str(self.num_scenarios)
                self.scenarios[scenname]= ParamScenario(name=scenname, 
                                                        rangeid=rangeid, 
                                                        p=p, 
                                                        r={'seed':int(seeds[i])}, 
                                                        paramfunc=paramfunc,
                                                        inputparams=inputparams,
                                                        fixedargs=args,
                                                        fixedkwargs=fixedkwargs,
                                                        prob = 1/(len(fullspace)*replicates))

                self.ranges[rangeid]['scenarios'].append(scenname)
                if replicates>1:    self.ranges[rangeid]['levels'][level_key].append(scenname)
                else:               self.ranges[rangeid]['levels'][level_key]=scenname

    def update_factor_seeds(self, rangeid, inputparam, seeds='new'):
        """
        Changes/randomizes the seeds along a given factor in a range

        Parameters
        ----------
        rangeid : str
            Name of the range being updated
        inputparam : str
            Name of the parameter to vary the seeds over
        seeds : str/list, optional
            List of seeds to update to. The default is 'new', which picks them randomly
        """
        param_loc = [*self.ranges[rangeid]['inputranges'].keys()].index(inputparam)
        levels = [i for i in range(*self.ranges[rangeid]['inputranges'][inputparam])]
        if type(seeds) == list:
            if len(seeds)!=levels: raise Exception("Seeds (len: "+str(len(seeds))+") should math number of levels for "+inputparam+': '+str(len(levels)))
        elif seeds=="new":
            seeds = np.random.SeedSequence.generate_state(np.random.SeedSequence(), len(levels))
        else: raise Exception("Invalid option for seeds: "+str(seeds))
        for i, level in enumerate(levels):
            scens = [scen for lev, scen in self.ranges[rangeid]['levels'].items() if lev[param_loc]==level]
            for scen in scens:
                self.scenarios[scen].r['seed'] = int(seeds[i])

    def assoc_probs(self, rangeid, prob_weight=1.0, **inputpdfs):
        """
        Associates a probability model (assuming variable independence) with a 
        given previously-defined range of scenarios using given pdfs

        Parameters
        ----------
        rangeid : str
            Name of the range to apply the probability model to.
        prob_weight : float, optional
            Overall probability for the set of scenarios (to use if adding more ranges 
            or if the range does not cover the space of probability). The default is 1.0.
        **inputpdfs : key=(pdf, params)
            pdf to associate with the different variables of the model. 
            Where the pdf has form pdf(x, **kwargs) where x is the location and **kwargs is parameters
            (for example, scipy.stats.norm.pdf)
            and params is a dictionary of parameters (e.g., {'mu':1,'std':1}) to use '
            as the key/parameter inputs to the pdf
        """
        scenprobs = dict()
        for scenname in self.ranges[rangeid]['scenarios']:
            inputparams = self.scenarios[scenname].inputparams
            inputprobs = [inpdf[0](inputparams[name], **inpdf[1]) for name, inpdf in inputpdfs.items()]
            scenprobs[scenname] = np.prod(inputprobs)
        totprobs = sum([*scenprobs.values()])
        for scenname in self.ranges[rangeid]['scenarios']:
            prob = scenprobs[scenname]*prob_weight/totprobs
            self.scenarios[scenname] = self.scenarios[scenname].copy_with(prob=prob)

    def add_rand_params(self, paramfunc, rangeid, *fixedargs, prob_weight=1.0, replicates=1000, seeds='shared', **randvars):
        """
        Adds a set of random scenarios to the approach.

        Parameters
        ----------
        paramfunc : method
            Python method which generates a set of model parameters given the input arguments.
            method should have form: method(fixedarg, fixedarg..., inputarg=X, inputarg=X)
        rangeid : str
            Name for the range being used. Default is 'nominal'
        prob_weight : float (0-1)
            Overall probability for the set of scenarios (to use if adding more ranges). Default is 1.0
        *fixedargs : any
            Fixed positional arguments in the parameter generator function. 
            Useful for discrete modes with different parameters.
        seeds : str/list
            Options for seeding models/replicates: (Default is 'shared')
                - 'shared' creates random seeds and shares them between parameters and models
                - 'independent' creates separate random seeds for models and parameter generation
                - 'keep_model' uses the seed provided in the model for all of the model
            When a list is provided, these seeds are are used (and shared). Must be of length replicates.
        **randvars : key=tuple
            Specification for each random input parameter, specified as 
            input = (randfunc, param1, param2...)
            where randfunc is the method producing random outputs (e.g. numpy.random.rand)
            and the successive parameters param1, param2, etc are inputs to the method
        """
        if type(seeds) == list:
            if len(seeds) != replicates:
                raise Exception("list seeds must be of length replicates")
        else:
            seedstr = seeds
            seeds = np.random.SeedSequence.generate_state(
                np.random.SeedSequence(), replicates)
        if seedstr == 'shared':
            mdlseeds = seeds
        elif seedstr == 'independent':
            mdlseeds = np.random.SeedSequence.generate_state(
                np.random.SeedSequence(), replicates)
        elif seedstr == 'keep_model':
            mdlseeds = [None for i in range(replicates)]

        self.ranges[rangeid] = {'fixedargs': fixedargs,
                                'randvars': randvars, 'scenarios': [], 'num_pts': replicates}
        for i in range(replicates):
            self.num_scenarios += 1
            np.random.seed(seeds[i])
            inputparams = {name: (ins() if callable(ins) else ins[0](*ins[1:]))
                           for name, ins in randvars.items()}
            p = paramfunc(*fixedargs, **inputparams)
            scenname = rangeid+'_'+str(self.num_scenarios)
            self.scenarios[scenname] = ParamScenario(name=scenname,
                                                     rangeid=rangeid,
                                                     p=p,
                                                     r={'seed': int(mdlseeds[i])},
                                                     paramfunc=paramfunc,
                                                     fixedargs=fixedargs,
                                                     inputparams=inputparams,
                                                     prob=prob_weight/replicates)
            self.ranges[rangeid]['scenarios'].append(scenname)

    def copy(self):
        """Copy the given sampleapproach. Used in nested scenario sampling."""
        newapp = NominalApproach()
        newapp.scenarios = copy.deepcopy(self.scenarios)
        newapp.ranges = copy.deepcopy(self.ranges)
        newapp.num_scenarios = self.num_scenarios
        return newapp

    def get_param_scens(self, *params, only_params=False, default="NA"):
        data = [(scenname, scen.get_params(*params, default=default))
                for scenname, scen in self.scenarios.items()
                if not only_params or (default in scen.get_params(params, default=default))]
        return data


class SampleApproach():
    """
    Class for defining the sample approach to be used for a set of faults.

    Attributes
    ----------
    phases : dict
        phases given to sample the fault modes in
    globalphases : dict
        phases defined in the model
    modephases : dict
        Dictionary of modes associated with each state
    mode_phase_map : dict
        Mapping of modes to their corresponding phases
    tstep : float
        timestep defined in the model
    fxnrates : dict
        overall failure rates for each function
    comprates : dict
        overall failure rates for each component
    jointmodes : list
        (if any) joint fault modes to be injected in the approach
    rates/comprates/rates_timeless : dict
        rates of each mode (fxn, mode) in each model phase, structured
        {fxnmode: {phaseid:rate}}
    sampletimes : dict
        faults to inject at each time in each phase, structured
        {phaseid:time:fnxmode}
    weights : dict
        weight to put on each time each fault was injected, structured
        {fxnmode:phaseid:time:weight}
    sampparams : dict
        parameters used to sample each mode
    scenlist : list
        list of fault scenarios (dicts of faults and properties) that fault propagation
        iterates through
    scenids : dict
        a list of scenario ids associated with a given fault in a given phase,
        structured {(fxnmode,phaseid):listofnames}
    mode_phase_map : dict
        a dict of modes and their respective phases to inject with structure
        {fxnmode: {mode_phase_map: [starttime, endtime]}}
    units : str
        time-units to use in the approach probability model
    unit_factors : dict
        multiplication factors for converting some time units to others.
    """

    def __init__(self, mdl, faults='all', phases='global', modephases={},join_modephases=False, jointfaults={'faults':'None'}, 
                 sampparams={}, defaultsamp={'samp':'evenspacing','numpts':1}, reduce_to=False):
        """
        Initializes the sample approach for a given model

        Parameters
        ----------
        mdl : Model or Block
            Model to sample.
        faults : str/list/tuple, optional
            - The default is 'all', which gets all fault modes from the model.
            - 'single-component' uses faults from a single component to represent faults from all components 
            - 'single-function' uses faults from a single function to represent faults from that type
            - passing the function name only includes modes from that function
            - List of faults of form [(fxn, mode)] to inject in the model.
            -Tuple arguments 
                - ('mode type', 'mode','notmode'), gets all modes with 'mode' as a string (e.g. "mech", "comms", "loss" faults). 'notmode' (if given) specifies strings to remove
                - ('mode types', ('mode1', 'mode2')), gets all modes with the listed strings (e.g. "mech", "comms", "loss" faults)
                - ('mode name', 'mode'), gets all modes with the exact name 'mode'
                - ('mode names', ('mode1', 'mode2')), gets all modes with the exact names defined in the tuple
                - ('function class', 'Classname'), which gets all modes from a function with class 'Classname'
                - ('function classes', ('Classname1', 'Classname2')), which gets all modes from a function with the names in the tuple
                - ('single-component', ('fxnname2', 'fxnname2')), which specifies single-component modes in the given functions
        phases: dict or 'global' or list
            Local phases in the model to sample. 
                Dict has structure: {'Function':{'phase':[starttime, endtime]}}
                List has structure: ['phase1', 'phase2'] where phases are phases in mdl.sp.phases
            Defaults to 'global',here only the phases defined in mdl.sp.phases are used.
            Phases and modephases can be gotten from process.modephases(mdlhist)
        modephases: dict
            Dictionary of modes associated with each phase. 
            For use when the opportunity vector is keyed to modes and each mode is 
            entered multiple times in a simulation, resulting in 
            multiple phases associated with that mode. Has structure:
                {'Function':{'mode':{'phase','phase1', 'phase2'...}}}
                Phases and modephases can be gotten from process.modephases(mdlhist)
        join_modephases: bool
            Whether to join phases with the same modes defined in modephases. Default is False
        jointfaults : dict, optional
            Defines how the approach considers joint faults. The default is {'faults':'None'}. Has structure:
                - faults : float    
                    # of joint faults to inject. 'all' specifies all faults at the same time
                - jointfuncs :  bool 
                    determines whether more than one mode can be injected in a single function
                - pcond (optional) : float in range (0,1) 
                    conditional probabilities for joint faults. If not give, independence is assumed.
                - inclusive (optional) : bool
                    specifies whether the fault set includes all joint faults up to the given level, or only the given level
                    (e.g., True with 'all' means SampleApproach includes every combination of joint fault modes while
                           False with 'all' means SampleApproach only includes the joint fault mode with all faults)
                - limit jointphases (optional) : int
                    Limits the number of jointphases to sample (by randomly sampling them instead). Necessary when the
                    number of faults is large
        sampparams : dict, optional
            Defines how specific modes in the model will be sampled over time. The default is {}. 
            Has structure: {key: sampparam}, where a key may be 'fxnmode','fxnname','mode', 'phase', or ('fxnmode','phase') 
            and sampparam has structure:
                - 'samp' : str ('quad', 'fullint', 'evenspacing','randtimes','symrandtimes')
                    sample strategy to use (quadrature, full integral, even spacing, random times, likeliest, or symmetric random times)
                - 'numpts' : float
                    number of points to use (for evenspacing, randtimes, and symrandtimes only)
                - 'quad' : dict
                    dict with structure {'nodes'[nodelist], 'weights':weightlist}
                    where the nodes in the nodelist range between -1 and 1
                    and the weights in the weightlist sum to 2.
        defaultsamp : TYPE, optional
            Defines how the model will be sampled over time by default. The default is {'samp':'evenspacing','numpts':1}. Has structure:
                - 'samp' : str ('quad', 'fullint', 'evenspacing','randtimes','symrandtimes')
                    sample strategy to use (quadrature, full integral, even spacing, random times,likeliest, or symmetric random times)
                - 'numpts' : float
                    number of points to use (for evenspacing, randtimes, and symrandtimes only)
                - 'quad' : dict
                    dict with structure {'nodes'[nodelist], 'weights':weightlist}
                    where the nodes in the nodelist range between -1 and 1
                    and the weights in the weightlist sum to 2.
        reduce_to : int, optional
            Size of random sample to reduce the number of scenarios to (if any). Default is False.
        """
        self.unit_factors = {'sec':1, 'min':60,'hr':360,'day':8640,'wk':604800,'month':2592000,'year':31556952}
        mdl_phases = {v[0]:[v[1], v[2]] for v in mdl.sp.phases}
        
        if phases=='global':                self.globalphases = mdl_phases; self.phases = {}; self.modephases = modephases
        elif type(phases) in [list, set]:   self.globalphases = {ph:mdl_phases[ph] for ph in phases}; self.phases={}; self.modephases = modephases
        elif type(phases)==dict: 
            if   type(tuple(phases.values())[0])==dict:         self.globalphases = mdl_phases; self.phases = phases; self.modephases = modephases
            elif type(tuple(phases.values())[0][0]) in [int, float]:  self.globalphases = phases; self.phases ={}; self.modephases = modephases
            else:                                               self.globalphases = mdl_phases; self.phases = phases; self.modephases = modephases
        #elif type(phases)==set:    self.globalphases=mdl.phases; self.phases = {ph:mdl.phases[ph] for ph in phases}
        self.mdltype = mdl.__class__.__name__
        self.tstep = mdl.sp.dt
        self.units = mdl.sp.units
        self.fxns = mdl.get_fxns()

        self.init_modelist(mdl,faults, jointfaults)
        self.init_rates(mdl, jointfaults=jointfaults, modephases=modephases, join_modephases=join_modephases)
        self.create_sampletimes(mdl, sampparams, defaultsamp)
        self.create_scenarios()
        if reduce_to: self.reduce_scens_to_samp(reduce_to)
    def __repr__(self):
        modes=list(self._fxnmodes)
        if len(modes)>10:  modes=modes[0:10]+[["...more"]]
        modestr = "\n -"+"\n -".join([": ".join(mode) for mode in list(modes)])
        phases = {ph:tm[0] for fxnphases in self.mode_phase_map.values() for ph,tm in fxnphases.items()}
        phasestr = "\n -"+"\n -".join([str(k)+": "+str(v) for k,v in phases.items()])
        jointphasestr = "\n -"+str(self.num_joint)+" combinations, making: "+str(len(self.jointmodes))+' total'
        return "SampleApproach for "+self.mdltype+" model with "+str(len(self._fxnmodes))+" modes: "+modestr+"\n"\
            +str(self.num_joint)+" joint modes ("+str(len(self.jointmodes))+" combinations), \nin "+str(len(phases))+" phases: "+phasestr+\
                " \nsampled at "+str(len(self.times))+" times: \n -"+str(self.times)+"\nresulting in "+str(len(self.scenlist))+" total fault scenarios."
    def init_modelist(self,mdl, faults, jointfaults={'faults':'None'}):
        """Initializes comprates, jointmodes internal list of modes"""
        self.comprates={}
        self._fxnmodes={}
        if faults=='all':
            self.fxnrates=dict.fromkeys(self.fxns)
            for fxnname, fxn in  self.fxns.items():
                for mode, params in fxn.m.faultmodes.items():
                    if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(fxn.m.faultmodes),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                    else:               self._fxnmodes[fxnname, mode] = params
                self.fxnrates[fxnname]=fxn.m.failrate
                if hasattr(fxn, 'c'):
                    self.comprates[fxnname] = {compname:comp.m.failrate for compname, comp in fxn.c.components.items()}
                else:
                    self.comprates[fxnname] = {}
        elif faults=='single-component' or faults[0]=='single-component':
            if type(faults)==tuple: 
                if faults[1]=='all':        fxns_to_sample = self.fxns
                elif type(faults[1])==str:  fxns_to_sample = [faults[1]]
                else:                       fxns_to_sample=faults[1]
            else:                           fxns_to_sample = self.fxns
            self.fxnrates=dict.fromkeys(fxns_to_sample)
            for fxnname in fxns_to_sample:
                fxn = self.fxns[fxnname]
                if getattr(fxn, 'c', {}):
                    firstcomp = list(fxn.c.components)[0]
                    for mode, params in fxn.m.faultmodes.items():
                        comp = fxn.c.faultmodes.get(mode, 'fxn')
                        if comp==firstcomp or comp=='fxn':
                            if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(fxn.m.faultmodes),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                            else:               self._fxnmodes[fxnname, mode] = params
                    self.fxnrates[fxnname]=fxn.m.failrate
                    self.comprates[fxnname] = {firstcomp: sum([comp.m.failrate for compname, comp in fxn.c.components.items()])}
                else:
                    for mode, params in fxn.m.faultmodes.items():
                        if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(fxn.m.faultmodes),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                        else:               self._fxnmodes[fxnname, mode] = params
                    self.fxnrates[fxnname]=fxn.m.failrate
                    self.comprates[fxnname] = {}
        elif faults=='single-function':
            fxnclasses = mdl.fxnclasses();
            fxns_for_class = {f:mdl.fxns_of_class(f) for f in fxnclasses} 
            fxns_to_use = {list(fxns)[0]: len(fxns) for f, fxns in fxns_for_class.items()}
            self.fxnrates=dict.fromkeys(fxns_to_use)
            for fxnname in fxns_to_use:
                fxn = self.fxns[fxnname]
                for mode, params in fxn.m.faultmodes.items():
                    if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(fxn.m.faultmodes),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                    else:               self._fxnmodes[fxnname, mode] = params
                self.fxnrates[fxnname]=fxn.m.failrate * fxns_to_use[fxnname]
                if hasattr(fxn, 'c'):
                    self.comprates[fxnname] = {compname:comp.m.failrate for compname, comp in fxn.c.components.items()}
                else:
                    self.comprates[fxnname] = {}
        else:
            if type(faults)==str:   faults = [(faults, mode) for mode in self.fxns[faults].m.faultmodes] #single-function modes
            elif type(faults)==tuple:
                if faults[0]=='mode name':          faults = [(fxnname, mode) for fxnname,fxn in self.fxns.items() for mode in fxn.m.faultmodes if mode==faults[1]]  
                elif faults[0]=='mode names':       faults = [(fxnname, mode) for f in faults[1] for fxnname,fxn in self.fxns.items() for mode in fxn.m.faultmodes if mode==f]  
                elif faults[0]=='mode type':        
                    faults = [(fxnname, mode) for fxnname,fxn in self.fxns.items() for mode in fxn.m.faultmodes if (faults[1] in mode and (len(faults)<3 or not faults[2] in mode))]
                elif faults[0]=='mode types':       
                    if type(faults[1])==str:    secondarg=(faults[1],)
                    else:                       secondarg=faults[1]
                    faults = [(fxnname, mode) for fxnname,fxn in self.fxns.items() for mode in fxn.m.faultmodes if any([f in mode for f in secondarg])]
                elif faults[0]=='function class':   faults = [(fxnname, mode) for fxnname,fxn in mdl.fxns_of_class(faults[1]).items() for mode in fxn.m.faultmodes]
                elif faults[0]=='function classes': faults = [(fxnname, mode) for f in faults[1] for fxnname,fxn in mdl.fxns_of_class(f).items() for mode in fxn.m.faultmodes]
                else: raise Exception("Invalid option in tuple argument: "+str(faults[0]))
            elif type(faults)==list: 
                if type(faults[0])!=tuple: raise Exception("Invalid list option: "+str(faults)+" , provide list of tuples") 
                faults=faults
            else: raise Exception("Invalid option for faults: "+str(faults)) 
            self.fxnrates=dict.fromkeys([fxnname for (fxnname, mode) in faults])
            for fxnname, mode in faults: 
                params = self.fxns[fxnname].m.faultmodes[mode]
                if params=='synth': self._fxnmodes[fxnname, mode] = {'dist':1/len(faults),'oppvect':[1], 'rcost':0,'probtype':'prob','units':'hrs'}
                else:               self._fxnmodes[fxnname, mode] = params
                self.fxnrates[fxnname]=self.fxns[fxnname].m.failrate
                if hasattr(self.fxns[fxnname], 'c'):
                    self.comprates[fxnname] = {compname:comp.m.failrate for compname, comp in self.fxns[fxnname].c.components.items()}
                else:
                    self.comprates[fxnname] = {}
        if type(jointfaults['faults'])==int or jointfaults['faults']=='all':
            if jointfaults['faults']=='all': 
                if not jointfaults.get('jointfuncs', False): num_joint = len({i[0] for i in self._fxnmodes})
                else:                                        num_joint= len(self._fxnmodes)
            else:                                            num_joint=jointfaults['faults']
            self.jointmodes=[]; self.num_joint=num_joint
            inclusive = jointfaults.get('inclusive', True)
            if inclusive:
                for numjoint in range(2, num_joint+1):
                    jointmodes = list(itertools.combinations(self._fxnmodes, numjoint))
                    if not jointfaults.get('jointfuncs', False): 
                        jointmodes = [jm for jm in jointmodes if not any([jm[i-1][0] ==j[0] for i in range(1, len(jm)) for j in jm[i:]])]
                    self.jointmodes = self.jointmodes + jointmodes
            elif not inclusive:
                jointmodes = list(itertools.combinations(self._fxnmodes, num_joint))
                if not jointfaults.get('jointfuncs', False): 
                    jointmodes = [jm for jm in jointmodes if not any([jm[i-1][0] ==j[0] for i in range(1, len(jm)) for j in jm[i:]])]
                self.jointmodes=jointmodes
            else: raise Exception("Invalid option for jointfault['inclusive']")
        elif type(jointfaults['faults'])==list: self.jointmodes = jointfaults['faults']; self.num_joint='Custom'
        elif jointfaults['faults']!='None': raise Exception("Invalid jointfaults argument type: "+str(type(jointfaults['faults'])))
        else: self.jointmodes=[]; self.num_joint='None'
    def calc_intervaltime(self,times, tstep):
        return float(times[1]-times[0])+tstep
    def init_rates(self,mdl, jointfaults={'faults':'None'}, modephases={}, join_modephases=False):
        """ Initializes rates, rates_timeless"""
        self.rates = dict.fromkeys(self._fxnmodes)
        self.rates_timeless = dict.fromkeys(self._fxnmodes)
        self.mode_phase_map = dict.fromkeys(self._fxnmodes)

        for (fxnname, mode) in self._fxnmodes:
            try:
                self.init_rate(mdl, fxnname, mode,
                               modephases=modephases, join_modephases=join_modephases)
            except Exception as e:
                raise Exception("Error in fxnname, mode: " +
                                fxnname + "," + mode) from e
        if getattr(self, 'jointmodes', False):
            self.init_jointmode_rates(jointfaults)

    def init_rate(self, mdl, fxnname, mode, modephases={}, join_modephases=False):
        self.rates[fxnname, mode] = dict()
        self.rates_timeless[fxnname, mode] = dict()
        self.mode_phase_map[fxnname, mode] = dict()
        overallrate = self.fxnrates[fxnname]
        dist = self._fxnmodes[fxnname, mode]['dist']
        mode_oppvect = self._fxnmodes[fxnname, mode]['oppvect']
        if self.comprates[fxnname] and mode in self.fxns[fxnname].c.faultmodes:
            compname = self.fxns[fxnname].c.faultmodes[mode]
            overallrate = self.comprates[fxnname][compname]

        key_phases = self.fxns[fxnname].m.key_phases_by
        if key_phases == 'self':
            key_phases = fxnname
        if modephases and type(mode_oppvect) == list and key_phases != 'global':
            if len(mode_oppvect) < 1:
                mode_oppvect = {}
            elif len(mode_oppvect) == 1:
                mode_oppvect = {phase: 1 for phase in modephases[key_phases]}
            else:
                raise Exception("Poorly specified oppvect for fxn: " + fxnname +
                                " mode: " + mode +
                                "--provide a dict to use with modephases")

        if modephases and join_modephases and (key_phases not in ['global', 'none']):
            oppvect = {**{phase: 0 for phase in modephases[fxnname]},
                       **mode_oppvect}
            fxnphases = {m: [self.phases[fxnname][ph] for ph in m_phs]
                         for m, m_phs in modephases[fxnname].items()}
        else:
            if key_phases == 'global':
                fxnphases = self.globalphases
            elif key_phases == 'none':
                fxnphases = {'operating': [mdl.sp.times[0], mdl.sp.times[-1]]}
            else:
                fxnphases = self.phases.get(key_phases, self.globalphases)
            fxnphases = dict(sorted(fxnphases.items(),
                                    key=lambda item: item[1][0]))
            if modephases and (key_phases not in ['global', 'none']):
                modevect = mode_oppvect
                oppvect = {phase: 0 for phase in fxnphases}
                oppvect.update({phase: modevect.get(mode, 0)/len(phases)
                                for mode, phases in modephases[key_phases].items()
                                for phase in phases})
            else:
                oppvect = {phase: 0 for phase in fxnphases}
                if type(mode_oppvect) == dict:
                    oppvect.update(mode_oppvect)
                else:
                    opplist = mode_oppvect
                    if len(opplist) > 1:
                        oppvect.update({phase: opplist[i]
                                        for (i, phase) in enumerate(fxnphases)})
                    else:
                        oppvect.update({phase: opplist[0]
                                        for (i, phase) in enumerate(fxnphases)})
        for phase, times in fxnphases.items():
            opp = oppvect[phase]/(sum(oppvect.values())+1e-100)
            
            if self._fxnmodes[fxnname, mode]['probtype'] == 'prob':
                dt = self.tstep
                unitfactor = 1
            elif type(times[0]) == list:
                dt = sum([self.calc_intervaltime(ts, self.tstep) for ts in times])
                unitfactor = self.unit_factors[self.units]/self.unit_factors[self._fxnmodes[fxnname, mode]['units']]
            elif self._fxnmodes[fxnname, mode]['probtype']=='rate' and len(times)>1:      
                dt = self.calc_intervaltime(times, self.tstep)
                unitfactor = self.unit_factors[self.units]/self.unit_factors[self._fxnmodes[fxnname, mode]['units']]
                times=[times]
            elif self._fxnmodes[fxnname, mode]['probtype']=='rate':  
                dt = self.tstep
                unitfactor = self.unit_factors[self.units]/self.unit_factors[self._fxnmodes[fxnname, mode]['units']]
            self.rates[fxnname, mode][key_phases, phase] = overallrate*opp*dist*dt*unitfactor #TODO: update with units
            self.rates_timeless[fxnname, mode][key_phases, phase] = overallrate*opp*dist
            self.mode_phase_map[fxnname, mode][key_phases, phase] = times

    def init_jointmode_rates(self, jointfaults):
        for (j_ind, jointmode) in enumerate(self.jointmodes):
            self.init_jointmode_rate(jointfaults, j_ind, jointmode)

        if not jointfaults.get('inclusive', True):
            for (fxnname, mode) in self._fxnmodes:
                self.rates.pop((fxnname, mode))
                self.rates_timeless.pop((fxnname, mode))
                self.mode_phase_map.pop((fxnname, mode))

    def init_jointmode_rate(self,jointfaults, j_ind, jointmode):
        self.rates.update({jointmode: dict()})
        self.rates_timeless.update({jointmode: dict()})
        self.mode_phase_map.update({jointmode: dict()})
        jointphase_list = [self.mode_phase_map[mode] for mode in jointmode]
        jointphase_dict = {k: v for mode in jointmode
                           for k, v in self.mode_phase_map[mode].items()}
        phasecombos = [i for i in itertools.product(*jointphase_list)]
        if 'limit jointphases' in jointfaults and jointfaults['limit jointphases']<len(phasecombos): 
            rng = np.random.default_rng()
            pc_inds = [i for i in range(len(phasecombos))]
            pc_choices = rng.choice(pc_inds,
                                    jointfaults['limit jointphases'], replace=False)
            phasecombos = [phasecombos[i] for i in pc_choices]
        for phase_combo in phasecombos:
            intervals = [jointphase_dict[phase] for phase in phase_combo]
            overlap, intervals_times = find_overlap_n(intervals)
            if overlap:
                phaseid = tuple(set(phase_combo))
                if len(phaseid) == 1:
                    phaseid = phaseid[0]
                    rates = [self.rates[fmode][phaseid] for fmode in jointmode]
                else:
                    rates = [self.rates[fmode][phase_combo[i]] *
                             len(overlap)/intervals_times[i]
                             for i, fmode in enumerate(jointmode)]
                # if no input, assume independence
                if not jointfaults.get('pcond', False):
                    prob = np.prod(1-np.exp(-np.array(rates)))
                    self.rates[jointmode][phaseid] = -np.log(1.0-prob)
                elif type(jointfaults['pcond']) in [float, int]:
                    self.rates[jointmode][phaseid] = jointfaults['pcond']*max(rates)
                elif type(jointfaults['pcond']) == list:
                    self.rates[jointmode][phaseid] = jointfaults['pcond'][j_ind]*max(rates)
                else:
                    raise Exception("Invalid pcond argument in jointfaults: " +
                                    str(jointfaults['pcond']))
                if len(overlap) > 1:
                    self.rates_timeless[jointmode][phaseid] = self.rates[jointmode][phaseid]/(len(overlap)*self.tstep)
                else:
                    self.rates_timeless[jointmode][phaseid] = self.rates[jointmode][phaseid]
                self.mode_phase_map[jointmode][phaseid] = overlap

    def create_sampletimes(self,mdl, params={}, default={'samp':'evenspacing','numpts':1}):
        """ Initializes weights and sampletimes """
        self.sampletimes={}
        self.weights={fxnmode:dict.fromkeys(rate) for fxnmode,rate in self.rates.items()}
        self.sampparams={}
        for fxnmode, ratedict in self.rates.items():
            for phaseid, rate in ratedict.items():
                if rate > 0.0:
                    times = self.mode_phase_map[fxnmode][phaseid]
                    if phaseid in params:       param = params.get(phaseid, default)
                    elif fxnmode in params:     param = params.get(fxnmode, default)
                    elif fxnmode[0] in params:  param = params.get(fxnmode[0], default)
                    elif fxnmode[1] in params:  param = params.get(fxnmode[1], default)
                    else:                       param = params.get((fxnmode,phaseid), default)
                    self.sampparams[fxnmode, phaseid] = param
                    if type(times[0])!=list: times=[times]
                    possible_phasetimes=[]
                    for ts in times: 
                        if len(ts)==1:      possible_phasetimes = ts
                        elif len(ts)<2:     possible_phasetimes= ts
                        else:               possible_phasetimes = possible_phasetimes + list(np.arange(ts[0], ts[-1]+self.tstep, self.tstep))
                    possible_phasetimes=list(set([np.round(t,4) for t in possible_phasetimes]))
                    possible_phasetimes.sort()
                    if len(possible_phasetimes)<=1: 
                        a=1
                        self.add_phasetimes(fxnmode, phaseid, possible_phasetimes)
                    else:
                        if param['samp']=='likeliest':
                            weights=[]
                            if self.rates[fxnmode][phaseid] == max(list(self.rates[fxnmode].values())):
                                phasetimes = [round(np.quantile(possible_phasetimes, 0.5)/self.tstep)*self.tstep]
                            else: phasetimes = []
                        else: 
                            pts, weights = self.select_points(param, [pt for pt, t in enumerate(possible_phasetimes)])
                            phasetimes = [possible_phasetimes[pt] for pt in pts]
                        self.add_phasetimes(fxnmode, phaseid, phasetimes, weights=weights)
    def select_points(self, param, possible_pts):
        """
        Selects points in the list possible_points according to a given sample strategy.

        Parameters
        ----------
        param : dict
            Sample parameter. Has structure:
                - 'samp' : str ('quad', 'fullint', 'evenspacing','randtimes','symrandtimes')
                    sample strategy to use (quadrature, full integral, even spacing, random times, or symmetric random times)
                - 'numpts' : float
                    number of points to use (for evenspacing, randtimes, and symrandtimes only)
                - 'quad' : dict
                    dict with structure {'nodes'[nodelist], 'weights':weightlist}
                    where the nodes in the nodelist range between -1 and 1
                    and the weights in the weightlist sum to 2.
        possible_pts : 
            list of possible points in time.

        Returns
        -------
        pts : list
            selected points
        weights : list
            weights for each point
        """
        weights=[]
        if param['samp']=='fullint': pts = possible_pts
        elif param['samp']=='evenspacing':
            if param['numpts']+2 > len(possible_pts): pts = possible_pts
            else: pts= [int(round(np.quantile(possible_pts, p/(param['numpts']+1)))) for p in range(param['numpts']+2)][1:-1]
        elif param['samp']=='quadrature':
            quantiles = np.array(param['quad']['nodes'])/2 +0.5
            if len(quantiles) > len(possible_pts): pts = possible_pts
            else: 
                pts= [int(round(np.quantile(possible_pts, q))) for q in quantiles]
                weights=np.array(param['quad']['weights'])/sum(param['quad']['weights'])
        elif param['samp']=='randtimes':
            if param['numpts']>=len(possible_pts): pts = possible_pts
            else: pts= [possible_pts.pop(np.random.randint(len(possible_pts))) for i in range(min(param['numpts'], len(possible_pts)))]
        elif param['samp']=='symrandtimes':
            if param['numpts']>=len(possible_pts): pts = possible_pts
            else: 
                if len(possible_pts) %2 >0:  pts = [possible_pts.pop(int(np.floor(len(possible_pts)/2)))]
                else: pts = [] 
                possible_pts_halved = np.reshape(possible_pts, (2,int(len(possible_pts)/2)))
                possible_pts_halved[1] = np.flip(possible_pts_halved[1])
                possible_inds = [i for i in range(int(len(possible_pts)/2))]
                inds = [possible_inds.pop(np.random.randint(len(possible_inds))) for i in range(min(int(np.floor(param['numpts']/2)), len(possible_inds)))]
                pts= pts+ [possible_pts_halved[half][ind] for half in range(2) for ind in inds ]
                pts.sort()
        else: print("invalid option: ", param)
        if not any(weights): weights = [1/len(pts) for t in pts]
        if len(pts)!=len(set(pts)):
            raise Exception("Too many pts for quadrature at this discretization")
        return pts, weights
    def add_phasetimes(self, fxnmode, phaseid, phasetimes, weights=[]):
        """ Adds a set of times for a given mode to sampletimes"""
        if phasetimes:
            if not self.weights[fxnmode].get(phaseid): self.weights[fxnmode][phaseid] = {t: 1/len(phasetimes) for t in phasetimes}
            for (ind, time) in enumerate(phasetimes):
                if not self.sampletimes.get(phaseid): 
                    self.sampletimes[phaseid] = {time:[]}
                if self.sampletimes[phaseid].get(time): self.sampletimes[phaseid][time] = self.sampletimes[phaseid][time] + [(fxnmode)]
                else: self.sampletimes[phaseid][time] = [(fxnmode)]
                if any(weights): self.weights[fxnmode][phaseid][time] = weights[ind]
                else:       self.weights[fxnmode][phaseid][time] = 1/len(phasetimes)
    def create_scenarios(self):
        """ Creates list of scenarios to be iterated over in fault injection. Added as scenlist and scenids """
        self.scenlist=[]
        self.times = []
        self.scenids = {}
        for phaseid, samples in self.sampletimes.items():
            if samples:
                for time, faultlist in samples.items():
                    self.times+=[time]
                    for fxnmode in faultlist:
                        if self.sampparams[fxnmode, phaseid]['samp']=='maxlike':    
                            rate = sum(self.rates[fxnmode].values())
                        else: 
                            rate = self.rates[fxnmode][phaseid] * self.weights[fxnmode][phaseid][time]
                        if type(fxnmode[0])==str:
                            name = fxnmode[0]+'_'+fxnmode[1]+'_'+t_key(time)
                            scen = SingleFaultScenario(sequence = {time:Injection(faults={fxnmode[0]:[fxnmode[1]]})},
                                            function = fxnmode[0],
                                            fault = fxnmode[1],
                                            rate = rate,
                                            time = time,
                                            name=name)
                        else:
                            name = ' '.join([fm[0]+'_'+fm[1]+'_' for fm in fxnmode])+t_key(time)
                            faults = dict.fromkeys([fm[0] for fm in fxnmode])
                            for fault in faults:
                                faults[fault] = [fm[1] for fm in fxnmode if fm[0]==fault]
                            scen = JointFaultScenario(sequence = {time:Injection(faults=faults)},
                                                      joint_faults=len(fxnmode),
                                                      functions = tuple(fm[0] for fm in fxnmode),
                                                      modes = tuple(fm[1] for fm in fxnmode),
                                                      rate=rate, time=time, name=name)
                        self.scenlist=self.scenlist+[scen]
                        if self.scenids.get((fxnmode, phaseid)): self.scenids[fxnmode, phaseid] = self.scenids[fxnmode, phaseid] + [name]
                        else: self.scenids[fxnmode, phaseid] = [name]
        self.times = list(set(self.times))
        self.times.sort()
    def reduce_scens_to_samp(self, samp_size=100,seed=None):
        """Reduces the number of scenarios (in the scenlist) to a given sample size samp_size. Useful for
        choosing a random subset of an approach which would otherwise have a large number of scenarios.
        Note that many structures may not be preserved and some artefacts may be present."""
        if samp_size<len(self.scenlist):
            rng = np.random.default_rng(seed)
            self.scenlist = rng.choice(self.scenlist, samp_size, replace=False)

    def list_modes(self, joint=False):
        """ Returns a list of modes in the approach """
        if joint and hasattr(self, 'jointmodes'):
            return [(fxn, mode) for fxn, mode in self._fxnmodes.keys()] + self.jointmodes
        else:
            return [(fxn, mode) for fxn, mode in self._fxnmodes.keys()]
    def list_moderates(self):
        """ Returns the rates for each mode """
        return {(fxn, mode): sum(self.rates[fxn,mode].values()) for (fxn, mode) in self.rates.keys()}

    def get_scenid_groups(self,group_by='phases', group_dict={}):
        """
        Returns a dict with different scenario ids grouped according to group_by.


        group_by: str, with options:
        - 'none': Returns {'scenid':'scenid'} for all scenarios
        - 'phase': Returns {(fxnmode, fxnphase): {scenids}}--identical scenarios within
        given phase are grouped 
        - 'fxnfault': Returns {fxnmode: {scenids}} All identical scenarios (fxn, mode)
        are grouped
        - 'mode': Returns {mode:{scenids}}. All scenarios with the same mode name are
        grouped
        - 'mode type': Returns {modetype:scenids}. All scenarios with the same mode type
        (mode types must be given to the sampleapproach) are grouped
        - 'functions': Returns {function:scenids}. All scenarios and modes from a given
        function are grouped.
        - 'times': Returns {time:scenids}. All scenarios at a given time are grouped.
        - 'fxnclassfault': Returns {(fxnclass, mode):scenids}. All scenarios
        (fxnclass, mode) from a given function class are grouped.
        - 'fxnclass': Returns {fxnclass:scendis}. All scenarios from a given function
        class are grouped.

        For 'fxnclass', 'fxnclassfault', and 'modetype', a group_dict dictionary must be
        provided that groups the function/mode classes/types.
        -------------------
        Returns:
        - grouped_scens: dict
              A dictionary of the scenario ids associated with the given group
              Has structure: {group: scenids}
        """
        if group_by in ['fxnclass', 'fxnclassfault', 'modetype'] and not group_dict:
            raise Exception("group_dict must be provided to group by these")
        if group_by == 'none':
            grouped_scens = {s: [s] for v in self.scenids.values() for s in v}
        elif group_by == 'phase':
            grouped_scens = self.scenids
        elif group_by == 'fxnfault':
            grouped_scens = {m: set() for m in self.list_modes(True)}
            for modephase, ids in self.scenids.items():
                grouped_scens[modephase[0]].update(ids)
        elif group_by == 'mode':
            grouped_scens = {m[1]: set() for m in self.list_modes(True)}
            for modephase, ids in self.scenids.items():
                grouped_scens[modephase[0][1]].update(ids)
        elif group_by == 'functions':
            grouped_scens = {m[0]: set() for m in self.list_modes(True)}
            for modephase, ids in self.scenids.items():
                grouped_scens[modephase[0][0]].update(ids)
        elif group_by == 'times':
            grouped_scens = {float(t): set() for t in set(self.times)}
            for scen in self.scenlist:
                time = float(scen.time)
                grouped_scens[time].add(scen.name)
        elif group_by == 'fxnclass':
            fxn_groups = {sub_v: k for k, v in group_dict.items() for sub_v in v}
            grouped_scens = {fxn_groups[fxnmode[0]]: set()
                             for fxnmode in self.list_modes(True)}
            grouped_scens['nominal'] = {'nominal'}
            for modephase, ids in self.scenids.items():
                fxn = modephase[0][0]
                group = fxn_groups[fxn]
                grouped_scens[group].update(ids)
        elif group_by == 'fxnclassfault':
            fxn_groups = {sub_v: k for k, v in group_dict.items() for sub_v in v}
            grouped_scens = {(fxn_groups[fxnmode[0]], fxnmode[1]): set()
                             for fxnmode in self.list_modes(True)}
            grouped_scens['nominal'] = {'nominal'}
            for modephase, ids in self.scenids.items():
                fxn, mode = modephase[0]
                group = fxn_groups[fxn]
                grouped_scens[group, mode].update(ids)
        elif group_by == 'modetype':
            grouped_scens = {group: set() for group in group_dict}
            grouped_scens['ungrouped'] = set()
            for modephase, ids in self.scenids.items():
                mode = modephase[0][1]
                grouped = False
                for group in grouped_scens:
                    if group in mode:
                        grouped_scens[group].update(ids)
                        grouped = True
                        break
                if not grouped:
                    grouped_scens['ungrouped'].update(ids)
        else:
            raise Exception("Invalid option for group_by: "+group_by)
        return grouped_scens

    def get_id_weights(self):
        """
        Return a dictionary with weights for each scenario.

        Dict has structure.{scenid: weight}.
        """
        id_weights = {}
        for scens, ids in self.scenids.items():
            num_phases = len([n for n, i in self.weights[scens[0]].items() if i])
            weights = np.array([*self.weights[scens[0]][scens[1]].values()])/num_phases
            id_weights.update({scenid: weights[i] for i, scenid in enumerate(ids)})
        return id_weights

def find_overlap_n(intervals):
    """
    Find the overlap between given intervals.

    Used to sample joint fault modes with different (potentially overlapping) phases
    """
    try:
        joined_times = {}
        intervals_times = []
        for i, interval in enumerate(intervals):
            if type(interval[0]) in [float, int]:
                interval = [interval]
            possible_times = set()
            possible_times.update(*[{*np.arange(i[0], i[-1]+1)} for i in interval])
            if i == 0:
                joined_times = possible_times
            else:
                joined_times = joined_times.intersection(possible_times)
            intervals_times.append(len(possible_times))
        if not joined_times:
            return [], intervals_times
        else:
            return [*np.sort([*joined_times])], intervals_times
    except IndexError:
        if all(intervals[0] == i for i in intervals):
            return intervals[0]
        else:
            return 0

if __name__ == "__main__":

    import doctest
    doctest.testmod(verbose=True)
