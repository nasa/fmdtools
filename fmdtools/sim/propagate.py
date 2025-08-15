# -*- coding: utf-8 -*-
"""
Functions to propagate faults through a user-defined fault model.

Main Methods:

- :func:`nominal()`: Runs the model over time in the nominal scenario.
- :func:`one_fault()`:  Runs one fault in the model at a specified time.
- :func:`sequence()`: Runs arbitrary scenario of fault modes at specified times.
- :func:`single_faults()`: Creates and propagates a list of failure scenarios in a model
  over given model times.
- :func:`fault_sample`: Injects and propagates faults defined by a FaultSample.
- :func:`parameter_sample`: Simulates a model over a range of parameters defined by a
  ParameterSample
- :func:`nested_sample`: Injects and propagates faults in the model defined by a
  given SampleApproach over a range of parameters defined by a ParameterSample.

Shared Method Parameters:

- :data:`sim_kwargs`: Simulation keyword arguments.
- :data:`run_kwargs`: Run keyword arguments.
- :data:`mult_kwargs`: Multi-scenario keyword arguments

Private Methods:

- :func:`list_init_faults()`: Creates a list of single-fault scenarios for the graph,
  given the modes set up in the fault model
- :func:`prop_one_scen()`: Runs a fault scenario in the model over time
- :func:`save_helper()`: Helper function for inline results saving.
- :func:`unpack _res_list()`: Helper function for unpacking results
- :func:`exec_nom_par`: Helper function for executing nominal scenarios in parallel
- :func:`exec_nom_helper`: Helper function for executing nominal scenarios
- :func:`nom_helper`: Helper function for initial run of nominal scenario
- :func:`scenlist_helper`: Helper function for `approach`
- :func:`exec_scen_par`:  Helper function for executing the scenario in parallel
- :func:`exec_scen`: Executes a scenario and generates results and classifications given
  a model and nominal model history
- :func:`check_hist_memory`: Checks if the memory will be exhausted given the size of
  the mdlhist and number of scenarios
- :func:`check_mdl_memory`: Raises exception if model size is too large.
- :func:`check_overwrite`: Checks if file can be overwritten
- :func:`check_end_condition`: Helper function for `prop_one_scen` to end simulation
  earlier.
- :func:`get_result`: Helper function for `prop_one_scen` to get result at specific
  timestep.
- :func:`get_endclass_vars`: Helper function for `get_result`

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

from fmdtools.define.base import t_key, filter_kwargs, get_dict_repr
from fmdtools.define.block.base import Simulable, Block
from fmdtools.define.container.base import BaseContainer
from fmdtools.sim.sample import SampleApproach, FaultDomain, FaultSample
from fmdtools.sim.scenario import Injection, Sequence, Scenario, SingleFaultScenario
from fmdtools.analyze.common import create_indiv_filename, file_check
from fmdtools.analyze.result import Result, clean_to_return
from fmdtools.analyze.history import History
from fmdtools.analyze.phases import from_hist


import numpy as np
import copy
import tqdm
import os
import warnings

# DEFAULT ARGUMENTS
sim_kwargs = {'desired_result': 'endclass',
              'staged': False,
              'cut_hist': True,
              'use_end_condition': True,
              'warn_faults': True}
"""
Simulation keyword arguments.

Parameters
----------
desired_result : dict/str/list
    Desired quantities to return in the first argument.
    Options are:

    - 'endclass': a dict returned by classify (default)
    - 'endfaults': a dict of returned fault modes and their propagation, e.g., ::

        {'endfaults':faultdict, 'faultprops':faultpropdict}

    - 'graph'/'flowgraph'/etc: a networkx graph of the model with fault modes
      superimposed
    - 'fxnname.varname': variable values to get
    - a list of the above arguments (for multiple at the end)
    - a dict of lists (for multiple over time), e.g., ::

        {time:[varnames,... 'endclass']}

    The default is 'all'.

staged : bool, optional
    Whether to inject the faults in a copy of the nominal model at the fault time
    (True) or instantiate a new model for the fault (False). Setting to True
    roughly halves execution time. The default is False.
cut_hist : bool
    Whether to cut the history.
use_end_condition : bool
    Whether to use the end-condition
warn_faults : bool
    Whether to produce a warning when faults occur in a nominal sim.
"""


def unpack_sim_kwargs(**kwargs):
    """Unpack :data:`sim_kwargs` parameters for :func:`prop_one_scen`."""
    return (kwargs.get(k, v) for k, v in sim_kwargs.items())


def pack_sim_kwargs(**kwargs):
    """Create :data:`sim_kwargs` for :func:`prop_one_scen`."""
    return {k: kwargs.get(k, v) for k, v in sim_kwargs.items()}


run_kwargs = {'save_args': {},
              'mdl_kwargs': {},
              'protect': True}
"""
Run keyword arguments.

Parameters
----------
protect : bool
    Whether or not to protect the model object via copying.
    Options:

    - True (default): re-instances the model so that multiple simulations can
      be run successively without causing problems
    - False : Thus, the model object that is returned can be modified and
      analyzed if needed

save_args : dict (optional)
    Dictionary specifying if/how to save results. Default is {}, which doesn't
    save anything.
    Has structure ::
        {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':indiv}

    where mdlhistargs and endclassargs are dictionaries of save arguments, e.g.,::

    {'filename':'filename.npz', 'filetype':'npz', 'overwrite':True}

    and indiv is an (optional) bool specifying whether to save results individually
    (in a folder) or as a monolythic file.

mdl_kwargs: dict (optional)
    Parameter dictionary to be instantiated in the model prior to simulation.
    Has structure ::
        {"p": Parameter, "sp":SimParam, "track":track}

    Parameter dictionaries do not need to be complete (if incomplete).
"""


def pack_run_kwargs(**kwargs):
    """Create subset of run kwargs for :func:`nom_helper` and :data:`run_kwarg`."""
    return {k: copy.deepcopy(kwargs.get(k, v)) for k, v in run_kwargs.items()}


mult_kwargs = {'max_mem': 2e9,
               'showprogress': True,
               'pool': False,
               'close_pool': True}


"""
Multi-scenario keyword arguments.

Parameters
----------
    pool : process pool, optional
        Process Pool Object from multiprocessing or pathos packages.
        e.g. parallelpool = mp.pool(n) for n cores (multiprocessing)
        or parallelpool = ProcessPool(nodes=n) for n cores (pathos)
        If False, the set of scenarios is run serially. The default is False
    showprogress: bool, optional
        whether to show a progress bar during execution. default is true
    max_mem : int
        Max memory (warns the user when memory is above threshold)
"""


def pack_mult_kwargs(**kwargs):
    """Create subset of mult kwargs."""
    return {k: kwargs.get(k, v) for k, v in mult_kwargs.items()}


def unpack_mult_kwargs(kwargs):
    """Unpack the mult kwarg parameters for the :func:`parameter_sample`."""
    return (kwargs.pop(k, v) for k, v in mult_kwargs.items())

# FAULT PROPAGATION

def get_sim_call_kwargs(sim, **kwargs):
    return {**filter_kwargs(sim, **kwargs),
            **filter_kwargs(SimEvent.run, **kwargs)}


def nominal(mdl, **kwargs):
    """
    Run the model over time in the nominal scenario.

    Parameters
    ----------
    mdl : Simulable
        Model of the system
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others

    Returns
    -------
    result: Result
        dict of result corresponding to desired_result, e.g. ::

            Result({'endclass': endclasses,
                    'endfaults': endfaults,
                    'varname': var,
                    t: {'endclass': endclasses...} ...})

    nomhist : History
        A History dict with a history of modelstates
    """
    sim = Simulation(mdl=mdl, **filter_kwargs(Simulation, **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


def save_helper(save_args, endclass, mdlhist, indiv_id='', result_id=''):
    """
    Save results (helper function).

    Parameters
    ----------
    save_args : dict
        Dict with structure ::

            {'mdlhists': mdlhistargs,
             'endclass': endclassargs,
             'indiv': individual_saving}

        where mdlhistargs and endclassargs are dictionaries of arguments to Result.save
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
        and individual_saving is a bool (True/False)
    endclass : dict
        dict of end-state classifications (from simulation)
    mdlhist : dict
        dict of model histories (from simulation)
    """
    if 'mdlhists' in save_args:
        save_args['mdlhist'] = save_args.pop('mdlhists')
    if 'endclasses' in save_args:
        save_args['endclass'] = save_args.pop('endclasses')
    for save_arg in save_args:
        if save_arg not in {'mdlhist', 'endclass', 'indiv'}:
            raise Exception("Invalid key in save_args: "+save_arg)
    if save_args.get('indiv', False) and indiv_id:
        if 'endclass' in save_args:
            newfilename = create_indiv_filename(save_args['endclass']['filename'],
                                                indiv_id, splitchar="/")
            endclass.save(**{**save_args['endclass'], 'filename': newfilename},
                          result_id=result_id)
        if 'mdlhist' in save_args:
            newfilename = create_indiv_filename(save_args['mdlhist']['filename'],
                                                indiv_id, splitchar="/")
            mdlhist.save(**{**save_args['mdlhist'], 'filename': newfilename},
                         result_id=result_id)
    elif not save_args.get('indiv', False) and not indiv_id:
        if 'mdlhist' in save_args:
            mdlhist.save(**save_args['mdlhist'])
        if 'endclass' in save_args:
            endclass.save(**save_args['endclass'])


def parameter_sample(mdl, ps, **kwargs):
    """
    Simulate a set of nominal scenarios through a model.

    Useful for exploring/understanding the sets of parameters where the system will run
    nominally and/or fail.

    Parameters
    ----------
    mdl : Simulable
        Model to simulate
    ps: ParameterSample
        Parameter Sample defining the nominal scenarios to run the system over.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others
        - :data:`mult_kwargs` : kwargs
              Multi-scenario options

    Returns
    -------
    nomresults: Result
        Result dict of result corresponding to desired result {'scenname': result}
    nomhists : History
        History of model histories, with structure {'scenname': mdlhist}
    """
    sim = MultiSimulation(mdl=mdl, samp=ps, **filter_kwargs(MultiSimulation, **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


def one_fault(mdl, *fxnfault, time=0, f_kw={}, **kwargs):
    """
    Run one fault in the model at a specified time.

    Parameters
    ----------
    mdl : Simulable
        The model to inject the fault in.
    *fxnfault : str
        Has options:
        - 'fxnname', 'faultmode' when a Model is provided, or
        - 'faultmode' when a Block/Function is provided
    time : float, optional
        Time to inject fault. Must be in the range of model times
        (i.e. in range(0, end, mdl.sp.dt)). The default is 0.
    f_kw : dict
        Non-default fault keyword args.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others

    Returns
    -------
    result: Result
        Result dict of result corresponding to desired_result, with structure,::

        Result({'nominal': nomresult, 'faulty': faultyresult})

    mdlhists : History
        A dictionary of the states of the model of each fault scenario over time with
        structure: {'nominal': nomhist, 'faulty': faulthist}
    """
    if len(fxnfault) == 2:
        fxnname, fault = fxnfault
    elif len(fxnfault) == 3:
        fxnname, fault, time = fxnfault
    else:
        fxnname, fault = mdl.name, fxnfault[0]

    scen = SingleFaultScenario.from_fault((fxnname, fault), time, mdl=mdl, **f_kw)
    return sequence(mdl, scen=scen, **kwargs)


def sequence(mdl, seq={}, faultseq={}, disturbances={}, scen={}, rate=np.nan,
             include_nominal=True, **kwargs):
    """
    Run a sequence of faults and disturbances in the model at given times.

    Parameters
    ----------
    mdl : Simulable
        The model to inject the fault in.
    seq : dict
        Scenario dict defining the scenario
        {time:{`faults`:faults, `disturbances`:disturbances}}
    faultseq : dict
        Dict of times and modes defining the fault scenario {time:{fxns: [modes]},}
    disturbances : dict
        Dict of times and states defining the disturbances in the scenario::

        {time: {path.to.state: stateval}}

    scen : Scenario, optional
        Scenario dictionary, if already constructed (for external calls)
    rate : float, optional
        Input rate for the sequence (must be calculated elsewhere)
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is True.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others

    Returns
    -------
    result: Result
        Result dict of result corresponding to desired_result, with structure,::

        Result({'nominal': nomresult, 'faulty': faultyresult})

    mdlhists : dict
        A dictionary of the states of the model of each fault scenario over time with
        structure: {'nominal': nomhist, 'faulty': faulthist}
    """
    if not scen:
        if not seq:
            seq = Sequence(faultseq=faultseq, disturbances=disturbances)
        scen = Scenario(sequence=seq,
                        rate=rate,
                        name='faulty',
                        times=tuple([*seq.keys()]))

    if include_nominal:
        sim = MultiEventSimulation(mdl=mdl, samp=scen,
                                   **filter_kwargs(MultiEventSimulation, **kwargs))
    else:
        sim = Simulation(mdl=mdl, scen=scen, **filter_kwargs(Simulation, **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


def fault_sample(mdl, fs, **kwargs):
    """
    Injects and propagates faults in the model defined by a FaultSample/SampleApproach.

    NOTE: When calling in a script/module using parallel=True, execute using the
    protection statement ::

        if __name__ == 'main':
            results, mdlhists = fault_sample(mdl, fs)

    Otherwise, the method will keep spawning parallel processes.
    See multiprocessing documentation.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in.
    fs : FaultSample/SampleApproach
        FaultSample used to define the list of faults and sample time for the model.
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is True.
    get_phasemap : bool, optional
        Whether to regenerate the FaultSample using new phase information.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others
        - :data:`mult_kwargs` : kwargs
              Multi-scenario options

    Returns
    -------
    results : Result
        A Result dictionary with results desired from each scenario corresponding to
        desired_result over the set of scenarios.
    mdlhists : History
        A History dictionary with the tracked scenario (including the nominal)
    """
    sim = MultiEventSimulation(mdl=mdl, samp=fs,
                               **filter_kwargs(MultiEventSimulation, **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


def fault_sample_from(mdl, faultdomains={}, faultsamples={}, get_phasemap=True,
                      scen={}, **kwargs):
    """
    Create and simulate a fault_sample from the given arguments.

    Use to generate and sample from phases in the same simulation.

    Parameters
    ----------
    mdl : Simulable
        Model to simulate
    faultdomains : dict
        Dict of arguments to SampleApproach.add_faultdomains
    faultsamples : dict
        Dict of arguments to SampleApproach.add_faultsamples
        FaultSamples to add to othe SampleApproach and their arguments.
        Has structure::
        {'fs_name': (*args, **kwargs)}
        where args and kwargs are arguments/kwargs to SampleApproach.add_faultsamples.
    get_phasemap : bool, optional
        Whether to generate the FaultSample from the phasemap. The default is True.
    scen : scenario, optional
        Scenario to use as nominal. The default is {}.
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is False.
    **kwargs : kwargs
        kwargs to simulate over

    Returns
    -------
    res : Result
        A Result dictionary with results desired from each scenario corresponding to
        desired_result over the set of scenarios.
    hist : History
        A History dictionary with the tracked scenario (including the nominal)
    app : SampleApproach
        Generated SampleApproach
    """
    sim = Simulation(mdl=mdl, scen=scen, protect=True, to_return={})
    nomres, nomhist = sim()
    app = gen_sampleapproach(sim.mdl, faultdomains, faultsamples, get_phasemap, nomhist)
    res, hist = fault_sample(sim.mdl, app, **kwargs)
    return res, hist, app


def single_faults(mdl, times=[0.0], include_nominal=True, **kwargs):
    """
    Create and propagates a list of failure scenarios in a model.

    NOTE: When calling in a script/module using parallel=True, execute using the
    protection statement ::

        if __name__ == 'main':
            results, mdlhists = single_faults(mdl)

    Otherwise, the method will keep spawning parallel processes.
    See multiprocessing documentation.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in
    times : list
        List of times to inject the single faults in. Default is [1.0]
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is True.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others
        - :data:`mult_kwargs` : kwargs
              Multi-scenario options

    Returns
    -------
    results: Result
        Result dict of result corresponding to desired_result {scenname:result}
    mdlhists : History
        History dict with the history of all tracked model states for each scenario
        (including the nominal)
    """
    fd = FaultDomain(mdl)
    fd.add_all()
    fs = FaultSample(fd)
    fs.add_fault_times(times)
    sim = MultiEventSimulation(mdl=mdl, samp=fs,
                               **filter_kwargs(MultiEventSimulation, **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


def nested_sample(mdl, ps, get_phasemap=False, faultdomains={}, faultsamples={},
                  include_nominal=True, **kwargs):
    """
    Simulate a set of fault scenarios within a ParameterSample.

    NOTE: When calling in a script/module using parallel=True, execute using the
    protection statement ::

        if __name__ == "main":
            results, mdlhists = nested_sample(mdl, ps)

    Otherwise, the method will keep spawning parallel processes.
    See multiprocessing documentation.

    Parameters
    ----------
    mdl : Simulable
        Model Object to use in the simulation.
    ps : ParameterSample
        Parameter Scenario defining the parameters the model will be run over
    get_phasemap : Bool/List/Dict, optional
        Whether to use nominal simulation phasemap to set up the SampleApproach.
    faultdomains : dict
        Dict of arguments to SampleApproach.add_faultdomains
    faultsamples : dict
        Dict of arguments to SampleApproach.add_faultsamples
        FaultSamples to add to othe SampleApproach and their arguments.
        Has structure::
        {'fs_name': (*args, **kwargs)}
        where args and kwargs are arguments/kwargs to SampleApproach.add_faultsamples.
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is False.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`.
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others.
        - :data:`mult_kwargs` : kwargs
              Multi-scenario options
        - app_args : mdl_kwargs
              Keyword arguments for the SampleApproach.
              See sim.sample.SampleApproach documentation.

    Returns
    -------
    nested_results : Result
        A nested Result dictionary with the desired results of each scenario.
    nested_mdlhists : History
        A nested History dictionary with the history of all model states for each
        scenario
    apps : dict
        A dictionary of the SampleApproaches generated corresponding to each nominal
        scenario with structure {'nomscen1': app1}
    """
    save_args = kwargs.get('save_args', {})
    check_overwrite(save_args)
    max_mem, showprogress, pool, close_p = unpack_mult_kwargs(kwargs)
    sim_kwarg = pack_sim_kwargs(**kwargs)
    run_kwargs_nest = pack_run_kwargs(**kwargs)
    scennames = ps.scen_names()
    nest_hist = History.fromkeys(scennames)
    nest_res = Result.fromkeys(scennames)
    apps = dict.fromkeys(scennames)
    for scenname, scen in tqdm.tqdm(ps.named_scenarios().items(),
                                    disable=not (showprogress),
                                    desc="NESTED SCENARIOS COMPLETE"):
        loc_kwarg = {**sim_kwarg, 'scen': scen, 'close_pool': False,
                     'get_phasemap': get_phasemap, 'showprogress': False,
                     'include_nominal': include_nominal}
        res, hist, app = fault_sample_from(mdl, faultdomains, faultsamples, **loc_kwarg)
        save_helper(save_args, res, hist, indiv_id=scenname, result_id=scenname)
        nest_res[scenname] = res
        nest_hist[scenname] = hist
        apps[scenname] = app
    save_helper(save_args, nest_res, nest_hist)
    close_pool({'pool': pool, 'close_pool': close_p})
    return nest_res.flatten(), nest_hist.flatten(), apps





def close_pool(kwargs):
    """Close pool to avoid memory problems."""
    if kwargs.get('pool', False) and kwargs.get('close_pool', True):
        kwargs['pool'].close()
        kwargs['pool'].terminate()
        kwargs['pool'].join()


def check_overwrite(save_args):
    for arg, args in save_args.items():
        if arg != 'indiv':
            filename = args['filename']
            if args.get('filename', False):
                file_check(filename, args.get('overwrite', False))
            if save_args.get('indiv', False):
                last_split_index = filename.rfind(".")
                foldername = filename[:last_split_index]
                if not os.path.exists(foldername):
                    os.makedirs(foldername)


class SimEvent(BaseContainer):
    """
    Event in a Simulation.

    Parameters
    ----------
    time : float
        Time when the SimEvent is to Occur
    copy : bool
        Whether to copy the model just before the time.
    mdl_copy : Simulable
        Copied model at the simevent (returned by run())
    injection : Injection
        Injection of faults and disturbances to inject at time.
    to_return : dict
        Dict specifying what to return as a Result from the simulation.
    simulated : bool
        Whether the event has simulated. Default is False.
    """

    time: float = 0.0
    copy: bool = False
    mdl_copy: Simulable = None
    injection: Injection = None
    to_return: dict = None
    result: Result = Result()
    simulated: bool = False

    def create_repr(self, fields=['copy', 'injection', 'to_return', 'simulated'],
                    **kwargs):
        """Represent the SimEvent as a string."""
        if not fields:
            fields = self.__fields__
        fields = [f for f in fields if self[f]]
        return super().create_repr(fields=fields, **kwargs)

    def __repr__(self):
        """Show Simulation event as a string."""
        return self.create_repr(fields=[])

    def run(self, mdl, scen={}, nomhist={}, nomresult={}, **kwargs):
        """
        Run the Simulated event.

        Parameters
        ----------
        mdl : Simulable
            Model to simulate the event in.
        **kwargs : kwargs
            Keyword arguments to __call__ and Model.get_result()
        """
        if not self.simulated:
            kwar = {}
            if self.injection:
                kwar.update(self.injection.asdict())
            if self.time == "end":
                kwar["end_of_simulation"] = True
            if self.copy:
                self.mdl_copy = mdl(time=self.time, **kwar, copy=True)
            else:
                mdl(time=self.time, **kwar)
            if self.to_return:
                nomresult = getattr(nomresult, t_key(self.time), {})
                self.result = mdl.get_result(to_return=self.to_return, scen=scen,
                                             nomhist=nomhist, nomresult=nomresult,
                                             **kwargs)
            self.simulated = True
        else:
            raise Exception("Event already simulated.")


class BaseSimulation(BaseContainer):
    name: str = ''
    mdl: Simulable = Block()
    result: Result = Result()
    history: History = History()
    tosave: bool = False
    to_return: dict = {"end": "class"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_return = clean_to_return(self.to_return)

    def save(self, result_filename="result.csv", history_filename="history.csv"):

        if self.result and result_filename:
            self.result.save(result_filename, result_id=self.name)
        if self.history and history_filename:
            self.history.save(history_filename, result_id=self.name)

    def __call__(self, **kwargs):
        self.run(**kwargs)
        if self.tosave:
            self.save(**filter_kwargs(self.save, **kwargs))
        return self.result.flatten(), self.history.flatten()


def exec_sim(args):
    sim = Simulation(**args[0])
    if len(args) > 1:
        return sim(**args[1])
    else:
        return sim()


class MultiSimulation(BaseSimulation):
    samp: object = Scenario()
    save_indiv: bool = False
    pool: object = None
    auto_close_pool: bool = True
    showprogress: bool = True
    max_mem: int = 2e9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.save_indiv is True:
            self.tosave = False
        self.check_hist_memory()
        self.check_mdl_memory()

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        if self.auto_close_pool:
            self.close_pool()
        return self.result.flatten(), self.history.flatten()

    def close_pool(self):
        if self.pool:
            self.pool.close()
            self.pool.terminate()
            self.pool.join()

    def run(self, **kwargs):
        if self.pool:
            runner = self.pool_runner
        else:
            runner = self.std_runner
        scenlist = self.samp.scenarios()
        inputs = self.gen_inputs(scenlist)
        res_list = list(tqdm.tqdm(runner(inputs),
                                  total=len(inputs),
                                  disable=not (self.showprogress),
                                  desc="SCENARIOS COMPLETE"))
        for i, scen in enumerate(scenlist):
            self.result[scen.name] = res_list[i][0]
            self.history[scen.name] = res_list[i][1]

    def gen_default_kwargs(self, **kwargs):
        def_kwar = dict(mdl=self.mdl,
                        to_return=self.to_return,
                        protect=not bool(self.pool))
        def_kwar = {**def_kwar, **kwargs}
        def_kwar = filter_kwargs(Simulation, **def_kwar)
        return def_kwar

    def gen_inputs(self, scenlist, **kwargs):
        def_kwar = self.gen_default_kwargs(**kwargs)
        return [({**def_kwar, 'scen': scen},) for scen in scenlist]

    def pool_runner(self, inputs):
        return self.pool.imap(exec_sim, inputs)

    def std_runner(self, inputs):
        return [exec_sim(i) for i in inputs]

    def check_hist_memory(self):
        """Check if the memory will be exhausted by the hist over the scenarios."""
        mem_total, mem_profile = self.mdl.h.get_memory()
        nscens = len(self.samp.scenarios())
        tot_mem = int(mem_total) * int(nscens)
        if tot_mem > self.max_mem:
            raise Exception("Mdlhist has size: " + str(mem_total)
                            + " bytes. With " + str(nscens)
                            + " scenarios, it is expected that this run will pass the"
                            + " user-defined max_mem=" + str(self.max_mem)
                            + " byte limit by a factor of: "+str(tot_mem/self.max_mem)
                            + ". To avoid, use the track= option to track less info"
                            + " in the mdlhist")

    def check_mdl_memory(self):
        """Check if memory will be exhausted by the model over the scenarios."""
        mem_total, mem_profile = self.mdl.get_memory()
        nscens = len(self.samp.scenarios())
        tot_mem = int(mem_total) * int(nscens)
        if tot_mem > self.max_mem:
            raise Exception("Model has size: " + str(mem_total)
                            + " bytes. With "+str(nscens)
                            + " scenarios, it is expected that this run will pass the"
                            + " user-defined max_mem="+str(self.max_mem)
                            + " byte limit by a factor of: "+str(tot_mem/self.max_mem)
                            + ". To avoid, increase mem_total, reduce model size "
                            + "or number of scenarios, or run outside a parallel pool")


class MultiEventSimulation(MultiSimulation):
    staged: bool = True
    mdls: dict = dict()
    include_nominal: bool = True
    gen_samp: bool = False

    def __call__(self, **kwargs):
        rets = super().__call__(**kwargs)
        if self.gen_samp:
            rets = (*rets, self.samp)
        return rets

    def run_nom(self, with_copy=False, gen_samp=False, **kwargs):
        kwar = {'mdl': self.mdl, 'to_return': self.to_return}
        if with_copy:
            kwar['ctimes'] = self.samp.get_times()
        nomsim =  Simulation(**kwar)
        sim_kwar = dict(with_mdls=with_copy, gen_samp=gen_samp, **kwargs)
        if self.pool:
            outs = self.pool.apply(nomsim, (), sim_kwar)
        else:
            outs = nomsim(**sim_kwar)
        self.result['nominal'] = outs[0]
        self.history['nominal'] = outs[1]
        if with_copy:
            self.mdls = outs[2]
        if gen_samp:
            self.samp = outs[-1]

    def run(self, **kwargs):
        if self.gen_samp:
            self.run_nom(gen_samp=True, **kwargs)
            if self.staged:
                self.run_nom(with_copy=True)
        elif self.staged:
            self.run_nom(with_copy=True)
        elif self.include_nominal:
            self.run_nom()

        super().run(**kwargs)
        if not self.include_nominal:
            self.result.pop('nominal', None)
            self.history.pop('nominal', None)

    def gen_inputs(self, scenlist, **kwargs):
        sim_kwar = {'nomresult': self.result['nominal'],
                    'nomhist': self.history['nominal']}
        if self.staged:
            def_kwar = self.gen_default_kwargs(**kwargs, protect=False,
                                               copy=not bool(self.pool))
            return [({**def_kwar, 'scen': scen, 'mdl': self.mdls[scen.time].copy()},
                     sim_kwar) for scen in scenlist]
        else:
            def_kwar = self.gen_default_kwargs(**kwargs)
            return [({**def_kwar, 'scen': scen}, sim_kwar) for scen in scenlist]


def exec_fault_sim(args):
    kwar = args[0]
    kwar['mdl'] = kwar['mdl'].new(**get_mdl_kwargs(kwar.pop('scen', {})))
    sim = MultiEventSimulation(**kwar)
    if len(args) > 1:
        return sim(**args[1])
    else:
        return sim()

class NestedSimulation(MultiSimulation):
    get_phasemap: bool = True
    apps: dict = {}
    faultdomains: dict = {}
    faultsamples: dict = {}

    def pool_runner(self, inputs):
        return self.pool.imap(exec_fault_sim, inputs)

    def std_runner(self, inputs):
        return [exec_fault_sim(i) for i in inputs]

    def gen_default_kwargs(self, **kwargs):
        def_kwar = dict(mdl=self.mdl,
                        showprogress=False,
                        gen_samp=True)
        def_kwar = {**def_kwar, **kwargs}
        def_kwar = filter_kwargs(MultiEventSimulation, **def_kwar)
        return def_kwar

    def gen_inputs(self, scenlist, **kwargs):
        def_kwar = self.gen_default_kwargs(**kwargs)
        run_kwar = dict(get_phasemap=self.get_phasemap,
                        faultdomains=self.faultdomains,
                        faultsamples=self.faultsamples)
        return [({**def_kwar, 'mdl': self.mdl, 'scen': scen}, run_kwar)
                for scen in scenlist]


class Simulation(BaseSimulation):
    scen: Scenario = Scenario()
    protect: bool = True
    copy: bool = False
    ctimes: list = []
    mdl_kwargs: dict = {}
    simevents: list = []
    warn_faults = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.name:
            self.name = self.scen.name
        self.init_model()
        self.history = self.mdl.h
        self.create_simevents()

    def get_models(self):
        return {ev.time: ev.mdl_copy for ev in self.simevents if ev.mdl_copy}

    def gen_sampleapproach(self, faultdomains={}, faultsamples={}, get_phasemap=False):
        if get_phasemap:
            pm = from_hist(self.history)
            app = SampleApproach(self.mdl, phasemaps=pm)
        else:
            app = SampleApproach(self.mdl)
        app.add_faultdomains(**faultdomains)
        app.add_faultsamples(**faultsamples)
        return app

    def __call__(self, with_mdls=False, gen_samp=False, **kwargs):
        res, hist = super().__call__(**kwargs)
        self.check_faults()
        rets = [res, hist]
        if with_mdls:
            rets.append(self.get_models())
        if gen_samp:
            rets.append(self.gen_sampleapproach(**kwargs))
        return tuple(rets)

    def check_faults(self):
        if self.warn_faults:
            endfaults, faultprops = self.mdl.return_faultmodes()
            if endfaults:
                warnings.warn("Faults found during the nominal run " + str(endfaults))

    def init_model(self):
        if self.copy:
            self.mdl = self.mdl.copy()
        elif self.protect:
            self.mdl = self.mdl.new(**get_mdl_kwargs(self.scen))

    def create_simevents(self):
        res_sequence = {k: v for k, v in self.to_return.items()
                        if isinstance(k, float) or isinstance(k, int)}
        times = [float(i) for i in [*self.scen.sequence, *self.ctimes, *res_sequence]
                 if i > self.mdl.t.time]
        times.sort()

        for time in times:
            copy = time in self.ctimes
            injection = self.scen.sequence.get(time, None)
            to_ret = self.to_return.get(time, None)
            self.simevents.append(SimEvent(time=time, copy=copy, injection=injection,
                                           to_return=to_ret))
        to_ret = self.to_return.get('end', None)
        self.simevents.append(SimEvent(time='end', to_return=to_ret))

    def __repr__(self):
        repstr = get_dict_repr({str(ev.time): ev for ev in self.simevents},
                               one_line=False)
        return "Simulation with SimEvents:"+repstr

    def run(self, **kwargs):
        for simevent in self.simevents:
            simevent.run(self.mdl, scen=self.scen, **kwargs)
            if simevent.result:
                self.result[t_key(simevent.time)] = simevent.result


def get_mdl_kwargs(scen):
    return {'p':  scen.get("p", {}), 'sp': scen.get("sp", {}), 'r': scen.get("r", {})}


if __name__ == "__main__":
    from fmdtools.define.block.function import ExampleFunction
    from fmdtools.define.architecture.function import ExFxnArch
    from multiprocessing import Pool
    from fmdtools.sim.sample import ParameterDomain, ParameterSample, expd, exp_ps
    esf = ExampleFunction()
    n = NestedSimulation(mdl=esf, samp=exp_ps,
                         faultdomains={"fd": (("all",), {})},
                         faultsamples={"fs": (("fault_times", "fd", [1]), {})})
    n()

    efa = ExFxnArch()
    res5, hist5 = single_faults(efa, times=[5.0, 10.0])


    esf = ExampleFunction()
    s = SingleFaultScenario.from_fault(("examplefunction", "low"), 3.0, esf)
    sim = Simulation(mdl=esf, scen=s, ctimes = [2, 4], to_return={1.0: 's.x', "end": ["class", "graph"]})
    sim.run()
    nomres, nomhist = nominal(esf, to_return={5.0: 's.x', 7.0: 'class', 'end': 'class'})

    faultres, faulthist = one_fault(esf, "examplefunction", "low", 3.0)

    sim2 = MultiEventSimulation(mdl=esf, samp=s)
    res2, hist2 = sim2()

    sim3 = MultiEventSimulation(mdl=esf, samp=s, staged=False,
                                to_return={'end': ["class", "graph"]})
    res3, hist3 = sim3()


    s = SingleFaultScenario.from_fault(("ex_fxn", "low"), 3.0, efa)
    sim4 = MultiEventSimulation(mdl=efa, samp=s, staged=True, pool = Pool(5),
                                to_return={'end': ["class", "graph"]})
    res4, hist4 = sim4()



    psim = MultiSimulation(mdl=esf, samp=exp_ps)
    res7, hist7 = psim()



