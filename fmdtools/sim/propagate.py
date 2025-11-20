# -*- coding: utf-8 -*-
"""
Functions to propagate faults through a user-defined fault model.

Propagate provides the following convience methods:

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

Which themselves call the following classes:

- :class:`SimEvent`: Simulate an event in a given Simulable.
- :class:`Simulation`: Simulate a scenario in a given Simulable.
- :class:`MultiSimulation`: Simulate a sample of scenarios in a given Simulable.
- :class`MultiEventSimulation`: Simulate a sample of Fault scenarios in a Simulable.
- :class:`NestedSimulation`: Simulate fault scenarios within sets of parameters.

There are also private functions:

- :func:`get_sim_call_kwargs`: Get keyword arguments for a given simulation call
- :func:`get_mdl_kwargs`: Get keyword arguments from a scenario for model instantiation
- :func:`exec_sim`: Execute a Simulation (for mapping)
- :func:`exec_fault_sim`: Execute a MultiEventSimulation (for mapping)

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
from fmdtools.analyze.result import Result, clean_to_return
from fmdtools.analyze.history import History
from fmdtools.analyze.phases import from_hist


import numpy as np
import tqdm
import warnings


def get_sim_call_kwargs(sim, **kwargs):
    """Get keyword arguments corresponding to a simulation call."""
    return {**filter_kwargs(sim, **kwargs),
            **filter_kwargs(SimEvent.run, **kwargs)}


def get_mdl_kwargs(scen):
    """Get keyword arguments from a scenario corresponding to model parameters."""
    return {'p':  scen.get("p", {}), 'sp': scen.get("sp", {}), 'r': scen.get("r", {})}


def nominal(mdl, **kwargs):
    """
    Run the model over time in the nominal scenario.

    Parameters
    ----------
    mdl : Simulable
        Model of the system
    **kwargs : kwargs
        kwargs to Simulation and Simulation.__call__

    Returns
    -------
    result: Result
        A Result dict corresponding to values specified in to_return
    hist : History
        A History dict with a history of tracked model properties over time.
    """
    sim = Simulation(mdl=mdl, **filter_kwargs(Simulation, **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


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
        kwargs to MultiSimulation and MultiSimulation.__call__

    Returns
    -------
    result: Result
        Result dict of result corresponding to desired result {'scenname': result}
    hist : History
        Overall dict of model histories, with structure {'scenname': mdlhist}
    """
    sim = MultiSimulation(mdl=mdl, samp=ps, name="parameter sample",
                          **filter_kwargs(MultiSimulation, **kwargs))
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
        kwargs to propagate.sequence()

    Returns
    -------
    result: Result
        Result dict of result corresponding to to_return, with structure,::

        Result({'nominal': nomresult, 'scenname': faultyresult})

    mdlhists : History
        A dictionary of the states of the model of each fault scenario over time with
        structure: {'nominal': nomhist, 'scenname': faulthist}
    """
    if len(fxnfault) == 2:
        fxnname, fault = fxnfault
    elif len(fxnfault) == 3:
        fxnname, fault, time = fxnfault
    else:
        fxnname, fault = mdl.name, fxnfault[0]

    scen = SingleFaultScenario.from_fault((fxnname, fault), time, mdl=mdl, **f_kw)
    return sequence(mdl, scen=scen, name=str((fxnname, fault)), **kwargs)


def sequence(mdl, seq={}, faultseq={}, disturbances={}, scen={}, rate=np.nan,
             include_nominal=True, showprogress=False, **kwargs):
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
        Whether to additionally run the nominal scenario or just the given sequence.
        Default is True.
    **kwargs : kwargs
        kwargs to Simulation/MultiEventSimulation and __call__.

    Returns
    -------
    result: Result
        Result dict of result corresponding to to_return, with structure,::

        Result({'nominal': nomresult, 'scenname': faultyresult})

    mdlhists : History
        A dictionary of the states of the model of each fault scenario over time with
        structure: {'nominal': nomhist, 'scenname': faulthist}
    """
    if not scen:
        if not seq:
            seq = Sequence(faultseq=faultseq, disturbances=disturbances)
        scen = Scenario(sequence=seq,
                        rate=rate,
                        name=kwargs.get('name', 'sequence'),
                        times=tuple([*seq.keys()]),
                        time=min([*seq.keys()]))

    if include_nominal:
        sim = MultiEventSimulation(mdl=mdl, samp=scen,
                                   **filter_kwargs(MultiEventSimulation,
                                                   showprogress=showprogress, **kwargs))
    else:
        sim = Simulation(mdl=mdl, scen=scen,
                         **filter_kwargs(Simulation, showprogress=showprogress,
                                         **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


def fault_sample(mdl, fs, **kwargs):
    """
    Injects and propagates faults in the model defined by a FaultSample/SampleApproach.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in.
    fs : FaultSample/SampleApproach
        FaultSample used to define the list of faults and sample time for the model.
    **kwargs : kwargs
        Arguments to MultiEventSimulation and MultiEventSimulation.__call__

    Returns
    -------
    results : Result
        A Result dictionary with results desired from each scenario corresponding to
        to_return over the set of scenarios with structure {'scen': scenresult}
    mdlhists : History
        A History dictionary with the tracked scenario with structure
        {'scen': scenhist}
    """
    sim = MultiEventSimulation(mdl=mdl, samp=fs, name="fault sample",
                               **filter_kwargs(MultiEventSimulation, **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


def fault_sample_from(mdl, **kwargs):
    """Injects and propagates faults in a model given SampleApproach arguments."""
    return nested_sample(mdl, ps=Scenario(), **kwargs)


def single_faults(mdl, times=[0.0], **kwargs):
    """
    Create and propagates a list of failure scenarios in a model.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in
    times : list
        List of times to inject the single faults in. Default is [1.0]
    **kwargs : kwargs
        Arguments to MultiEventSimulation and MultiEventSimulation.__call__

    Returns
    -------
    results : Result
        A Result dictionary with results desired from each scenario corresponding to
        to_return over the set of scenarios with structure {'scen': scenresult}
    mdlhists : History
        A History dictionary with the tracked scenario with structure
        {'scen': scenhist}
    """
    fd = FaultDomain(mdl)
    fd.add_all()
    fs = FaultSample(fd)
    fs.add_fault_times(times)
    sim = MultiEventSimulation(mdl=mdl, samp=fs, name="single faults",
                               **filter_kwargs(MultiEventSimulation, **kwargs))
    return sim(**get_sim_call_kwargs(sim, **kwargs))


def nested_sample(mdl, ps, **kwargs):
    """
    Simulate a set of fault scenarios within a ParameterSample.

    Generates SampleApproaches to sample from based on the faultdomains, faultsamples
    keyword arguments within each scenario from the parameter sample.

    Parameters
    ----------
    mdl : Simulable
        Model Object to use in the simulation.
    ps : ParameterSample
        Parameter Scenario defining the parameters the model will be run over
    **kwargs : kwargs
        Keyword arguments to NestedSimulation and NestedSimulation.__call__

    Returns
    -------
    nested_results : Result
        A nested Result dictionary with the desired results of each scenario.
    nested_mdlhists : History
        A nested History dictionary with the history of all model states for each
        scenario
    apps : dict
        A dictionary of the SampleApproaches generated corresponding to each parameter
        scenario with structure {'nomscen1': app1}
    """
    sim = NestedSimulation(mdl=mdl, samp=ps, name="nested sample",
                           **filter_kwargs(NestedSimulation, **kwargs))
    res, hist = sim(**get_sim_call_kwargs(sim, **kwargs))
    return res, hist, sim.apps


class BaseSimContainer(BaseContainer):
    """Base Container class for SimEvent and BaseSimulation."""

    def __setattr__(self, field, value):
        """Revert to dataobject __setattr__."""
        super(BaseContainer, self).__setattr__(field, value)




class SimEvent(BaseSimContainer):
    """
    Event in a Simulation.

    SimEvent is a class that simulations use to run scenarios that describes the
    time to simulate the model to as well as the operations to perform at that time,
    such as copying, injecting faults, and returning results.

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

    Examples
    --------
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> se = SimEvent(time=5.0, copy=True, to_return=["classify"])
    >>> se
    SimEvent(time=5.0, copy=True, to_return=['classify'])
    >>> se.run(ExampleFunction())
    >>> se.mdl_copy # note that copied models are gotten just before the simulation
    examplefunction ExampleFunction
    - t=Time(time=4.0, timers={})
    - s=ExampleState(x=4.0, y=0.0)
    - m=ExampleMode(mode='standby', faults=set(), sub_faults=False)
    - exf=ExampleFlow(s=(x=1.0, y=1.0))
    >>> se.result
    classify: 
    --xy:                                4.0
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
        Simulate the event on the given model.

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


class BaseSimulation(BaseSimContainer):
    """
    Base Simulation class that other simulations inherit from.

    Does not in and of itself simulate, but provides the overall patterns that other
    simulations inherit.

    In general, simulations have a few common interfaces:
    (1) sim = Simulation(mdl=mdl, **kwargs) initializes the simulation for mdl
    (2) sim.run() runs the simulation and updates its properties
    (3) sim() runs the simulation, and saves and returns its result and history

    Parameters
    ----------
    name : str
        Name of the simulation.
    mdl : Simulable
        Model associated with the simulation
    to_return : dict
        Specification of results to return from the model during simulation. Can be
        specified in a few ways, including: {'time': ['val', 'classify', 'graph']}, where
        time would be the sim time to get the result from, 'val' would be a value to
        get, 'classify' would be what mdl.classify() returns, and 'graph' would be a graph.
    tosave : bool
        Whether to save the results/history of the sim. Default is False
    overwrite : bool
        Whether to overwrite existing files
    indiv_id : bool
        Whether to put individual id at the beginning of the save file.
    result_filename : str
        Name of the result to save as
    history_filename : str
        Name of the history to save as
    result : Result
        Result returned by the simulation
    history : History
        History of model states
    """

    name: str = ''
    mdl: Simulable = Block()
    to_return: dict = {"end": "classify"}
    tosave: bool = False
    overwrite: bool = False
    indiv_id: bool = False
    result_filename: str = "result.csv"
    history_filename: str = "history.csv"
    result: Result = Result()
    history: History = History()

    def __init__(self, *args, **kwargs):
        """Instantiate the sim and clean up to_return."""
        super().__init__(*args, **kwargs)
        self.to_return = clean_to_return(self.to_return)

    def save(self):
        """Save the results of the sim."""
        if self.indiv_id:
            result_id = self.name
        else:
            result_id = ''

        if self.result and self.result_filename:
            self.result.save(self.result_filename, result_id=result_id,
                             overwrite=self.overwrite)
        if self.history and self.history_filename:
            self.history.save(self.history_filename, result_id=result_id,
                              overwrite=self.overwrite)

    def __call__(self, **kwargs):
        """Run the sim, save its results (if needed), return its result and history."""
        try:
            self.run(**kwargs)
        except Exception as e:
            raise Exception("Error simulating "+self.name+" scenario(s)") from e
        if self.tosave:
            self.save(**filter_kwargs(self.save, **kwargs))
        return self.result.flatten(), self.history.flatten()


class Simulation(BaseSimulation):
    """
    Simulate a single scenario.

    Simulations can be used to do a few things, including:
    (1) Simulating a model in a particular faulty or nominal scenario
    (2) Getting model copies at particular times during a (usually nominal) scenario
    to simulate from. This is helpful for staged simulation where the copies are used
    to simulate faults from the point when the fault occurs instead of simulating
    up to that timestep first
    (3) Generating SampleApproaches that line up with the particular phases of the
    (usually nominal) scenario. This is helpful for nested simulations where each
    set of nominal parameters may change the phases of the simulation and thus the
    fault modes it may enter.

    Parameters
    ----------
    scen : Scenario
        Scenario to simulation
    protect : bool
        Whether to protect the model by re-initializing it.
    copy : bool
        Whether to copy the model prior to simulation
    ctimes : list
        Times to copy the model from during simulation
    simevents : list
        SimEvents to run the system over. Generated from scenario, ctypes, etc.
    warn_faults : bool
        Whether to warn if the simulation is faulty and no faults were injected. Used
        for nominal simulations.

    Examples
    --------
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> sim = Simulation(mdl=ExampleFunction()) # default is nominal simulation
    >>> sim.name
    'nominal'
    >>> sim
    Simulation with SimEvents:
    - end=SimEvent(to_return={'classify': None})
    >>> res, hist = sim()
    >>> res
    tend.classify.xy:                  100.0

    >>> esf = ExampleFunction() # here we try sometime more complicated:
    >>> s = SingleFaultScenario.from_fault(("examplefunction", "low"), 3.0, esf)
    >>> sim = Simulation(mdl=esf, scen=s, ctimes = [2, 4], to_return={1.0: 's.x', "end": ["classify", "graph"]})
    >>> sim
    Simulation with SimEvents:
    - 1.0=SimEvent(to_return={'s.x': None})
    - 2.0=SimEvent(copy=True)
    - 3.0=SimEvent(injection=Injection(faults={'examplefunction': ['low']}, disturbances={}))
    - 4.0=SimEvent(copy=True)
    - end=SimEvent(to_return={'classify': None, 'graph': None})
    >>> res, hist = sim()
    >>> sim.mdl
    examplefunction ExampleFunction
    - t=Time(time=100.0, timers={})
    - s=ExampleState(x=20.0, y=294.0)
    - m=ExampleMode(mode='low', faults={'low'}, sub_faults=False)
    - exf=ExampleFlow(s=(x=1964.0, y=1.0))
    >>> [*res.keys()]
    ['t1p0.s.x', 'tend.classify.xy', 'tend.graph']
    >>> sim.get_models()
    {2.0: examplefunction ExampleFunction
    - t=Time(time=1.0, timers={})
    - s=ExampleState(x=1.0, y=0.0)
    - m=ExampleMode(mode='standby', faults=set(), sub_faults=False)
    - exf=ExampleFlow(s=(x=1.0, y=1.0)), 4.0: examplefunction ExampleFunction
    - t=Time(time=3.0, timers={})
    - s=ExampleState(x=20.0, y=3.0)
    - m=ExampleMode(mode='low', faults={'low'}, sub_faults=False)
    - exf=ExampleFlow(s=(x=1.0, y=1.0))}
    """

    scen: Scenario = Scenario()
    protect: bool = True
    copy: bool = False
    ctimes: list = []
    simevents: list = []
    warn_faults: bool = True

    def __init__(self, *args, **kwargs):
        """Initialize and create the list of simulation events to run."""
        super().__init__(*args, **kwargs)
        if not self.name:
            self.name = self.scen.name
        self.init_model()
        self.history = self.mdl.h
        self.create_simevents()

    def __repr__(self):
        """Show event sequence in str."""
        repstr = get_dict_repr({str(ev.time): ev for ev in self.simevents},
                               one_line=False)
        return "Simulation with SimEvents:"+repstr

    def init_model(self):
        """
        Initialize the model.

        If self.protect, re-intiates the simulation with scenario parameters.
        If self.copy, copies the simulation.
        """
        if self.copy:
            self.mdl = self.mdl.copy()
        elif self.protect:
            self.mdl = self.mdl.new(**get_mdl_kwargs(self.scen))

    def create_simevents(self):
        """Create the list of SimEvents to run for during the simulation."""
        res_sequence = {k: v for k, v in self.to_return.items()
                        if isinstance(k, float) or isinstance(k, int)}
        try:
            times = [float(i) for i in [*self.scen.sequence, *self.ctimes, *res_sequence]
                     if i > self.mdl.t.time]
        except:
            raise
        times = [*set(times)]
        times.sort()

        for time in times:
            copy = time in self.ctimes
            injection = self.scen.sequence.get(time, None)
            to_ret = self.to_return.get(time, None)
            self.simevents.append(SimEvent(time=time, copy=copy, injection=injection,
                                           to_return=to_ret))
        to_ret = self.to_return.get('end', None)
        self.simevents.append(SimEvent(time='end', to_return=to_ret))

    def __call__(self, with_mdls=False, gen_samp=False, **kwargs):
        """Call the sim. Options return copied models and generated sampleapproches."""
        res, hist = super().__call__(**kwargs)
        self.check_faults()
        rets = [res, hist]
        if with_mdls:
            rets.append(self.get_models())
        if gen_samp:
            rets.append(self.gen_sampleapproach(**kwargs))
        return tuple(rets)

    def run(self, **kwargs):
        """Run the simevents and add results to the result dict."""
        for simevent in self.simevents:
            simevent.run(self.mdl, scen=self.scen, **kwargs)
            if simevent.result:
                self.result[t_key(simevent.time)] = simevent.result

    def get_models(self):
        """Get copied models from the simulation."""
        return {ev.time: ev.mdl_copy for ev in self.simevents if ev.mdl_copy}

    def gen_sampleapproach(self, faultdomains={}, faultsamples={}, get_phasemap=False):
        """
        Generate a SampleApproach from the simulation model.

        Parameters
        ----------
        faultdomains : dict, optional
            dict of arguments to add_faultdomains. The default is {}.
        faultsamples : dict, optional
            dict of arguments to add_faultsamples. The default is {}.
        get_phasemap : bool, optional
            Whether to use the history to generate the phasemap used by SampleApproach.
            The default is False.

        Returns
        -------
        app : SampleApproach
            Appropriate SampleApproach corresponding to this scenario model/hist
        """
        if get_phasemap:
            pm = from_hist(self.history)
            app = SampleApproach(self.mdl, phasemaps=pm)
        else:
            app = SampleApproach(self.mdl)
        app.add_faultdomains(**faultdomains)
        app.add_faultsamples(**faultsamples)
        return app

    def check_faults(self):
        """Check whether there are faults in a nominal simulation (a bad sign)."""
        if self.warn_faults and self.scen.name == 'nominal':
            endfaults = self.mdl.return_faultmodes()
            if endfaults:
                warnings.warn("Faults found during the nominal run " + str(endfaults))


def exec_sim(args):
    """Execute a given simulation. Used to interface with multiprocessing.imap."""
    sim = Simulation(**args[0])
    if len(args) > 1:
        return sim(**args[1])
    else:
        return sim()


def close_pool(pool=None):
    """Close a pool if present."""
    if pool:
        pool.close()
        pool.terminate()
        pool.join()


class MultiSimulation(BaseSimulation):
    """
    Simulation of multiple models.

    Generally, MultiSimulations are used to simulate a model over a range of parameters.

    MultiSimulations can be passed pool arguments to speed up simulation. Remember to
    to use the following protection statement like this prior to any call:;
        if __name__ == 'main':
            result, hist = MultiSimulation(mdl=mdl, pool=Pool(5))

    Otherwise, the method will keep spawning parallel processes.
    See multiprocessing documentation.

    Parameters
    ----------
    samp : BaseSample
        Sample to simulate the model over.
    save_indiv : bool
        Whether to save the results of the simulation together or apart
    pool : Pool
        Pool (e.g., multiprocessing.Pool) to use for parallel simulation
    auto_close_pool : bool
        Whether to automatically close the pool when done simulating
    showprogress : bool
        Whether to provide a tqdm-based progress bar
    max_mem : int
        Maximum memory limit for simulations. Produces an exception if breached to
        avoid overwhelming computational resources

    Examples
    --------
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> from fmdtools.sim.sample import exp_ps_45
    >>> esf = ExampleFunction()
    >>> psim = MultiSimulation(mdl=esf, samp=exp_ps_45, showprogress=False)
    >>> psim
    MultiSimulation of
    ParameterSample of scenarios:
     - rep0_range_0
     - rep0_range_1
     - rep0_range_2
     - rep0_range_3
     - rep0_range_4
     - rep0_range_5
     - rep0_range_6
     - rep0_range_7
     - rep0_range_8
     - rep0_range_9
     - ... (44 total)
     >>> res, hist = psim()

     # note the results - this parameter sample increases the x and y parameter, raising
     # the amount x increments and thus the "xy" classification at the end
     # y has no such effect since it only increments during faults
     >>> res
     rep0_range_0.tend.classify.xy:       0.0
     rep0_range_1.tend.classify.xy:     100.0
     rep0_range_2.tend.classify.xy:     200.0
     rep0_range_3.tend.classify.xy:     300.0
     rep0_range_4.tend.classify.xy:     400.0
     rep0_range_5.tend.classify.xy:     500.0
     rep0_range_6.tend.classify.xy:     600.0
     rep0_range_7.tend.classify.xy:     700.0
     rep0_range_8.tend.classify.xy:     800.0
     rep0_range_9.tend.classify.xy:     900.0
     rep0_range_10.tend.classify.xy:   1000.0
     rep0_range_11.tend.classify.xy:      0.0
     rep0_range_12.tend.classify.xy:    100.0
     rep0_range_13.tend.classify.xy:    200.0
     rep0_range_14.tend.classify.xy:    300.0
     rep0_range_15.tend.classify.xy:    400.0
     rep0_range_16.tend.classify.xy:    500.0
     rep0_range_17.tend.classify.xy:    600.0
     rep0_range_18.tend.classify.xy:    700.0
     rep0_range_19.tend.classify.xy:    800.0
     rep0_range_20.tend.classify.xy:    900.0
     rep0_range_21.tend.classify.xy:   1000.0
     rep0_range_22.tend.classify.xy:      0.0
     rep0_range_23.tend.classify.xy:    100.0
     rep0_range_24.tend.classify.xy:    200.0
     rep0_range_25.tend.classify.xy:    300.0
     rep0_range_26.tend.classify.xy:    400.0
     rep0_range_27.tend.classify.xy:    500.0
     rep0_range_28.tend.classify.xy:    600.0
     rep0_range_29.tend.classify.xy:    700.0
     rep0_range_30.tend.classify.xy:    800.0
     rep0_range_31.tend.classify.xy:    900.0
     rep0_range_32.tend.classify.xy:   1000.0
     rep0_range_33.tend.classify.xy:      0.0
     rep0_range_34.tend.classify.xy:    100.0
     rep0_range_35.tend.classify.xy:    200.0
     rep0_range_36.tend.classify.xy:    300.0
     rep0_range_37.tend.classify.xy:    400.0
     rep0_range_38.tend.classify.xy:    500.0
     rep0_range_39.tend.classify.xy:    600.0
     rep0_range_40.tend.classify.xy:    700.0
     rep0_range_41.tend.classify.xy:    800.0
     rep0_range_42.tend.classify.xy:    900.0
     rep0_range_43.tend.classify.xy:   1000.0
    """

    samp: object = Scenario()
    save_indiv: bool = False
    pool: object = None
    auto_close_pool: bool = True
    showprogress: bool = True
    max_mem: int = 2e9

    def __init__(self, *args, **kwargs):
        """Initialize the simulation an check memory."""
        super().__init__(*args, **kwargs)
        if not self.name and hasattr(self.samp, 'name'):
            self.name = self.samp.name
        if self.save_indiv is True:
            self.tosave = False
        self.check_hist_memory()
        self.check_mdl_memory()

    def __repr__(self, **kwargs):
        """Show the samp to run in repr."""
        return self.__class__.__name__+" of\n"+self.samp.__repr__()

    def close_pool(self):
        """Close the pool so that the threads do not keep running."""
        close_pool(self.pool)

    def run(self, **kwargs):
        """Simulate the scenarios, closing the pool when done."""
        if self.pool:
            runner = self.pool_runner
        else:
            runner = self.std_runner
        scenlist = self.samp.scenarios()
        try:
            inputs = self.gen_inputs(scenlist)
        except KeyError as e:
            raise Exception("Not enough models: "+str(self.mdls)) from e
        res_list = list(tqdm.tqdm(runner(inputs),
                                  total=len(inputs),
                                  disable=not (self.showprogress),
                                  desc="SCENARIOS COMPLETE"))
        for i, scen in enumerate(scenlist):
            self.get_output(res_list, scen.name, i)
        if self.auto_close_pool:
            self.close_pool()

    def get_output(self, res_list, name, i):
        """Attach the output of the results list to the history/result."""
        self.result[name] = res_list[i][0]
        self.history[name] = res_list[i][1]

    def gen_sim_kwargs(self, **kwargs):
        """Generate the arguments for the Simulations to run."""
        def_kwar = dict(mdl=self.mdl,
                        tosave=self.save_indiv,
                        result_filename=self.result_filename,
                        history_filename=self.history_filename,
                        indiv_id=True,
                        overwrite=self.overwrite,
                        to_return=self.to_return,
                        protect=not bool(self.pool))
        def_kwar = {**def_kwar, **kwargs}
        def_kwar = filter_kwargs(Simulation, **def_kwar)
        return def_kwar

    def gen_inputs(self, scenlist, **kwargs):
        """Generate the inputs to Simulation to map exec_sim over."""
        sim_kwar = self.gen_sim_kwargs(**kwargs)
        return [({**sim_kwar, 'scen': scen},) for scen in scenlist]

    def pool_runner(self, inputs):
        """Run the simulations using a parallel pool."""
        return self.pool.imap(exec_sim, inputs)

    def std_runner(self, inputs):
        """Run the simulations normally (without a parallel pool)."""
        return map(exec_sim, inputs)

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
    """
    Simulate one or multiple SimEvents over a list of scenarios.

    Can be used to simulate multiple fault scenarios at different times. Enables staged
    model execution, which simulates the nominal scenario and copies the simulation
    at that time to simulate the post-event response while minimizing duplicate
    pre-event simulation times.

    Parameters
    ----------
    staged : bool
        Whether to simulate using a staged fashion. If not, models are initialized at
        the start time instead of being copied
    mdls : dict
        dict of models at various times to simulate over. Returned by the simulation of
        the nominal scenario
    include_nominal : bool
        Whether to include the nominal simulation. If False, the nominal simulation is
        only run if necessary
    gen_samp : bool
        Whether to generate a SampleApproach from the nominal simulation. Used with
        NestedSimulations to generate the sample prior to simulation.

    Examples
    --------
    >>> from fmdtools.define.architecture.function import ExFxnArch
    >>> from fmdtools.sim.sample import exfs
    >>> sim = MultiEventSimulation(mdl=ExFxnArch(), samp=exfs, showprogress=False)
    >>> sim
    MultiEventSimulation of
    FaultSample of scenarios: 
     - exfxnarch_fxns_ex_fxn_no_charge_t1
     - exfxnarch_fxns_ex_fxn_no_charge_t2
     - exfxnarch_fxns_ex_fxn2_no_charge_t1
     - exfxnarch_fxns_ex_fxn2_no_charge_t2
     - exfxnarch_fxns_ex_fxn_short_t1
     - exfxnarch_fxns_ex_fxn_short_t2
     - exfxnarch_fxns_ex_fxn2_short_t1
     - exfxnarch_fxns_ex_fxn2_short_t2
    >>> res, hist = sim()
    >>> res
    nominal.tend.classify.flowval:   10100.0
    exfxnarch_fxns_ex_fx              5050.0
    exfxnarch_fxns_ex_fx              5150.0
    exfxnarch_fxns_ex_fx              5050.0
    exfxnarch_fxns_ex_fx              5150.0
    exfxnarch_fxns_ex_fx              5050.0
    exfxnarch_fxns_ex_fx              5150.0
    exfxnarch_fxns_ex_fx              5050.0
    exfxnarch_fxns_ex_fx              5150.0

    This should also be the result when not running using staged options...
    >>> sim = MultiEventSimulation(mdl=ExFxnArch(), samp=exfs, staged=False, showprogress=False)
    >>> res, hist = sim()
    >>> res
    nominal.tend.classify.flowval:   10100.0
    exfxnarch_fxns_ex_fx              5050.0
    exfxnarch_fxns_ex_fx              5150.0
    exfxnarch_fxns_ex_fx              5050.0
    exfxnarch_fxns_ex_fx              5150.0
    exfxnarch_fxns_ex_fx              5050.0
    exfxnarch_fxns_ex_fx              5150.0
    exfxnarch_fxns_ex_fx              5050.0
    exfxnarch_fxns_ex_fx              5150.0
    """

    staged: bool = True
    mdls: dict = dict()
    include_nominal: bool = True
    gen_samp: bool = False

    def __call__(self, **kwargs):
        """Return the SampleApproach if gen_samp."""
        rets = super().__call__(**kwargs)
        if self.gen_samp:
            rets = (*rets, self.samp)
        return rets

    def run_nom(self, with_copy=False, gen_samp=False, with_save=True, **kwargs):
        """
        Run the nominal scenario and get the relevant information.

        Parameters
        ----------
        with_copy : bool, optional
            Whether to get copies at sample times. Used for staged simulation.
            The default is False.
        gen_samp : bool, optional
            Whether to generate a SampleApproach using the simulation. Used for nested
            simulations. The default is False.
        **kwargs : kwargs
            Keyword arguments to Simulation.__call__, such as faultdomains and
            faultsamples for generating the SampleApproach
        """
        kwar = {'mdl': self.mdl, 'to_return': self.to_return}
        if with_copy:
            kwar['ctimes'] = self.samp.get_times()
        if with_save:
            kwar.update({'tosave': self.save_indiv,
                         'result_filename': self.result_filename,
                         'history_filename': self.history_filename, 'indiv_id': True})
        nomsim = Simulation(**kwar)
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
        """Run the simulations, starting with nominal (when applicable)."""
        if self.gen_samp:
            self.run_nom(gen_samp=True, **kwargs)
            if self.staged:
                self.run_nom(with_copy=True, with_save=True)
        elif self.staged:
            self.run_nom(with_copy=True)
        elif self.include_nominal:
            self.run_nom()

        super().run(**kwargs)
        if not self.include_nominal:
            self.result.pop('nominal', None)
            self.history.pop('nominal', None)

    def gen_inputs(self, scenlist, **kwargs):
        """Generate the inputs for exec_sim, including nominal history and result."""
        call_kwar = {'nomresult': self.result['nominal'],
                     'nomhist': self.history['nominal']}
        if self.staged:
            sim_kwar = self.gen_sim_kwargs(**kwargs, protect=False,
                                           copy=not bool(self.pool))
            return [({**sim_kwar, 'scen': scen, 'mdl': self.mdls[scen.time].copy()},
                     call_kwar) for scen in scenlist]
        else:
            sim_kwar = self.gen_sim_kwargs(**kwargs)
            return [({**sim_kwar, 'scen': scen}, call_kwar) for scen in scenlist]


def exec_fault_sim(args):
    """Execute MultiEventSimulation. Used so that results can be mapped."""
    kwar = args[0]
    kwar['mdl'] = kwar['mdl'].new(**get_mdl_kwargs(kwar.pop('scen', {})))
    sim = MultiEventSimulation(**kwar)
    if len(args) > 1:
        return sim(**args[1])
    else:
        return sim()


class NestedSimulation(MultiSimulation):
    """
    Simulate multiple MultiEventSimulations over a parameter sample.

    NestedSimulation works by call MultiEventSimulation over a set of scenarios
    defined in ParameterSample, using them to generate fault scenarios using
    the given faultdomains and faultsamples arguments

    Parameters
    ----------
    faultdomains: dict
        Dict of arguments to SampleApproach.add_faultdomain
    faultsamples: dict
        Dict of agruments to SampleApproach.add_faultsample
    staged : bool
        Whether the samples should be simulated in a staged manner
    get_phasemap: bool
        Whether to use the phasemap to generate the SampleApproaches
    apps : dict
        Returned approaches from MultiEventSimulation
    include_nominal : bool
        Whether to include the nominal scenario in the results

    Examples
    --------
    >>> from fmdtools.sim.sample import ParameterSample, expd
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> esf = ExampleFunction()
    >>> exp_ps_4 = ParameterSample(expd, seed=1)
    >>> exp_ps_4.add_variable_replicates([[1,0], [1,1], [2,1], [2,2]])
    >>> n = NestedSimulation(mdl=esf, samp=exp_ps_4, faultdomains={"fd": (("fault", "examplefunction", "low"), {})}, faultsamples={"fs": (("fault_times", "fd", [0]), {})}, showprogress=False)
    >>> n
    NestedSimulation of
    ParameterSample of scenarios:
     - rep0_var_0
     - rep0_var_1
     - rep0_var_2
     - rep0_var_3
     over 
    SampleApproach({'fd': (('fault', 'examplefunction', 'low'), {})},
    {'fs': (('fault_times', 'fd', [0]), {})})
    >>> res, hist = n()
    >>> res
    rep0_var_0.nominal.tend.classify.xy: 0.0
    rep0_var_0.examplefu               120.0
    rep0_var_1.nominal.tend.classify.xy: 100.0
    rep0_var_1.examplefu               120.0
    rep0_var_2.nominal.tend.classify.xy: 100.0
    rep0_var_2.examplefu               220.0
    rep0_var_3.nominal.tend.classify.xy: 200.0
    rep0_var_3.examplefu               220.0
    >>> n.apps
    {'rep0_var_0': SampleApproach for examplefunction with: 
     faultdomains: fd
     faultsamples: fs, 'rep0_var_1': SampleApproach for examplefunction with: 
     faultdomains: fd
     faultsamples: fs, 'rep0_var_2': SampleApproach for examplefunction with: 
     faultdomains: fd
     faultsamples: fs, 'rep0_var_3': SampleApproach for examplefunction with: 
     faultdomains: fd
     faultsamples: fs}

    Note that these results are consistent with what we expect from the behavior. In the
    nominal scenarios, s.x increments every timestep by p.x. In the faulty scenario,
    s.x is set to 20 and s.y increments by p.y. This gives the above
    results for x+y at the end of 100 timesteps.
    The results should be the same with the various options we can provide.
    >>> n = NestedSimulation(mdl=esf, samp=exp_ps_4, faultdomains={"fd": (("fault", "examplefunction", "low"), {})}, faultsamples={"fs": (("fault_times", "fd", [0]), {})}, showprogress=False, staged=False)
    >>> res, hist = n()
    >>> res
    rep0_var_0.nominal.tend.classify.xy: 0.0
    rep0_var_0.examplefu               120.0
    rep0_var_1.nominal.tend.classify.xy: 100.0
    rep0_var_1.examplefu               120.0
    rep0_var_2.nominal.tend.classify.xy: 100.0
    rep0_var_2.examplefu               220.0
    rep0_var_3.nominal.tend.classify.xy: 200.0
    rep0_var_3.examplefu               220.0
    """

    faultdomains: dict = {}
    faultsamples: dict = {}
    staged: bool = True
    get_phasemap: bool = False
    apps: dict = {}
    include_nominal: bool = True

    def __repr__(self):
        """Add str representing sample arguments."""
        st = ("\n over \nSampleApproach(" +
              str(self.faultdomains) +
              ",\n" +
              str(self.faultsamples) +
              ")")
        return super().__repr__()+st

    def pool_runner(self, inputs):
        """Call exec_fault_sim during run."""
        return self.pool.imap(exec_fault_sim, inputs)

    def std_runner(self, inputs):
        """Call exec_fault_sim during run."""
        return map(exec_fault_sim, inputs)

    def get_output(self, reslist, name, i):
        """Get approaches from output so they can be inspected."""
        super().get_output(reslist, name, i)
        self.apps[name] = reslist[i][2]

    def gen_sim_kwargs(self, **kwargs):
        """Generate applicable simulation kwargs for MultiEventSimulation."""
        def_kwar = dict(mdl=self.mdl,
                        tosave=self.save_indiv,
                        result_filename=self.result_filename,
                        history_filename=self.history_filename,
                        indiv_id=True,
                        overwrite=self.overwrite,
                        to_return=self.to_return,
                        include_nominal=self.include_nominal,
                        staged=self.staged,
                        showprogress=False,
                        gen_samp=True)
        def_kwar = {**def_kwar, **kwargs}
        def_kwar = filter_kwargs(MultiEventSimulation, **def_kwar)
        return def_kwar

    def gen_inputs(self, scenlist, **kwargs):
        """Generate input to exec_fault_sim, including args to SampleApproach."""
        sim_kwar = self.gen_sim_kwargs(**kwargs)
        run_kwar = dict(get_phasemap=self.get_phasemap,
                        faultdomains=self.faultdomains,
                        faultsamples=self.faultsamples)
        return [({**sim_kwar, 'mdl': self.mdl, 'scen': scen, 'name': scen.name}, run_kwar)
                for scen in scenlist]


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
