#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: A simple model for explaining fault model definition.

This model constitudes an extremely simple functional model of an
electric-powered pump.

The functions are:
    -import EE
    -import Water
    -import Signal
    -move Water
    -export Water

The flows are:
    - EE (to power the pump)
    - Water_in
    - Water_out
    - Signal input (on/off)

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode
from fmdtools.define.flow.base import Flow
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.architecture.function import FunctionArchitectureGraph
from fmdtools.define.architecture.base import check_model_pickleability
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.container.time import Time
import fmdtools.analyze as an
import fmdtools.sim.propagate as propagate

import numpy as np

"""
DEFINING MODEL FLOWS
Flows can be defined using Python classes that are instantiated as objects.

Flows contain State objects which hold variables, but may be given other attributes
(parameters etc)
"""


class WaterStates(State):
    """States for Water Flows."""

    flowrate: float = 1.0
    pressure: float = 1.0
    area: float = 1.0
    level: float = 1.0


class Water(Flow):
    """Flow connecting water from input to pump."""

    __slots__ = ()
    container_s = WaterStates


class EEStates(State):
    """States for electrical energy flows."""

    current: float = 1.0
    voltage: float = 1.0


class Electricity(Flow):
    """Flow connecting electricity from input to pump."""

    __slots__ = ()
    container_s = EEStates


class SignalStates(State):
    """States of Signal Flows."""

    power: float = 1.0


class Signal(Flow):
    """Flow connecting signal from input to pump."""

    __slots__ = ()
    container_s = SignalStates


"""
DEFINING RESILIENCE METRICS
Below we define certain functions used in the value function in find_classification
"""


def reseting_accumulate(vec):
    """
    Accummulate vector for all positive output.

    Examples
    --------
    >>> reseting_accumulate([1,1,1, 0, 1,1])
    [1, 2, 3, 0, 1, 2]
    """
    newvec = vec
    val = 0
    for ind, i in enumerate(vec):
        if i > 0:
            val = i + val
        else:
            val = 0
        newvec[ind] = val
    return newvec


def accumulate(vec):
    """
    Accummulate vector.

    Examples
    --------
    >>> accumulate([1, 1, 1, 0, 1, 1])
    [1, 2, 3, 3, 4, 5]
    """
    return [sum(vec[:i+1]) for i in range(len(vec))]


"""
DEFINING MODEL PARAMETERS
Below we define a class that defines the parameter of the model
"""


class PumpParam(Parameter, readonly=True):
    """PumpParam defines the parameters which the pump may be simulated over."""

    # costs to tabulate in cost model (see find_classification)
    cost: tuple = ("repair", "water")
    # delay to use in MoveWater function
    delay: int = 10
    # valid limits for delay
    delay_lim = (0, 100)


"""
DEFINE MODEL FUNCTIONS
Functions are defined using Python classes that are instantiated as objects.

Functions are additionally composed of the following classes:
    - Mode for faulty and operational modes
    - State for variables
    - Flow(s) for flow connections
    - ... and several others
"""


class ImportEEMode(Mode):
    """
    Mode contains the probability model for faults.

    Mode may be associated with each function:
        - failrate = X sets the failure rate for the function
        (to be distributed over all modes)
        - fm_args defines a probability model for each mode, where modes is:
            - {modename: (%of failures, (% at each phase in mdl.phases), repaircosts)
    These failure rates will then be used to generate a list of scenarios for
    propagate.single_faults and FaultSample()

    Note that these rates are given in occurences/hr by default. To change the units,
    use the option units='sec'/'min'/'hr'/'day' etc.
    """

    failrate = 1e-5
    fm_args = {'no_v': (0.80, 10000),
               'inf_v': (0.20, 5000)}
    phases = {'start': 0, 'on': 1, 'end': 0}


class ImportEEState(State):
    """Effectiveness of importing electrical energy."""

    effstate: float = 1.0


class ImportEE(Function):
    """
    Import EE is the line of electricity going into the pump.

    We define it here as a subclass of the Function superclass (imported from define.py)
    the Function superclass, which adds the common aspects of the function objects.

    Notice how container_m, container_s, flow_ee_out variables are assigned to
    Modes, States classes, and the Electricity flow used in this Function. This binds
    types to the Function so they are instiantiated and take the `m` (for mode) and `s`
    (for state) container in the Function, respectively. For flow `ee_out`, this defines
    a Flow variable which will be electricity and will additional be held in .flows
    here. The flownames variable then tells us that `ee_1` at the model level will be
    `ee_out` at the function level
    (which isn't necessary if they are given the same name).
    """

    __slots__ = ['ee_out']
    container_m = ImportEEMode
    container_s = ImportEEState
    flow_ee_out = Electricity
    flownames = {"ee_1": "ee_out"}

    def set_faults(self):
        """
        Conditional fault behavior.

        set_faults() changes the state of the system if there is a change in state in a
        flow.

        Using methods for specific behaviors is optional but can be helpful, in this
        case for delinating between the determination of a faults and their resulting
        behaviors.

        In this example,  if the current is too high, the line becomes an open circuit
        (e.g. due to a fuse or line burnout)
        """
        if self.ee_out.s.current > 15.0:
            self.m.add_fault('no_v')

    def static_behavior(self, time):
        """
        Electricity input behavior.

        behavior() defines the behavior of the function in terms of
        how the system behaves normally and under faults.
        """
        self.set_faults()
        if self.m.has_fault('no_v'):
            self.s.effstate = 0.0  # an open circuit means no voltage is exported
        elif self.m.has_fault('inf_v'):
            self.s.effstate = 100.0  # a voltage spike means voltage is much higher
        else:
            self.s.effstate = 1.0  # normally, voltage is 500 V
        self.ee_out.s.voltage = self.s.effstate * 500


class ImportWaterMode(Mode):
    """Fault modes for importing water."""

    failrate = 1e-5
    fm_args = {'no_wat': (1.0, 1000), 'less_wat': (1.0, 0.0)}


class ImportWater(Function):
    """Import Water is the pipe with water going into the pump."""

    __slots__ = ['wat_out']
    container_m = ImportWaterMode
    flow_wat_out = Water
    flownames = {"wat_1": "wat_out"}

    def static_behavior(self, time):
        """If the flow has a no_wat fault, the water level goes to zero."""
        if self.m.has_fault('no_wat'):
            self.wat_out.s.level = 0.0
        elif self.m.has_fault('less_wat'):
            self.wat_out.s.level = 0.5
        else:
            self.wat_out.s.level = 1.0


class ExportWaterMode(Mode):
    """Fault modes for exporting water."""

    failrate = 1e-5
    fm_args = {'block': (1.0, 5000)}
    phases = {'start': 1.5, 'on': 1, 'end': 1}


class ExportWater(Function):
    """Export Water is the pipe with water going out of the pump."""

    __slots__ = ['wat_in']
    container_m = ExportWaterMode
    flow_wat_in = Water
    flownames = {'wat_2': 'wat_in'}

    def static_behavior(self, time):
        """Blockage changes the area the output water flows through."""
        if self.m.has_fault('block'):
            self.wat_in.s.area = 0.01


class ImportSigMode(Mode):
    """Fault modes for signal input."""

    failrate = 1e-6
    fm_args = {'no_sig': (1.0, 10000, {'start': 1.5, 'on': 1, 'end': 1})}


class ImportSig(Function):
    """Import Signal is the on/off switch."""

    __slots__ = ['sig_out']
    container_m = ImportSigMode
    flow_sig_out = Signal
    flownames = {'sig_1': 'sig_out'}

    def static_behavior(self, time):
        """
        Time-dependent behavior for the function.

        To have different operational modes depending on the time, use if/else
        statements on the time variable, which is the simulation time.

        In this case, the power turns on at t=5 and turns back off at t=50.
        """
        if self.m.has_fault('no_sig'):
            self.sig_out.s.power = 0.0
            # an open circuit means no voltage is exported
        else:
            if time < 5:
                self.sig_out.s.power = 0.0
            elif time < 50:
                self.sig_out.s.power = 1.0
            else:
                self.sig_out.s.power = 0.0


class MoveWatTime(Time):
    """Specialized time class specifying to keep a pressure_limit timer."""

    timernames = ('pressure_limit',)


class MoveWatStates(State):
    """State of the pump effectiveness."""

    eff: float = 1.0


class MoveWatParams(Parameter, readonly=True):
    """Delay parameter affecting how long it takes for the pump to break."""

    delay: int = 1


class MoveWatMode(Mode):
    """Failure modes involved in moving water."""

    failrate = 1e-5
    fm_args = {'mech_break': (0.6, 5000, {'start': 0.1, 'on': 1.2, 'end': 0.1}),
               'short': (1.0, 10000, {'start': 1.5, 'on': 1, 'end': 1})}


class MoveWat(Function):
    """
    Move Water is the pump itself.

    While one could decompose this further, one function is used for simplicity.

    Note how this Function has more roles being filled:

    - s (states) by MoveWatStates
    - p (parameter) by MoveWatParams, which lets us parameterize a delay
    - m (mode) by MoveWatMode
    - t (time) by MoveWatTime, which will be used so we can have a timer
    """

    __slots__ = ['ee_in', 'sig_in', 'wat_in', 'wat_out']
    container_s = MoveWatStates
    container_p = MoveWatParams
    container_m = MoveWatMode
    container_t = MoveWatTime
    flow_ee_in = Electricity
    flow_sig_in = Signal
    flow_wat_in = Water
    flow_wat_out = Water
    flownames = {"ee_1": "ee_in", "sig_1": "sig_in",
                 "wat_1": "wat_in", "wat_2": "wat_out"}

    def set_faults(self, time):
        """
        Here we use the timer to define a conditional fault that only occurs after a
        state is present after X seconds.

        We do that by incrementing the timer when the state is present.

        Note that this is done with the internal timestep dt, which we can change
        locally (for the function) by passing dt=timestep in the super().__init__ method
        or globally by changing 'tstep' in modelparams.

        When the timer exceeds the delay defined by the external variable, the fault is
        added.
        """
        if self.p.delay:
            if self.indicate_over_pressure(time):
                if time > self.t.time:
                    self.t.pressure_limit.inc(self.t.dt)
                if self.t.pressure_limit.time >= self.p.delay:
                    self.m.add_fault('mech_break')
        else:
            if self.indicate_over_pressure(time):
                self.m.add_fault('mech_break')

    def indicate_over_pressure(self, time):
        """
        Use methods with names indicate_XXX to mark conditions met by the model.

        Indicators return booleans which are then recorded in the .i structure in the
        model history.
        """
        return self.wat_out.s.pressure > 15.0

    def static_behavior(self, time):
        """Define how the function will behave with different faults."""
        self.set_faults(time)
        if self.m.has_fault('short'):
            self.ee_in.s.current = 500*10/5000*self.sig_in.s.power*self.ee_in.s.voltage
            self.s.eff = 0.0
        elif self.m.has_fault('mech_break'):
            self.ee_in.s.current = 0.2*10/5000*self.sig_in.s.power*self.ee_in.s.voltage
            self.s.eff = 0.0
        else:
            self.ee_in.s.current = 10/5000*self.sig_in.s.power * \
                self.ee_in.s.voltage*min(13.0, self.wat_out.s.pressure)
            # if we wanted to enforce nominall eff state, we would include:
            # self.s.eff = 1.0

        velocity = self.sig_in.s.power*self.s.eff * \
            min(1000, self.ee_in.s.voltage)*self.wat_in.s.level
        self.wat_out.s.pressure = 10/500 * velocity/self.wat_out.s.area
        self.wat_out.s.flowrate = 0.3/500 * velocity*self.wat_out.s.area

        self.wat_in.s.assign(self.wat_out.s, 'pressure', 'flowrate')


# DEFINE MODEL OBJECT
class Pump(FunctionArchitecture):
    """
    Define the pump model as a Model.

    Models take a dictionary of parameters as input defining any veriables and
    values to use in the model.

    Note that sp is the SimParam defining the simulation. phases in this dictionary
    are queues for fault sampling which can be used by SampleApproach/FaultSample

    We can also chage dt to change the timestep, but note that this can change
    behavior.
    In this model, because every time we've entered occurs at a factor of 5, and
    there aren't any complicated controls/dynamics interactions that would need to
    be tuned, we can easily use the timestep t=1 OR t=5.
    """

    __slots__ = ()
    container_p = PumpParam
    default_sp = dict(phases=(('start', 0, 4), ('on', 5, 49), ('end', 50, 55)),
                      end_time=55.0, dt=1.0, units='hr')
    default_track = {'flows': {'wat_2': {'s': 'flowrate'},
                               'ee_1': {'s': {'current'}}}, 'i': 'all'}

    def init_architecture(self, **kwargs):
        """
        Use init_architecture to create the model architecture.

        Here add_flow() is used to instantiate a given flow object with a given type
        to a given name. Non-default values (for s, p, etc) can be passed, and we
        can also pass already-instantiated objects if desired.
        """
        self.add_flow('ee_1', Electricity)
        self.add_flow('sig_1', Signal)
        self.add_flow('wat_1', Water)
        self.add_flow('wat_2', Water)

        """
        Functions are added to the model using the add_fxn() method, which must be
        called after add_flow, and needs:
           - a unique function name
           - the class to instantiate the function with (defined above)
           - non-default values (in this case, we are passing p from the model to
             function level for move_water)
        """
        self.add_fxn('import_ee', ImportEE, 'ee_1')
        self.add_fxn('import_water', ImportWater, 'wat_1')
        self.add_fxn('import_signal', ImportSig, 'sig_1')
        self.add_fxn('move_water', MoveWat, 'ee_1', 'sig_1',
                     'wat_1', 'wat_2', p={'delay': self.p.delay})
        self.add_fxn('export_water', ExportWater, 'wat_2')

    def indicate_finished(self, time):
        """
        Indicate that the pump is finished.

        Indicators can addtionally be used to log conditions and even stop the
        simulation when these conditions are met.

        This method is optional, but helpful when the simulation is expensive and there
        are defined end conditions (e.g., reaching a destination or failing to do so).

        It returns True when the end condition is met and False otherwise. Here
        a dummy method is provided to demonstrate, in practice this would depend on
        the intended end-states of the model.
        """
        if time > self.sp.end_time:
            return True
        else:
            return False

    def indicate_on(self, time):
        """Indicate that the pump is on."""
        return self.flows['wat_1'].s.flowrate > 0

    def find_classification(self, scen, mdlhists):
        """
        Classify the simulation run/scenario.

        Propagation methods use find_classification() to classify the results based on
        the effects of a fault scenario, returning whatever metrics are desired. In this
        case, a dictionary with rate, cost, and expected cost is calculated.

        In this example, there are three costs--water, electrical, and repair costs:
        - repair costs depends on the cost of each mode, while
        - electrical and water costs depend on the lost water in the non-nominal case
        """
        # get fault costs and rates
        if 'repair' in self.p.cost:
            repcost = self.calc_repaircost()
        else:
            repcost = 0.0
        if 'water' in self.p.cost:
            lostwat = sum(mdlhists['nominal'].flows.wat_2.s.flowrate -
                          mdlhists['faulty'].flows.wat_2.s.flowrate)
            watcost = 750 * lostwat * self.sp.dt
        elif 'water_exp' in self.p.cost:
            wat = mdlhists['nominal'].flows.wat_2.s.flowrate - \
                mdlhists['faulty'].flows.wat_2.s.flowrate
            watcost = 100 * sum(np.array(accumulate(wat))**2) * self.sp.dt
        else:
            watcost = 0.0
        if 'ee' in self.p.cost:
            eespike = [spike for spike in mdlhists['faulty'].flows.ee_1.s.current -
                       mdlhists['nominal'].flows.ee_1.s.current if spike > 1.0]
            if len(eespike) > 0:
                eecost = 14 * \
                    sum(np.array(reseting_accumulate(eespike))) * self.sp.dt
            else:
                eecost = 0.0
        else:
            eecost = 0.0

        totcost = repcost + watcost + eecost

        rate = scen.rate

        life = 1e5
        expcost = rate*life*totcost
        return {'rate': rate, 'cost': totcost, 'expected_cost': expcost}


def script_show_graphs(**kwargs):
    """Show graphs of Pump structure."""
    mdl = Pump(**kwargs)
    mg = FunctionArchitectureGraph(mdl)
    mg.set_exec_order(mdl)
    fig, ax = mg.draw()

    mg = FunctionArchitectureGraph(mdl)
    fig, ax = mg.plot_high_degree_nodes()


def script_try_faults(**kwargs):
    """Try some fault scenarios."""
    mdl = Pump(**kwargs)
    endclass, mdlhist = propagate.one_fault(mdl, 'export_water', 'block', time=29,
                                            staged=True)

    endclass, mdlhist = propagate.one_fault(
        mdl, 'import_water', 'no_wat', time=29, staged=True)
    endclass, mdlhist = propagate.nominal(mdl, mdl_kwargs=dict(track='all'))
    fig, ax = mdlhist.plot_line('flows.wat_2.s.flowrate', 'i.on')
    mdl = Pump(**kwargs)

    endclass, mdlhist = propagate.one_fault(
        mdl, 'import_water', 'no_wat', time=29, staged=True)

    endclass, mdlhist = propagate.one_fault(
        mdl, 'move_water', 'mech_break', time=0, staged=False)


def script_fault_degradation_tables(**kwargs):
    """Show fault/degradation tables/plots for a given fault scenario."""
    mdl = Pump(**kwargs, track="all")
    endclass, mdlhist = propagate.one_fault(
        mdl, 'import_ee', 'no_v', time=29, staged=True)

    ks = mdl.get_roles_as_dict("fxn", "flow", flex_prefixes=True)
    deghist = mdlhist.get_degraded_hist(*ks)
    exp = deghist.get_metrics()
    deghist
    a = deghist.as_table()
    b = mdlhist.get_fault_degradation_summary(*ks)

    exp = deghist.get_metrics()
    mg = FunctionArchitectureGraph(mdl)
    mg.set_heatmap({"pump."+k: v for k, v in exp.items()})
    mg.draw()
    return a, b


def script_sample_faults(track='all', **kwargs):
    """Sample all faults from the pump."""
    mdl = Pump(track=track, **kwargs)
    from fmdtools.sim.sample import SampleApproach
    faultapp = SampleApproach(mdl)
    faultapp.add_faultdomain("testdomain", "all")
    faultapp.add_faultsample("testsample", "fault_phases", "testdomain",
                             phasemap=mdl.sp.phases)

    endclasses, mdlhists = propagate.fault_sample(mdl, faultapp)
    flat = mdlhists.flatten()

    gh = mdlhists.get_comp_groups('flows.ee_1.s.current')

    endclasses, mdlhists_staged = propagate.fault_sample(mdl, faultapp,
                                                         staged=True, track='all')

    tab = an.tabulate.result_summary_fmea(
        endclasses, mdlhists, *mdl.fxns, *mdl.flows)

    h = mdlhists.get_expected(app=faultapp, with_nominal=True)
    ec = endclasses.get_expected()

    # degsumm = h.get_summary(*mdl.fxns, *mdl.flows)

    d = h.get_degraded_hist(*mdl.flows, nomhist=mdlhists.nominal)



    c = an.tabulate.Comparison(endclasses, faultapp, default_stat=np.average,
                               metrics=['cost', 'rate', 'expected_cost'],
                               ci_metrics=['cost'])
    c.as_table()
    c.sort_by_factor("time")
    c.as_plot("cost")
    c.as_plots("cost", "rate", "expected_cost", cols=2)

    fmea = an.tabulate.FMEA(endclasses, faultapp)
    fmea.as_table()
    fmea.sort_by_metric("expected_cost")
    fmea.as_plot("expected_cost", color_factor="function")

    # test cases for multiplot legend/axis sharing
    mdlhists.plot_line("flows.ee_1.s.current", "flows.sig_1.s.power",
                       "fxns.move_water.s.eff")
    mdlhists.plot_line("flows.ee_1.s.current", "flows.sig_1.s.power",
                       "fxns.move_water.s.eff", "flows.wat_1.s.flowrate", cols=3)

    endclasses.plot_metric_dist("rate", "cost", "expected_cost")


if __name__ == "__main__":
    # import doctest
    # doctest.testmod(verbose=True)
    Pump().get_vars("flows.ee_1")
    g = Pump().create_graph()
    from fmdtools.analyze.graph.base import Graph
    from fmdtools.define.object.base import ObjectGraph
    # horizontal alignment test:
    g2 = Graph(g)
    g2.set_node_labels(subtext="nodetype", title_style={'horizontalalignment': 'left'},
                       subtext_style={'horizontalalignment': 'right'})
    g2.draw_graphviz()
    Graph(ImportEE().create_graph()).draw_graphviz()
    script_show_graphs()
    script_fault_degradation_tables()
    script_try_faults()
    script_sample_faults()
    check_model_pickleability(Pump(), try_pick=True)
    import inspect
    source = inspect.getsource(Pump)
    og = ObjectGraph(Pump(), get_source=True)
    og.set_node_labels(title='shortname', title2='classname', subtext='docs')
    og.draw_graphviz()
