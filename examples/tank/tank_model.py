#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamical implementation of a human-operated tank system.

This tanks system is helpful for showing how fmdtools can be used to model human errors.

The functions of the system are:
    - ImportLiquid (Inlet Valve)
    - GuideLiquid (Inlet Pipe)
    - StoreLiquid (Tank)
    - GuideLiquid (Outlet Pipe)
    - ExportLiquid (Outlet Valve)
The Tank stores a set amount of water, the level of which is controlled by
inlet and outlet valves. In this model we (will) use an action sequence graph
to model the human interactions with the system.

For more information on this system, see:

Irshad, L., Ahmed, S., Demirel, O., & Tumer, I. Y. (2018). Identification of
human errors during early design stage functional failure analysis. In ASME
2018 International Design Engineering Technical Conferences and Computers and
Information in Engineering Conference. American Society of Mechanical Engineers
Digital Collection.

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
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode
from fmdtools.define.flow.base import Flow
from fmdtools.define.block.function import Function
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.action import Action
from fmdtools.define.architecture.action import ActionArchitecture


class WatState(State):
    effort: float = 1.0
    rate: float = 1.0


class Liquid(Flow):
    __slots__ = ()
    container_s = WatState


class SigState(State):
    indicator: int = 1
    action: int = 0


class Signal(Flow):
    __slots__ = ()
    container_s = SigState


class TransportLiquidState(State):
    amt_open: int = 1


class TransportLiquidMode(Mode):
    fm_args = {'stuck': (1e-5,)}
    phases = {'na': 1.0}
    units = 'hr'


class ImportLiquid(Function):

    __slots__ = ('sig', 'watout')
    container_s = TransportLiquidState
    container_m = TransportLiquidMode
    flow_sig = Signal
    flow_watout = Liquid
    flownames = {'wat_in_1': 'watout', 'valve1_sig': 'sig'}

    def static_behavior(self, time):
        if not self.m.has_fault('stuck'):
            if self.sig.s.action >= 2:
                self.s.amt_open = 2
            elif self.sig.s.action == 1:
                self.s.amt_open = 1
            elif self.sig.s.action == -1:
                self.s.amt_open = 0
        self.watout.s.effort = float(self.s.amt_open)
        self.sig.s.indicator = self.s.amt_open


class ExportLiquid(Function):
    __slots__ = ('sig', 'watin')
    container_s = TransportLiquidState
    container_m = TransportLiquidMode
    flow_sig = Signal
    flow_watin = Liquid
    flownames = {'wat_out_2': 'watin', 'valve2_sig': 'sig'}

    def static_behavior(self, time):
        if not self.m.has_fault('stuck'):
            if self.sig.s.action >= 1:
                self.s.amt_open = 1
            elif self.sig.s.action == -1:
                self.s.amt_open = 0
        self.watin.s.rate = self.s.amt_open*self.watin.s.effort
        self.sig.s.indicator = self.s.amt_open


class GuideLiquidMode(Mode):
    fm_args = {'leak': (1e-5,), 'clogged': (1e-5,)}
    phases = {'na': 1.0}


class GuideLiquid(Function):
    __slots__ = ('watin', 'watout')
    flow_watin = Liquid
    flow_watout = Liquid
    container_m = GuideLiquidMode

    def static_behavior(self, time):
        if self.m.has_fault('clogged'):
            self.watin.s.put(rate=0.0)
            self.watout.s.put(effort=0.0)
        elif self.m.has_fault('leak'):
            self.watout.s.effort = self.watin.s.effort - 1.0
            self.watin.s.rate = self.watout.s.rate - 1.0
        else:
            self.watout.s.effort = self.watin.s.effort
            self.watin.s.rate = self.watout.s.rate


class GuideLiquidIn(GuideLiquid):
    __slots__ = ()
    flownames = {'wat_in_1': 'watin', 'wat_in_2': 'watout'}


class GuideLiquidOut(GuideLiquid):
    __slots__ = ()
    flownames = {'wat_out_1': 'watin', 'wat_out_2': 'watout'}


class StoreLiquidState(State):
    level: float = 10.0
    net_flow: float = 0.0


class StoreLiquidMode(Mode):
    fm_args = {'leak': (1e-5, 0, {'na': 1.0})}


class StoreLiquid(Function):
    __slots__ = ('watin', 'watout', 'sig')
    container_s = StoreLiquidState
    container_m = StoreLiquidMode
    flow_watin = Liquid
    flow_watout = Liquid
    flow_sig = Signal
    flownames = {'wat_in_2': 'watin', 'wat_out_1': 'watout', 'tank_sig': 'sig'}

    def static_behavior(self, time):
        if self.s.level >= 20.0:
            self.watin.s.rate = 0.0 * self.watin.s.effort
            self.watout.s.effort = 2.0 * self.watin.s.effort
            self.s.level = 20.0
        elif self.s.level <= 0.0:
            self.watout.s.effort = 0.0
            self.watin.s.rate = self.watin.s.effort
        else:
            self.watin.s.put(rate=self.watin.s.effort)
        if self.s.level > 12:
            self.sig.s.indicator = -1
        elif self.s.level < 8:
            self.sig.s.indicator = 1
        else:
            self.sig.s.indicator = 0

        if self.m.has_fault('leak'):
            self.s.net_flow = self.watin.s.rate - self.watout.s.rate - 1.0
        else:
            self.s.net_flow = self.watin.s.rate - self.watout.s.rate

    def dynamic_behavior(self, time):
        self.s.inc(level=self.s.net_flow*self.t.dt)
        self.s.limit(level=(0.0, 25))
        # self.s.level = self.s.level + self.s.net_flow*self.dt


class HumanParam(Parameter):
    reacttime: int = 2


class HumanASG(ActionArchitecture):
    initial_action = "look"
    container_p = HumanParam

    def init_architecture(self, **kwargs):
        self.add_flow("tank_sig", Signal)
        self.add_flow("valve1_sig", Signal)
        self.add_flow("valve2_sig", Signal)
        self.add_flow("detect_sig", Signal)

        self.add_act('look', Look)
        self.add_act('detect', Detect, 'detect_sig', 'tank_sig',
                     duration=self.p.reacttime)
        self.add_act('reach', Reach)
        self.add_act('grasp', Grasp)
        self.add_act('turn', Turn, 'detect_sig',
                     'valve1_sig', 'valve2_sig', duration=1.0)

        self.add_cond('look', 'detect', 'looked', condition=self.acts['look'].looked)
        self.add_cond('detect', 'reach', 'detected',
                      condition=self.acts['detect'].detected)
        self.add_cond('reach', 'grasp', 'reached',
                      condition=self.acts['reach'].reached)
        self.add_cond('grasp', 'turn', 'grasped',
                      condition=self.acts['grasp'].grasped)
        self.add_cond('turn', 'look', 'done', condition=self.acts['turn'].turned)


class HumanActions(Function):
    __slots__ = ('valve1_sig', 'tank_sig', 'valve2_sig')
    container_p = HumanParam
    container_m = Mode
    arch_aa = HumanASG
    flow_valve1_sig = Signal
    flow_tank_sig = Signal
    flow_valve2_sig = Signal

    def dynamic_behavior(self, time):
        """
        Some testing code for ASG behavior and copying, etc. Raises exceptions when
        flows aren't copied correctly
        """
        if self.aa.acts['look'].looked.__self__.__hash__() != self.aa.conds['looked'].__self__.__hash__():
            raise Exception("Condition not passed")
        if self.aa.flows['valve1_sig'].__hash__() != self.valve1_sig.__hash__():
            raise Exception("Invalid connection hash in asg.flows")
        if self.aa.acts['detect'].tank_sig.__hash__() != self.tank_sig.__hash__():
            raise Exception("Invalid connection hash in asg.flows")
        if self.aa.flows['detect_sig'].__hash__() != self.aa.acts['detect'].detect_sig.__hash__():
            raise Exception("Invalid connection hash in asg.flows")
        if self.aa.flows['valve2_sig'].__hash__() != self.valve2_sig.__hash__():
            raise Exception("Invalid connection hash in asg.flows")
        if self.aa.acts['turn'].valve2_sig.__hash__() != self.valve2_sig.__hash__():
            raise Exception("Invalid connection hash")

        if not self.aa.acts['turn'].valve2_sig.s.action == self.valve2_sig.s.action:
            raise Exception("invalid connection: valve2_sig")

        if not self.aa.acts['turn'].valve1_sig.s.action == self.valve1_sig.s.action:
            raise Exception("invalid connection: valve1_sig")


class LookMode(Mode):
    failrate: float
    fm_args = {'not_visible': (1, )}
    phases = {'na': 1.0}
    # using lists as inputs leaves the EPCs unlabeled
    he_args = (0.02, [[4, 0.1], [4, 0.6], [1.1, 0.9]])


class Look(Action):
    __slots__ = ()
    container_m = LookMode

    def looked(self):
        return not self.m.has_fault('not_visible')


class DetectMode(Mode):
    failrate: float
    fm_args = ('not_detected', 'false_high', 'false_low')
    phases = {'na': 1.0}
    he_args = (0.03, {2: [11, 0.1], 10: [10, 0.2], 13: [4, 0],
                      14: [4, 0.1], 17: [3, 0], 34: [1.1, 0.6]})


class Detect(Action):
    __slots__ = ('tank_sig', 'detect_sig')
    container_m = DetectMode
    flow_detect_sig = Signal
    flow_tank_sig = Signal

    def behavior(self, time):
        if self.m.has_fault('not_detected'):
            self.detect_sig.s.put(indicator=0, action=0)
        elif self.m.has_fault('false_high'):
            self.detect_sig.s.put(indicator=1, action=-1)
        elif self.m.has_fault('false_low'):
            self.detect_sig.s.put(indicator=-1, action=2)
        else:
            self.detect_sig.s.indicator = self.tank_sig.s.indicator
            if self.detect_sig.s.indicator >= 1:
                self.detect_sig.s.action = 2
            elif self.detect_sig.s.indicator <= -1:
                self.detect_sig.s.action = -1
            else:
                self.detect_sig.s.action = 1

    def detected(self):
        return self.detect_sig.s.indicator


class ReachMode(Mode):
    failrate: float
    fm_args = {'unable': (0.5, )}
    phases = {'na': 1.0}
    he_args = (0.09, {2: [11, 0.1], 10: [10, 0.0],
                      13: [4, 0], 14: [4, 0.1],
                      17: [3, 0], 34: [1.1, 0]})


class Reach(Action):
    __slots__ = ()
    container_m = ReachMode

    def reached(self):
        return not self.m.has_fault('unable')


class GraspMode(Mode):
    failrate: float
    fm_args = {'cannot': (1, )}
    phases = {'na': 1.0}
    failrate = 0.02


class Grasp(Action):
    __slots__ = ()
    container_m = GraspMode

    def grasped(self):
        return not self.m.has_fault('cannot')


class TurnMode(Mode):
    failrate: float
    fm_args = {'cannot': (1,),
                 'wrong_valve': (0.5,)}
    phases = {'na': 1.0}
    he_args = (0.009, {2: [11, 0.4], 10: [10, 0.2],
                       13: [4, 0], 14: [4, 0],
                       17: [3, 0.6], 34: [1.1, 0]})


class Turn(Action):
    __slots__ = ('detect_sig', 'valve1_sig', 'valve2_sig')
    container_m = TurnMode
    flow_detect_sig = Signal
    flow_valve1_sig = Signal
    flow_valve2_sig = Signal

    def behavior(self, time):
        if self.m.has_fault('cannot'):
            turned = 0
        else:
            turned = 1
        if turned and self.m.has_fault('wrong_valve'):
            self.valve2_sig.s.assign(self.detect_sig.s, 'action')
        elif turned:
            self.valve1_sig.s.assign(self.detect_sig.s, 'action')

    def turned(self):
        return not self.m.has_fault('cannot')


class TankParam(Parameter, readonly=True):
    reacttime: int = 2
    store_tstep: float = 1.0


class Tank(FunctionArchitecture):
    __slots__ = ()
    container_p = TankParam
    default_sp = dict(phases=(('na', 0, 0), ('operation', 1, 20)),
                      end_time=20, units='min')
    default_track = {'fxns': {'store_water': {'s': 'level'}}}

    def init_architecture(self, **kwargs):
        self.add_flow('wat_in_1', Liquid)
        self.add_flow('wat_in_2', Liquid)
        self.add_flow('wat_out_1', Liquid)
        self.add_flow('wat_out_2', Liquid)
        self.add_flow('valve1_sig', Signal, s={'indicator': 1, 'action': 0})
        self.add_flow('tank_sig', Signal, s={'indicator': 0, 'action': 0})
        self.add_flow('valve2_sig', Signal, s={'indicator': 1, 'action': 0})

        self.add_fxn('import_water', ImportLiquid, 'wat_in_1', 'valve1_sig')
        self.add_fxn('guide_water_in', GuideLiquidIn, 'wat_in_1', 'wat_in_2')
        self.add_fxn('store_water', StoreLiquid, 'wat_in_2', 'wat_out_1', 'tank_sig')
        self.add_fxn('guide_water_out', GuideLiquidOut, 'wat_out_1', 'wat_out_2')
        self.add_fxn('export_water', ExportLiquid, 'wat_out_2', 'valve2_sig')
        self.add_fxn('human', HumanActions, 'valve1_sig', 'tank_sig',
                     'valve2_sig', aa={'reacttime': self.p.reacttime})

    def find_classification(self, scen, hist):
        # here we define failure in terms of the water level getting too low or too high
        if any(self.h.fxns.store_water.s.level >= 20):
            totcost = 1000000
        elif any(self.h.fxns.store_water.s.level <= 0):
            totcost = 1000000
        else:
            totcost = 0
        rate = scen.rate
        life = 1e5
        return {'rate': rate, 'cost': totcost, 'expected_cost': rate*life*totcost}


if __name__ == '__main__':
    import fmdtools.sim.propagate as propagate
    from fmdtools.sim.sample import FaultDomain, FaultSample

    mdl = Tank(track='all')

    endclass, mdlhist = propagate.one_fault(mdl, 'human', 'look_not_visible', time=2,
                                            staged=True)

    from fmdtools.define.architecture.function import FunctionArchitectureGraph
    mg = FunctionArchitectureGraph(mdl)
    fig, ax = mg.draw_from(10, mdlhist)

    # nominal run
    endresults, mdlhist = propagate.nominal(mdl, desired_result=['endclass', 'graph'])
    mdlhist.plot_line("fxns.store_water.s.level")
    endresults.graph.draw()

    # faulty run
    endres, mdlhist = propagate.one_fault(
        mdl, 'store_water', 'leak', time=2, desired_result='graph')
    mdlhist.plot_line("fxns.store_water.s.level",
                      title='Leak Response', time_slice=2)
    endres.graph.draw(title="leak response at time=end")

    resgraph, mdlhist = propagate.one_fault(
        mdl, 'human', 'detect_false_high', time=2, desired_result='graph')

    mdlhist.plot_line("fxns.store_water.s.level",
                      title='detect_false_high', time_slice=2)
    resgraph.graph.draw(title='detect_false_high, t=2')

    resgraph, mdlhist = propagate.one_fault(
        mdl, 'human', 'turn_wrong_valve', time=2, desired_result='graph')

    mdlhist.plot_line("fxns.store_water.s.level",
                      title='turn_wrong_valve', time_slice=2)
    resgraph.graph.draw(title='turn_wrong_valve, t=2')

    mdl = Tank(p=TankParam(reacttime=2), sp=dict(dt=1.0))
    resgraph, mdlhist = propagate.one_fault(
        mdl, 'store_water', 'leak', time=3, desired_result='graph')
    mdlhist.plot_line("fxns.store_water.s.level",
                      title='Leak Response', time_slice=2)
    resgraph.graph.draw(title='turn_wrong_valve, t=end')

    # run all faults - note: all faults get caught!
    endclasses, hist = propagate.single_faults(mdl)

    mdl = Tank(p=TankParam(reacttime=2), sp=dict(dt=1.0))
    fd = FaultDomain(mdl)
    fd.add_all()
    fs = FaultSample(fd)
    fs.add_fault_times((0, 5, 10, 15, 20))
    endclasses, hist = propagate.fault_sample(mdl, fs)

    mdl.fxns['human'].t.dt = 2.0
    mg = mdl.as_modelgraph()
    mg.set_exec_order(mdl)
    mg.draw()

    ag = mdl.fxns['human'].aa.as_modelgraph()
    ag.draw()
    ag.draw_graphviz(layout='dot')
