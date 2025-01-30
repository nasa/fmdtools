#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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

from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.block.action import Action
from fmdtools.define.architecture.action import ActionArchitecture
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow


class OutcomeStates(State):
    """State tracking the number of actions and perceptions performed."""

    num_perceptions: int = 0
    num_actions: int = 0


class Outcome(Flow):

    __slots__ = ()
    container_s = OutcomeStates


class HazardState(State):
    """Whether a hazard is present, percieved, or mitigated by the human."""

    present: bool = False
    percieved: bool = False
    mitigated: bool = False


class Hazard(Flow):

    __slots__ = ()
    container_s = HazardState


class ActionMode(Mode):
    fm_args = ('failed', 'unable')
    mode: str = 'nominal'
    exclusive = True


class Perceive(Action):
    """A user's perception abilities/behaviors for percieving the hazard."""

    __slots__ = ('hazard', 'outcome')
    container_m = ActionMode
    flow_hazard = Hazard
    flow_outcome = Outcome

    def behavior(self, time):
        if not self.m.in_mode('failed', 'unable'):
            self.hazard.s.percieved = self.hazard.s.present
            self.outcome.s.num_perceptions += self.hazard.s.percieved
        else:
            self.hazard.s.percieved = False
            self.m.remove_fault('failed', 'nom')

    def percieved(self):
        return self.hazard.s.percieved


class Act(Action):
    """User actions to mitigate the hazard."""

    __slots__ = ('hazard', 'outcome')
    container_m = ActionMode
    flow_hazard = Hazard
    flow_outcome = Outcome

    def behavior(self, time):
        if not self.m.in_mode('failed', 'unable'):
            self.outcome.s.num_actions += 1
            self.hazard.s.mitigated = True
        elif self.m.in_mode('failed'):
            self.hazard.s.mitigated = False
            self.m.remove_fault('failed', 'nom')
        else:
            self.hazard.s.mitigated = False

    def acted(self):
        return not self.m.in_mode('failed')


class Done(Action):
    """User state after performing the action."""

    __slots__ = ('hazard')
    flow_hazard = Hazard

    def behavior(self, time):
        if not self.hazard.s.present:
            self.hazard.s.mitigated = False

    def ready(self):
        return not self.hazard.s.present


class Human(ActionArchitecture):
    """Overall human action sequence graph specifying what the user will do when."""
    initial_action = "perceive"

    def init_architecture(self, *args, **kwargs):
        # flows from external fxn/model can be defined here,
        self.add_flow("hazard", Hazard)
        # along with flows internal to the ASG class
        self.add_flow("outcome", fclass=Outcome)

        self.add_act("perceive", Perceive, "outcome", "hazard")
        self.add_act("act", Act, "outcome", "hazard")
        self.add_act("done", Done, "hazard")

        self.add_cond("perceive", "act", "percieved", self.acts['perceive'].percieved)
        self.add_cond("act", "done", "acted", self.acts['act'].acted)
        self.add_cond("done", "perceive", "ready", self.acts['done'].ready)


class DetectHazard(Function):
    """Function containing the human."""

    __slots__ = ('hazard')
    container_m = Mode
    arch_aa = Human
    flow_hazard = Hazard


class ProduceHazard(Function):
    """Function producing Hazards."""

    __slots__ = ('hazard',)
    flow_hazard = Hazard

    def dynamic_behavior(self, time):
        if not time % 4:
            self.hazard.s.present = True
        else:
            self.hazard.s.present = False


class PassStates(State):
    """Whether or not the hazard is ultimately passed or mitigated."""

    hazards_mitigated: int = 0
    hazards_propagated: int = 0


class PassHazard(Function):
    """Accumulates total hazards/mitigations."""

    __slots__ = ('hazard',)
    container_s = PassStates
    flow_hazard = Hazard

    def dynamic_behavior(self, time):
        if self.hazard.s.present and self.hazard.s.mitigated:
            self.s.hazards_mitigated += 1
        elif self.hazard.s.present and not self.hazard.s.mitigated:
            self.s.hazards_propagated += 1


class HazardModel(FunctionArchitecture):
    """Overall model of the human in context."""

    __slots__ = ()
    default_sp = dict(end_time=60, dt=1.0)

    def init_architecture(self, **kwargs):
        self.add_flow("hazard", Hazard)
        self.add_fxn("produce_hazard", ProduceHazard, 'hazard')
        self.add_fxn("detect_hazard", DetectHazard, 'hazard')
        self.add_fxn("pass_hazard", PassHazard, 'hazard')


if __name__ == '__main__':
    import fmdtools.sim.propagate as prop
    mdl = HazardModel()
    result_fault, mdlhist_fault = prop.one_fault(mdl, 'detect_hazard',
                                                 'act_unable', time=4,
                                                 desired_result='graph')

    result_fault.graph.draw()
    ex_fxn = DetectHazard('detect_hazard')
    result_indiv, hist_indiv = prop.nominal(ex_fxn,
                                            disturbances={5:{'aa.flows.hazard.s.present':True}})
    fig, axs = hist_indiv.plot_line('aa.flows.hazard.s.present',
                                    'aa.flows.hazard.s.percieved',
                                    'aa.flows.hazard.s.mitigated', figsize=(10,5))

    result_fault, mdlhist_fault = prop.one_fault(mdl, 'detect_hazard',
                                                 'perceive_failed', time=4,
                                                 desired_result='graph.fxns.detect_hazard.aa')
    result_fault.graph.fxns.detect_hazard.aa.draw_graphviz(layout='dot')