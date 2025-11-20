#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Human Rover Model.

Functions:
    - Communications
    - Avionics
    - Camera/Guidance
    - Structures
    - Power

Flows:
    - Communications
    - Ground
    - Force
    - EE
    - Camera

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

from examples.rover.rover_model import Switch, Comms, Video, Control
from examples.rover.rover_model import Rover, RoverParam
from examples.rover.rover_model import Ground, Pos, EE
from examples.rover.rover_model import FaultSig, Power, Perception, Communications
from examples.rover.rover_model import PlanPath, Override, Drive
from examples.rover.rover_model import PlanPathState
from examples.rover.rover_model import Operator as BaseOperator

from fmdtools.define.block.action import Action
from fmdtools.define.architecture.action import ActionArchitecture
from fmdtools.define.container.state import State
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.mode import Mode
from fmdtools.define.flow.base import Flow

import numpy as np


class PSFParam(Parameter):
    """
    Operator performance shaping factors that stay the same in an operation.

    Fields
    ------
    fatigue : float
        Fatigue level over the course of a day.
    stress : float
        Operator stress level over the course of a day.
    """

    fatigue: np.float64 = 0.0
    stress: np.float64 = 0.0


class PSFState(State):
    """
    Operator performance shaping factors that degrade over time.

    Fields
    ------
    fatigue : float
        Fatigue level over the course of a day.
    stress : float
        Operator stress level over the course of a day.
    attention : float
        Operator attention over the course of a day.
    """

    attention: np.float64 = 10.0


class PSFs(Flow):
    """Flow of operator performance shaping factors."""

    container_s = PSFState
    container_p = PSFParam


class OperatorSignal(Flow):
    """Flow of operator path planning information."""

    container_s = PlanPathState


class HumanActionMode(Mode):
    """Generic/shared modes for human actions."""
    fault_failed_no_action = ()
    opermodes = ('nominal', )
    mode: str = 'nominal'
    exclusive = True


class GenericHumanAction(Action):
    """Shared properties of human actions."""

    container_m = HumanActionMode

    def complete(self):
        """Action is finished if not in 'no_action' mode."""
        return not self.m.in_mode('no_action', 'failed_no_action')


class Look(GenericHumanAction):
    """
    Operator looking at the state of comms signals, switches, etc.

    Works as a simple pass-through. If the user doesn't look, they can't percieve,
    project, etc.

    Example
    -------
    >>> l = Look()
    >>> l.complete()
    True
    >>> l.m.to_fault("failed_no_action")
    >>> l.complete()
    False
    """


class PerceptionMode(HumanActionMode):
    """
    Modes for failed perception.

    Modes
    -----
    failed_no_action : Mode
        Action not completed.
    not_visible : Mode
        Video not percieved.
    wrong_position : Mode
        Position not percieved.
    """

    fault_failed_no_action = ()
    fault_not_visible = ()
    fault_wrong_position = ()
    opermodes = ('nominal', 'no_action')


class Percieve(GenericHumanAction):
    """
    Operator percieving the state of comms signals, switches, etc.

    Should take in view (from look) and pass as percieved info.

    Examples
    --------
    >>> p = Percieve()
    >>> p.comms.s.on = True
    >>> p.comms.s.pos.x = 1.0
    >>> p.comms.s.video.lin_ux = 2.0
    >>> p.dynamic_behavior()
    >>> p.pos_signal.s.x
    np.float64(1.0)
    >>> p.video.s.lin_ux
    np.float64(2.0)
    """

    container_m = PerceptionMode
    flow_comms = Comms
    flow_pos_signal = Pos
    flow_video = Video
    flow_psfs = PSFs

    def dynamic_behavior(self):
        if not self.comms.s.on:
            self.m.remove_any_faults(opermode='no_action')
        elif self.comms.s.video.quality == 0.0:
            self.m.to_fault('not_visible')
        elif self.psfs.p.fatigue > 8:
            self.m.to_fault("failed_no_action")
        elif self.psfs.s.attention < 3:
            self.m.to_fault("wrong_position")
        else:
            self.m.remove_any_faults(opermode='nominal')

        if not self.m.in_mode("not_visible", 'no_action'):
            self.video.s.assign(self.comms.s.video)
        if not self.m.in_mode("wrong_position", 'no_action'):
            self.pos_signal.s.assign(self.comms.s.pos)


class Comprehend(GenericHumanAction):
    """
    Operator comprehending the input state.

    Should take in percieved info and distill as situation (moving, turning, etc)

    May fail due to fatigue >8 or stress >80.

    Examples
    --------
    >>> c = Comprehend()
    >>> c.pos_signal.s.put(ux=1.0, uy=1.0)
    >>> c.dynamic_behavior()
    >>> c.signal.s.u_self
    array([1., 1.])
    """

    container_m = HumanActionMode
    flow_signal = OperatorSignal
    flow_pos_signal = Pos
    flow_video = Video
    flow_psfs = PSFs

    def dynamic_behavior(self):
        if self.psfs.p.fatigue > 8 or self.psfs.p.stress > 80:
            self.m.to_fault('failed_no_action')
        if self.m.in_mode('nominal'):
            self.signal.s.set_positions(self.pos_signal, self.video)


class ProjectMode(HumanActionMode):
    """
    Projection failure modes.

    Modes
    -----
    failed_turn_left : Mode
        Spontaneous left turn.
    failed_turn_right : Mode
        Spontaneous right turn.
    failed_slow : Mode
        Velocity adjusted slow.
    failed_fast : Mode
        Velocity adjusted to go fast.
    failed_no_action : Mode
        No projection performed.
    """

    fault_failed_turn_left = ()
    fault_failed_turn_right = ()
    fault_failed_slow = ()
    fault_failed_fast = ()
    fault_failed_no_action = ()


class Project(GenericHumanAction):
    """
    Operator projecting out how control actions might affect input state.

    Determines rdiff in signal.

    Examples
    --------
    >>> p = Project()
    >>> p.signal.s.put(u_self=(0.0, 1.0), u_lin=(1.0, 1.0))
    >>> p.dynamic_behavior()
    >>> p.signal.s.rdiff
    np.float64(-0.7854052343902613)
    """

    container_m = ProjectMode
    flow_signal = OperatorSignal
    flow_psfs = PSFs

    def dynamic_behavior(self):
        if self.m.in_mode('nominal'):
            self.signal.s.set_turn()
        elif self.m.in_mode('failed_turn_left'):
            self.signal.s.rdiff = -0.5
        elif self.m.in_mode('failed_turn_right'):
            self.signal.s.rdiff = 0.5
        elif self.m.in_mode('failed_slow'):
            self.signal.s.set_turn()
            self.signal.s.vel_adj = 0.2
        elif self.m.in_mode('failed_fast'):
            self.signal.s.set_turn()
            self.signal.s.vel_adj = 4.0
        # increment attention - degrades over the course of an operation if the
        # driving isn't "interesting" (aka there are no turns)
        # turns reset the attention to 10 if attention is above threshold
        if abs(self.signal.s.rdiff) < 0.01:
            if self.psfs.p.fatigue < 5:
                self.psfs.s.inc(attention=(-0.2, 0.0))
            else:
                self.psfs.s.inc(attention=(-0.4, 0.0))
        elif self.psfs.s.attention > 3:
            self.psfs.s.attention = 10.0


class DecideMode(Mode):
    """
    Decide failure modes.

    Modes
    -----
    failed_no_action : Mode
        No decision performed.
    failed_continue : Mode
        Decision not performed, but continues with actions.
    """

    fault_failed_no_action = ()
    fault_failed_continue = ()


class Decide(GenericHumanAction):
    """
    Operator deciding how to control actions based on projection of input state.

    Examples
    --------
    >>> d = Decide()
    >>> d.signal.s.rdiff = 1.0
    >>> d.dynamic_behavior()
    >>> d.control.s
    ControlState(rpower=2.0, lpower=0.0)
    """

    container_m = ProjectMode
    flow_signal = OperatorSignal
    flow_control = Control

    def dynamic_behavior(self):
        # if no action, stops here. If continue, doesn't pass new control info.
        if self.m.in_mode('nominal'):
            self.signal.s.set_control(self.control)


class Reach(GenericHumanAction):
    """
    Operator reaching for controls to operate.

    (passthrough)
    """

    container_m = HumanActionMode


class PressMode(Mode):
    """
    Operator pressing modes.

    Modes
    -----
    failed_left : Mode
        Operator unexpectedly presses to turn left.
    failed_right : Mode
        Operator unexpectedly presses to turn right.
    no_press : Mode
        Operator unexpectedly disengages controls (making power zero).
    """

    fault_failed_left = ()
    fault_failed_right = ()
    fault_no_press = ()


class Press(GenericHumanAction):
    """
    Operator presses the button/toggle for the controls.

    Examples
    --------
    >>> p = Press()
    >>> p.control.s.put(lpower=1.0, rpower=0.0)
    >>> p.dynamic_behavior()
    >>> p.comms.s.ctl
    ControlState(rpower=0.0, lpower=1.0)
    """

    container_m = PressMode
    flow_control = Control
    flow_comms = Comms

    def dynamic_behavior(self):
        if self.m.in_mode('nominal'):
            self.comms.s.ctl.assign(self.control.s)
        elif self.m.in_mode('failed_left'):
            self.comms.s.ctl.put(lpower=1.0, rpower=0.0)
        elif self.m.in_mode('failed_right'):
            self.comms.s.ctl.put(lpower=0.0, rpower=1.0)
        elif self.m.in_mode('no_press'):
            self.comms.s.ctl.put(lpower=0.0, rpower=0.0)


class HumanActions(ActionArchitecture):
    """
    Overall ASG for human operator driving the rover.

    Examples
    --------
    >>> ha = HumanActions()
    >>> ha.flows['comms'].s.on = True
    >>> ha.active_actions
    {'look'}
    >>> ha(proptype='dynamic', time=1.0)
    >>> ha.active_actions
    {'press'}
    """

    initial_action = "look"
    per_timestep = True
    container_p = PSFParam

    def init_architecture(self, **kwargs):
        self.add_flow('psfs', PSFs, p=self.p)
        self.add_flow('signal', OperatorSignal)
        self.add_flow('pos_signal', Pos)
        self.add_flow('switch', Switch)
        self.add_flow('control', Control)
        self.add_flow('comms', Comms)
        self.add_flow('video', Video)
        # flow for workload "local PSF"?

        self.add_act('look', Look)
        self.add_act('percieve', Percieve, 'comms', 'pos_signal', 'video', 'psfs')
        self.add_act('comprehend', Comprehend, 'pos_signal', 'video', 'signal', 'psfs')
        self.add_act('project', Project, 'signal', 'psfs')
        self.add_act('decide', Decide, 'signal', 'control')
        self.add_act('reach', Reach)
        self.add_act('press', Press, 'comms', 'control')

        self.add_cond('look', 'percieve', condition=self.acts['look'].complete)
        self.add_cond('percieve', 'comprehend',
                      condition=self.acts['percieve'].complete)
        self.add_cond('comprehend', 'project',
                      condition=self.acts['comprehend'].complete)
        self.add_cond('project', 'decide', condition=self.acts['project'].complete)
        self.add_cond('decide', 'reach', condition=self.acts['decide'].complete)
        self.add_cond('reach', 'press', condition=self.acts['reach'].complete)


class Operator(BaseOperator):
    """Overall function for operator (adds ASG to flipping switch)."""

    flow_switch = Switch
    flow_control = Control
    flow_pos_signal = Pos
    flow_comms = Comms
    flow_video = Video
    flow_psfs = PSFs
    flow_ground = Ground
    container_p = PSFParam
    arch_aa = HumanActions
    container_m = Mode

    def dynamic_behavior(self):
        self.set_power()
        if self.ground.at_end(self.pos_signal.s):
            self.switch.s.power = False
            self.comms.s.ctl.put(rpower=0, lpower=0)


class RoverHumanParam(RoverParam):
    """Human rover parameter (extends to add PSF parameter)."""

    psfs: PSFParam = PSFParam()


class RoverHuman(Rover):
    """Overall human model for the rover."""

    container_p = RoverHumanParam

    def init_architecture(self, **kwargs):
        """Initialize the functional architecture."""
        self.add_flow("ground", Ground, p=self.p.ground)
        self.add_flow("psfs", PSFs, p=self.p.psfs)
        self.add_flow("pos_signal", Pos)
        self.add_flow('pos', Pos)
        self.add_flow("ee_12", EE)
        self.add_flow("ee_5", EE)
        self.add_flow("ee_15", EE)
        self.add_flow("video", Video)
        self.add_flow("auto_control", Control)
        self.add_flow("motor_control", Control)
        self.add_flow("switch", Switch)
        self.add_flow("comms", Comms, s={'active': True})
        self.add_flow("fault_sig", FaultSig)

        self.add_fxn("power", Power, "ee_15", "ee_5", "ee_12", "switch")
        self.add_fxn("perception", Perception, "ground", 'pos', 'pos_signal',
                     "ee_12", "video")
        self.add_fxn("communications", Communications, "comms", "ee_12", "pos_signal",
                     "video")

        self.add_fxn("operator", Operator, "switch", "comms", "ground", "psfs", p=self.p.psfs)
        self.add_fxn("plan_path", PlanPath, "video", "pos_signal", "ground",
                     "auto_control", "fault_sig", p=self.p.correction)
        self.add_fxn("override", Override, "comms", "ee_5", "motor_control",
                     "auto_control")

        self.add_fxn("drive", Drive, "ground", 'pos', "ee_15", "motor_control",
                     "fault_sig", m=self.p.drive_modes, p=self.p.degradation)


asg_pos = {'look': [-0.9, 0.88], 'percieve': [-0.68, 0.62], 'comms': [-0.66, -0.68],
           'pos_signal': [-0.45, 0.91], 'video': [0.02, 0.7], 'psfs': [-0.55, 0.05],
           'comprehend': [-0.46, 0.4], 'signal': [0.46, 0.44], 'project': [-0.24, 0.1],
           'decide': [-0.01, -0.15], 'control': [0.81, -0.13], 'reach': [0.36, -0.44],
           'press': [0.9, -0.69]}


if __name__ == "__main__":
    mdl = RoverHuman()
    import doctest
    doctest.testmod(verbose=True)

    from fmdtools.define.architecture.action import ActionArchitectureGraph
    hum = HumanActions()
    ag = ActionArchitectureGraph(hum)
    ag.set_pos(**asg_pos)
    ag.draw()
    ag.draw_graphviz(layout='dot')

    from examples.rover.rover_model import plot_map
    from fmdtools.sim import propagate as prop
    endresults, mdlhist = prop.nominal(mdl)
    ec1, hist = prop.nominal(mdl)
    fig, ax = hist.plot_trajectories('flows.pos.s.x', 'flows.pos.s.y',
                                     time_groups=['nominal'])
    fig, ax = plot_map(mdl, mdlhist)
    # ax.set_xlim(0,5)
    # ax.set_ylim(-2,2)
    fig.show()
    import fmdtools.analyze as an
    from fmdtools.sim.sample import FaultDomain, FaultSample
    pm = an.phases.from_hist(hist)
    fd = FaultDomain(mdl)
    fd.add_all_fxn_modes('operator')
    fd
    fs = FaultSample(fd, phasemap = pm['override'])
    fs.add_fault_phases('override')
    fs
    ecs, hists = prop.fault_sample(mdl, fs)
    tab = an.tabulate.result_summary_fmea(ecs, hists,
                                          *mdl.fxns,
                                          metrics = ["in_bound", "at_finish", "end_dist", "faults", "classification", "end_x", "end_y"])

