# -*- coding: utf-8 -*-
"""
Human Rover Model

@authors: mmohame2 and dhulse


NOTE: Model not yet adapted but preserved here for historical reasons.
See: RAD-245

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
"""
from fmdtools.define.block.action import Action
from fmdtools.define.architecture.action import ActionArchitecture
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode
from fmdtools.define.flow.base import Flow

from examples.rover.rover_model import Switch, Comms, Video, Control
from examples.rover.rover_model import Rover, RoverParam
from examples.rover.rover_model import Ground, Pos, EE
from examples.rover.rover_model import FaultSig, Power, Perception, Communications
from examples.rover.rover_model import PlanPath, Override, Drive
from examples.rover.rover_model import PlanPathState


class OperatorSignal(Flow):

    __slots__ = ()
    container_s = PlanPathState


class HumanActionMode(Mode):
    """Generic/shared modes for human actions."""
    fm_args = ('failed_no_action', )
    opermodes = ('nominal', )
    mode: str = 'nominal'
    exclusive = True


class GenericHumanAction(object):
    """Shared properties of human actions."""

    __slots__ = ()
    container_m = HumanActionMode

    def complete(self):
        return not self.m.in_mode('no_action', 'failed_no_action')


class Look(Action, GenericHumanAction):
    """
    Operator looking at the state of comms signals, switches, etc.

    Works as a simple pass-through. If the user doesn't look, they can't percieve,
    project, etc.
    """

    __slots__ = ()


class PerceptionMode(HumanActionMode):
    fm_args = ("failed_no_action", "not_visible", "wrong_position")


class Percieve(Action, GenericHumanAction):
    """
    Operator percieving the state of comms signals, switches, etc.

    Should take in view (from look) and pass as percieved info.

    (passthrough?)
    """

    __slots__ = ('comms', 'pos_signal', 'video')
    container_m = PerceptionMode
    flow_comms = Comms
    flow_pos_signal = Pos
    flow_video = Video

    def behavior(self, t):
        if self.comms.video.quality == 0.0:
            self.set_mode('not_visible')
        if not self.m.in_mode("not_visible"):
            self.video.s.assign(self.comms.video)
        if not self.m.in_mode("wrong_position"):
            self.pos_signal.s.assign(self.comms.pos)


class Comprehend(Action, GenericHumanAction):
    """
    Operator comprehending the input state.

    Should take in percieved info and distill as situation (moving, turning, etc)

    May fail due to stress >8 or stress >80.
    (passthrough?)
    """

    __slots__ = ('signal', 'pos_signal', 'video')
    container_m = HumanActionMode
    flow_signal = OperatorSignal
    flow_pos_signal = Pos
    flow_video = Video

    def behavior(self, t):
        if self.m.in_mode('nominal'):
            self.signal.s.set_positions(self.pos_signal, self.video)


class ProjectMode(HumanActionMode):
    fm_args = ('failed_turn_left', 'failed_turn_right', 'failed_slow', 'failed_fast',
               'failed_no_action')


class Project(Action, GenericHumanAction):
    """
    Operator projecting out how control actions might affect input state.

    May fail to project turns or power?

    Causes and uses workload.
    (Inherit speed/projection PlanPath behavior here)
    """

    __slots__ = ('signal',)
    container_m = ProjectMode
    flow_signal = OperatorSignal

    def behavior(self, t):
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


class DecideMode(Mode):
    fm_args = ('failed_no_action', 'failed_continue')


class Decide(Action, GenericHumanAction):
    """
    Operator deciding how to control actions based on projection of input state.
    (Inherit motor control signal behavior here)
    """

    __slots__ = ('signal', 'control')
    container_m = ProjectMode
    flow_signal = OperatorSignal
    flow_control = Control

    def behavior(self, t):
        # if no action, stops here. If continue, doesn't pass new control info.
        if self.m.in_mode('nominal'):
            self.signal.set_control(self.control)


class Reach(Action, GenericHumanAction):
    """
    Operator reaching for controls to operate.

    (passthrough)
    """

    __slots__ = ()
    container_m = HumanActionMode


class PressMode(Mode):
    fm_args = ('failed_left', 'failed_right', 'no_press')

class Press(Action, GenericHumanAction):
    """
    Operator presses the button/toggle for the controls.

    (passthrough)
    """

    __slots__ = ('control', 'comms')
    container_m = PressMode
    flow_control = Control
    flow_comms = Comms

    def behavior(self, t):
        if self.m.in_mode('nominal'):
            self.comms.s.ctl.assign(self.control.s)
        elif self.m.in_mode('failed_left'):
            self.comms.s.ctl.put(lpower=1.0, rpower=0.0)
        elif self.m.in_mode('failed_right'):
            self.comms.s.ctl.put(lpower=0.0, rpower=1.0)
        elif self.m.in_mode('no_press'):
            self.comms.s.ctl.put(lpower=0.0, rpower=0.0)


class HumanActions(ActionArchitecture):

    __slots__ = ()
    initial_action = "look"

    def init_architecture(self, **kwargs):
        self.add_flow('signal', OperatorSignal)
        self.add_flow('pos_signal', Pos)
        self.add_flow('switch', Switch)
        self.add_flow('control', Control)
        self.add_flow('comms', Comms)
        self.add_flow('video', Video)
        # flow for workload "local PSF"?

        self.add_act('look', Look)
        self.add_act('percieve', Percieve, 'comms', 'pos_signal', 'video')
        self.add_act('comprehend', Comprehend, 'pos_signal', 'video', 'signal')
        self.add_act('project', Project, 'signal')
        self.add_act('decide', Decide, 'signal', 'control')
        self.add_act('reach', Reach)
        self.add_act('press', Press, 'comms', 'control', duration=1.0)

        self.add_cond('look', 'percieve', condition=self.acts['look'].complete)
        self.add_cond('percieve', 'comprehend',
                      condition=self.acts['percieve'].complete)
        self.add_cond('comprehend', 'project',
                      condition=self.acts['comprehend'].complete)
        self.add_cond('project', 'decide', condition=self.acts['project'].complete)
        self.add_cond('decide', 'reach', condition=self.acts['decide'].complete)
        self.add_cond('reach', 'press', condition=self.acts['reach'].complete)


class Operator(Function):
    __slots__ = ('signal', 'switch', 'control', 'comms', 'video')
    flow_signal = OperatorSignal
    flow_switch = Switch
    flow_control = Control
    flow_comms = Comms
    flow_video = Video
    flownames = {"operator_signal": "signal"}


class RoverHuman(Rover):

    def init_architecture(self, **kwargs):
        """Initialize the functional architecture."""
        self.add_flow("ground", Ground, p=self.p.ground)
        self.add_flow("pos_signal", Pos)
        self.add_flow('pos', Pos)
        self.add_flow("ee_12", EE)
        self.add_flow("ee_5", EE)
        self.add_flow("ee_15", EE)
        self.add_flow("video", Video)
        self.add_flow("auto_control", Control)
        self.add_flow("motor_control", Control)
        self.add_flow("switch", Switch)
        self.add_flow("comms", Comms)
        self.add_flow("fault_sig", FaultSig)

        self.add_fxn("power", Power, "ee_15", "ee_5", "ee_12", "switch")
        self.add_fxn("perception", Perception, "ground", 'pos', 'pos_signal',
                     "ee_12", "video")
        self.add_fxn("communications", Communications, "comms", "ee_12", "pos_signal",
                     "video")

        self.add_fxn("operator", Operator, "switch", "comms")

        self.add_fxn("override", Override, "comms", "ee_5", "motor_control",
                     "auto_control")
        self.add_fxn("plan_path", PlanPath, "video", "pos_signal", "ground",
                     "auto_control", "fault_sig", p=self.p.correction)
        drive_m = {"mode_args": self.p.drive_modes, 'deg_params': self.p.degradation}
        self.add_fxn("drive", Drive, "ground", 'pos', "ee_15", "motor_control",
                     "fault_sig", m=drive_m)


asg_pos = {'look': [-0.9, 0.88], 'percieve': [-0.68, 0.62], 'comms': [-0.66, -0.68],
           'pos_signal': [-0.45, 0.91], 'video': [0.02, 0.7],
           'comprehend': [-0.46, 0.4], 'signal': [0.46, 0.44], 'project': [-0.24, 0.1],
           'decide': [-0.01, -0.15], 'control': [0.81, -0.13], 'reach': [0.36, -0.44],
           'press': [0.9, -0.69]}


if __name__ == "__main__":
    from fmdtools.analyze.graph import ActionArchitectureGraph
    hum = HumanActions()
    ag = ActionArchitectureGraph(hum)
    ag.set_pos(**asg_pos)
    ag.draw()

    rvr = RoverHuman()
    # mdl = RoverHuman(params=RoverParam('sine', amp=4.0))


#     endresults, mdlhist = prop.nominal(mdl)
#     plot_map(mdl, mdlhist)
#     an.plot.mdlhists({'nominal':mdlhist}, fxnflowvals=['Power'])
#     an.plot.mdlhists({'nominal':mdlhist}, fxnflowvals={'Ground'})

