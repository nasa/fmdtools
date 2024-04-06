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
from examples.rover.rover_model import Ground, Pos_Signal, Pos, EE
from examples.rover.rover_model import FaultSig, Power, Perception, Communications
from examples.rover.rover_model import PlanPath, Override, Drive


class OperatorSignalState(State):
    """
    State defining qualitative actions the operator performs.

    Fields
    ------
    turn : str
        Turning actions ('none', 'left', or 'right'). Default is 'none'.
    power : str
        Power actions ('none', 'on', 'off'). Default is 'none'.
    """
    turn: str = 'none'
    power: str = 'none'


class OperatorSignal(Flow):

    __slots__ = ()
    container_s = OperatorSignalState


class HumanActionMode(Mode):
    """Generic/shared modes for human actions."""
    fm_args = ('failed_no_action', )
    opermodes = ('no_action', 'done')
    mode: float = 'no_action'
    exclusive = True


class GenericHumanAction(object):
    """Shared properties of human actions."""

    __slots__ = ()
    def complete(self):
        return not self.m.in_mode('no_action', 'failed_no_action')


class LookMode(HumanActionMode):
    fm_args = ('not_visible', 'wrong_data', 'failed_no_action')


class Look(Action, GenericHumanAction):
    """
    Operator looking at the state of comms signals, switches, etc.

    Should take in external info (comms, switches, etc, and relay perceived info).

    (use this for Comms.recieve)
    """

    __slots__ = ('signal', 'video', 'switch')
    container_m = LookMode
    flow_signal = OperatorSignal
    flow_video = Video
    flow_switch = Switch

    def behavior(self, t):
        if self.video.quality == 0.0:
            self.set_mode('not_visible')


class PerceptionMode(HumanActionMode):
    fm_args = ("failed_speed", "failed_no_action", "failed_turn")


class Percieve(Action, GenericHumanAction):
    """
    Operator percieving the state of comms signals, switches, etc.

    Should take in view (from look) and pass as percieved info.

    (passthrough?)
    """

    __slots__ = ('signal',)
    container_m = PerceptionMode
    flow_signal = OperatorSignal


class Comprehend(Action, GenericHumanAction):
    """
    Operator comprehending the input state.

    Should take in percieved info and distill as situation (moving, turning, etc)

    May fail due to stress >8 or stress >80.
    (passthrough?)
    """

    __slots__ = ('signal',)
    container_m = PerceptionMode
    flow_signal = OperatorSignal


class ProjectMode(HumanActionMode):
    fm_args = ('failed_turn_left', 'failed_turn_right', 'failed_noturn',
               'failed_poweron', 'failed_poweroff', 'failed_no_action')

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


class Decide(Action, GenericHumanAction):
    """
    Operator deciding how to control actions based on projection of input state.
    (Inherit motor control signal behavior here)
    """

    __slots__ = ('signal',)
    container_m = ProjectMode
    flow_signal = OperatorSignal


class Reach(Action, GenericHumanAction):
    """
    Operator reaching for controls to operate.

    (passthrough)
    """

    __slots__ = ('signal',)
    container_m = HumanActionMode
    flow_signal = OperatorSignal


class PressMode(Mode):
    fm_args = ('failed_long', 'failed_short', 'failed_no_action')

class Press(Action, GenericHumanAction):
    """
    Operator presses the button/toggle for the controls.

    (passthrough)
    """

    __slots__ = ('signal', 'switch', 'control')
    container_m = PressMode
    flow_signal = OperatorSignal
    flow_switch = Switch
    flow_control = Control


class HumanActions(ActionArchitecture):

    __slots__ = ()
    initial_action = "look"

    def init_architecture(self, **kwargs):
        self.add_flow('signal', OperatorSignal)
        self.add_flow('switch', Switch)
        self.add_flow('control', Control)
        self.add_flow('comms', Comms)
        self.add_flow('video', Video)
        # flow for workload "local PSF"?

        self.add_act('look', Look, 'signal', 'video', 'switch')
        self.add_act('percieve', Percieve, 'signal')
        self.add_act('comprehend', Comprehend, 'signal')
        self.add_act('project', Project, 'signal')
        self.add_act('decide', Decide, 'signal')
        self.add_act('reach', Reach, 'signal')
        self.add_act('press', Press, 'signal', 'switch', 'control', duration=1.0)

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
    flownames = {"operator_signal": "signal", "manual_control": "control"}


class RoverHuman(Rover):

    def init_architecture(self, **kwargs):
        """Initialize the functional architecture."""
        self.add_flow("ground", Ground, p=self.p.ground)
        self.add_flow("pos_signal", Pos_Signal)
        self.add_flow('pos', Pos)
        self.add_flow("ee_12", EE)
        self.add_flow("ee_5", EE)
        self.add_flow("ee_15", EE)
        self.add_flow("video", Video)
        self.add_flow("auto_control", Control)
        self.add_flow("manual_control", Control)
        self.add_flow("motor_control", Control)
        self.add_flow("switch", Switch)
        self.add_flow("operator_signal", OperatorSignal)
        self.add_flow("comms", Comms)
        self.add_flow("fault_sig", FaultSig)

        self.add_fxn("power", Power, "ee_15", "ee_5", "ee_12", "switch")
        self.add_fxn("operator", Operator,
                     "operator_signal", "switch", "manual_control", "comms", "video")
        self.add_fxn("communications", Communications, "comms", "ee_12", "pos_signal")
        self.add_fxn("perception", Perception, "ground", 'pos', "ee_12", "video")
        self.add_fxn("plan_path", PlanPath, "video", "pos_signal", "ground", 'pos',
                     "auto_control", "fault_sig", p=self.p.correction)
        self.add_fxn("override", Override,
                     "comms", "ee_5", "motor_control", "auto_control",
                     m={'mode': 'override'})
        drive_m = {"mode_args": self.p.drive_modes, 'deg_params': self.p.degradation}
        self.add_fxn("drive", Drive, "ground", 'pos', "ee_15", "motor_control",
                     "fault_sig", m=drive_m)


if __name__ == "__main__":
    from fmdtools.analyze.graph import ActionArchitectureGraph
    hum = HumanActions()
    ag = ActionArchitectureGraph(hum)
    ag.draw()

    rvr = RoverHuman()
    # mdl = RoverHuman(params=RoverParam('sine', amp=4.0))


#     endresults, mdlhist = prop.nominal(mdl)
#     plot_map(mdl, mdlhist)
#     an.plot.mdlhists({'nominal':mdlhist}, fxnflowvals=['Power'])
#     an.plot.mdlhists({'nominal':mdlhist}, fxnflowvals={'Ground'})

