# -*- coding: utf-8 -*-
"""
Human Rover Model

@authors: mmohame2 and dhulse

Functions:
    - Communications
    - Avionics
    - Camera/Guidance
    - Structures
    - Power
    - Thermal?

Flows:
    - Communications
    - Ground
    - Force
    - EE
    - Camera
"""
from fmdtools.define.role.parameter import Parameter, SimParam
from fmdtools.define.role.state import State
from fmdtools.define.role.mode import Mode
from fmdtools.define.block import FxnBlock
from fmdtools.define.model import Model
from fmdtools.define.flow import Flow
from fmdtools.sim.approach import SampleApproach, NominalApproach


import matplotlib.pyplot as plt
import numpy as np
import fmdtools.sim.propagate as prop
import fmdtools.analyze as an
import itertools

from rover_model import translate_angle, turn_func, sin_func, sin_angle_func, in_area, dist, find_line_dist, gen_param_space
from rover_model import Environment, Power
from rover_model import DegParam

class Operations(FxnBlock):
    def __init__(self, name, flows, params):
        self.p=params
        super().__init__(name, flows, flownames={'Stimulus': 'Control'}, states={'t_comp':np.NaN})
        self.assoc_modes({'no_con': [1e-4, 200]}, ['drive', 'standby'], initmode='standby', exclusive=True)

    def dynamic_behavior(self, time):
        if not self.in_mode('no_con'):
            if time == 1:
                self.set_mode('drive')
            #if time == 50:
            #   self.set_mode('standby')
        else:
            self.Control.put(powerswitch=-1, rdiff=0)

        if self.in_mode('drive'):
            self.Control.powerswitch = 1
            if in_area(self.p.end[0], self.p.end[1], 2, self.Video.x, self.Video.y):
                self.set_mode('standby')
                self.t_comp=time
            else:
                ycorrection = np.arctan((self.Video.y-self.Video.liney)/(self.Video.vel*np.cos(np.pi/180 * self.Video.heading)+0.001))
                xcorrection = np.arctan((self.Video.x-self.Video.linex)/(self.Video.vel*np.sin(np.pi/180 * self.Video.heading)+0.001))
                self.Control.rdiff = (self.Video.angle - self.Video.heading - 5*(xcorrection+ycorrection))/180
                # self.Control.put(rpower = 1+rdiff, lpower = 1-rdiff)
                # self.Control.limit(rpower=(-1,2), lpower=(-1,2))
        if self.in_mode('standby'):
            self.Control.powerswitch = 0
            self.Control.rdiff = 0

# Check about mech_loss
## Drive - inherit most parts from Drive (but not faults)
class Drive(FxnBlock):
    def __init__(self, name, flows, params):
        super().__init__(name, flows, flownames={"EE_15":"EE_in"})
        base_f, base_d = params.friction, params.drift

        self.assoc_faultstates({'friction':[base_f, {(base_f+0.5), 2*(base_f+0.5), 5*(base_f+0.5)}], 'drift':[base_d, {base_d+0.2, base_d-0.2}], 'transfer':[1.0,{0.0}]}, 'all')

        self.assoc_faultstate_modes(manual_modes={'elec_open':{'transfer':0.0}, 'stuck':{'friction':10.0}, 'stuck_right':{'friction':3.0, 'drift':-0.2}, 'stuck_left':{'friction':3.0, 'drift':0.2}})
        self.key_phases_by='global'
    def dynamic_behavior(self, time):
        rpower = self.transfer*self.EE_in.v*self.MotorControl.rpower/15 + self.drift
        lpower = self.transfer*self.EE_in.v*self.MotorControl.lpower/15 - self.drift
        if self.has_fault("elec_open"): self.EE_in.a = 0
        else:                           self.EE_in.a = (1.0+self.friction)*(lpower + rpower)/12
        if (lpower + rpower) >100: self.add_fault("elec_open")
        else:
            self.Ground.vel= (rpower + lpower)/(1.0+self.friction)
            self.Ground.inc(ang = 180/np.pi*np.arctan((rpower-lpower)/(rpower+lpower +0.001)))
            self.Ground.ang = translate_angle(self.Ground.ang)
            self.Ground.inc(x = np.cos(np.pi/180 *self.Ground.ang) * self.Ground.vel, \
                            y = np.sin(np.pi/180 *self.Ground.ang) * self.Ground.vel)


class GenerateVideo(FxnBlock):
    def __init__(self, name, flows):
        self.set_atts(rad=1)
        super().__init__(name, flows, flownames={'EE_12': 'EE'})
        self.assoc_modes({'failed'}, ['off', 'feed'], initmode='off')

    def dynamic_behavior(self, time):
        if self.in_mode('off'):
            self.EE.a = 0
            self.Video.put(linex=0, liney=0, angle=0, quality=0)
            self.Video.assign(self.Ground, 'x', 'y', 'vel', 'line')
            self.Video.heading = self.Ground.ang
            if self.EE.v == 12:
                self.set_mode("feed")
        elif self.in_mode("feed"):
            if self.EE.v > 8:
                if in_area(self.Ground.x, self.Ground.liney, self.rad, self.Ground.x, self.Ground.y):
                    self.Video.assign(self.Ground, 'x', 'y',
                                      'liney', 'linex', 'vel', 'line', 'angle')
                    self.Video.heading = self.Ground.ang
                    self.Video.quality = 1
                else:
                    self.Video.quality = 0
            elif self.EE.v == 0:
                self.set_mode("off")
            else:
                self.Video.quality = 0
        else:
            self.EE.a = 0
            self.Video.put(linex=0, liney=0, angle=0, quality=0)
            self.Video.assign(self.Ground, 'x', 'y', 'vel', 'line')
            self.Video.heading = self.Ground.ang


class Communications(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows,  flownames={'EE_12': 'EE'})
        self.assoc_modes({'failed_video', 'failed_motorcontrol', 'failed_powercontrol'}, [
                         'on', 'off'], initmode='off', exclusive=False)

    def dynamic_behavior(self, time):
        if self.in_mode('off'):
            self.EE.a = 0
            self.Video.put(linex=0.0, liney=0.0, angle=0.0, quality=0)
            self.MotorControl.put(rpower=0.0, lpower=0.0)
            self.Control.put(power=0)
            if self.EE.v == 12:
                self.set_mode("on")
        elif self.in_mode("on"):
            if self.EE.v == 12:
                self.EE.a = 1
                if self.any_faults():
                    if self.in_mode('failed_video'):
                        self.Video.put(linex=0.0, liney=0.0,
                                       angle=0.0, quality=0.0)
                    elif self.in_mode('failed_motorcontrol'):
                        self.MotorControl.put(rpower=0.0, lpower=0.0)
                    elif self.in_mode('failed_powercontrol'):
                        self.Control.put(power=0)
                else:
                    self.MotorControl.assign(self.ControllerSignal, 'lpower', 'rpower')
                    self.Control.power = self.ControllerSignal.power
            else:
                self.set_mode("off")

class Look(Action):
    def __init__(self, name, flows):
        super().__init__(name, flows)
        self.assoc_modes({'failed_look'}, ['visible', 'not_visible', 'no_action'], initmode='no_action', exclusive=True)
        self.quality = 1
    def generate_signal(self):
        if self.Stimulus.powerswitch == 1:
            if self.Stimulus.rdiff < 0:     self.Signal.S1 = 'TurnLeft'
            elif self.Stimulus.rdiff > 0:   self.Signal.S1 = 'TurnRight'
            else:                           self.Signal.S1 = 'NoAction'

            if self.Signal.LastPowerswitchState == 0:   self.Signal.S2 = 'TurnOn'
            else:                                       self.Signal.S2 = 'NoAction'

        elif self.Stimulus.powerswitch == 0:
            self.Signal.S1 = 'NoAction'
            if self.Signal.LastPowerswitchState == 1:   self.Signal.S2 = 'TurnOff'
            else:                                       self.Signal.S2 = 'NoAction'
        else:   self.Signal.put(S1='alternate', S2='alternate')
    def behavior(self, t):
        self.generate_signal()
        if not self.in_mode('failed_look'):
            if self.Video.vel == 0 and self.Signal.S2 == 'NoAction':
                self.set_mode('no_action')
            else:
                if self.Signal.same(['alternate', 'alternate'], 'S1','S2') or self.Video.quality == 0 :
                    self.set_mode('not_visible')
                else:
                    self.set_mode('visible')
        self.SeeOut.O = self.in_mode('visible')
    def complete(self):
        if self.in_mode('no_action') or self.in_mode('failed_look'):
            return False
        else:                return True


class Percieve(Action):
    def __init__(self, name, flows):
        super().__init__(name, flows)
        self.assoc_modes({'failed_S1', 'failed_S2', 'failed_Video', 'failed_noaction', 'failed_S1_temp', 'failed_S2_temp', 'failed_Video_temp'},
                         ['percieved_all', 'no_action'], initmode='no_action', exclusive=False, name='perc_')

    def behavior(self, t):
        if self.SeeOut.O == False:
            if self.any_faults():
                self.remove_any_faults('no_action', warnmessage='The screen is already not visible to the operator.')
            self.set_mode('no_action')
        if self.has_fault('failed_S1_temp'):
            self.remove_fault('failed_S1_temp')
        if self.has_fault('failed_S2_temp'):
            self.remove_fault('failed_S2_temp')
        if self.has_fault('failed_Video_temp'):
            self.remove_fault('failed_Video_temp')
        if self.GlobalPSF.attention < 3 or self.GlobalPSF.fatigue > 8:
            self.add_fault('failed_S1_temp', 'failed_S2_temp', 'failed_Video_temp')
        if not self.any_faults():
            self.set_mode('percieved_all')
        self.PercieveOut.put(S1=True, S2=True, Video=True)
        if self.in_mode('no_action'):
            self.PercieveOut.put(S1=False, S2=False, Video=False)
        if self.has_fault('failed_noaction'):
            self.PercieveOut.put(S1=False, S2=False, Video=False)
        if self.has_fault('failed_S1'):
            self.PercieveOut.S1 = False
        if self.has_fault('failed_S2'):
            self.PercieveOut.S2 = False
        if self.has_fault('failed_Video'):
            self.PercieveOut.Video = False
    def complete(self):
        if self.has_fault('failed_noaction') or self.in_mode('no_action'):
            return False
        else:                return True


class Comprehend(Action):
    def __init__(self, name, flows):
        super().__init__(name, flows)
        self.assoc_modes({'failed_S1', 'failed_S2', 'failed_Video', 'failed_noaction', 'failed_S1_temp', 'failed_S2_temp', 'failed_Video_temp'},
                         ['comprehend_all', 'no_action'], initmode='no_action', exclusive=False, name='comp_')

    def behavior(self, t):
        if self.has_fault('failed_S1_temp'):
            self.remove_fault('failed_S1_temp')
        if self.has_fault('failed_S2_temp'):
            self.remove_fault('failed_S2_temp')
        if self.has_fault('failed_Video_temp'):
            self.remove_fault('failed_Video_temp')
        if any(x == False for x in self.PercieveOut.values()):
            if self.PercieveOut.S1 == False:    self.add_fault('failed_S1_temp')
            if self.PercieveOut.S2 == False:    self.add_fault('failed_S2_temp')
            if self.PercieveOut.Video == False: self.add_fault('failed_Video_temp')
        if self.GlobalPSF.fatigue > 8 or self.GlobalPSF.stress > 80:        self.choose_rand_fault(['failed_S1_temp', 'failed_S2_temp', 'failed_Video_temp'])

        if not self.any_faults():             self.set_mode('comprehend_all')
        self.ComprehendOut.put(S1=True, S2=True, Video=True)
        if self.in_mode('noaction'):                self.ComprehendOut.put(S1=False, S2=False, Video=False)
        if self.has_fault('failed_noaction'):  self.ComprehendOut.put(S1=False, S2=False, Video=False)
        if self.has_fault('failed_S1', 'failed_S1_temp'):        self.ComprehendOut.S1 = False
        if self.has_fault('failed_S2', 'failed_S2_temp'):        self.ComprehendOut.S2 = False
        if self.has_fault('failed_Video', 'failed_Video_temp'):     self.ComprehendOut.Video = False

    def complete(self):
        if self.in_mode("no_action") or self.has_fault("failed_noaction"):
            return False
        else:                return True


class Project(Action):
    def __init__(self, name, flows, params):
        self.p=params
        super().__init__(name, flows)
        self.assoc_modes({'failed_turn_right', 'failed_turn_left', 'failed_poweron', 'failed_poweroff', 'failed_noturn', 'failed_nopower', 'failed_noaction'},
                         ['nominal'], initmode='nominal', exclusive=False, name='proj_')

    def project_calc(self, ):

        if in_area(self.p.end[0], self.p.end[1], 1, self.Video.x, self.Video.y):
            self.ProjectOut.powerswitch = 0
        else:
            ycorrection = np.arctan((self.Video.y-self.Video.liney)/(
                self.Video.vel*np.cos(np.pi/180 * self.Video.heading)+0.001))
            xcorrection = np.arctan((self.Video.x-self.Video.linex)/(
                self.Video.vel*np.sin(np.pi/180 * self.Video.heading)+0.001))
            self.ProjectOut.rdiff = (self.Video.angle - self.Video.heading -
                     5*(xcorrection+ycorrection))/180
        if self.ProjectOut.powerswitch == 1:
            if self.ProjectOut.rdiff < 0:       self.Signal.S1 = 'TurnLeft'
            elif self.ProjectOut.rdiff > 0:     self.Signal.S1 = 'TurnRight'
            else:                               self.Signal.S1 = 'NoAction'

            if self.Signal.LastPowerswitchState == 0:
                self.Signal.S2 = 'TurnOn'
            else:
                self.Signal.S2 = 'NoAction'

        elif self.ProjectOut.powerswitch == 0:

            self.Signal.S1 = 'NoAction'
            if self.Signal.LastPowerswitchState == 1:
                self.Signal.S2 = 'TurnOff'
            else:
                self.Signal.S2 = 'NoAction'

    def behavior(self, t):
        if t == 1:
            self.ProjectOut.powerswitch = self.Stimulus.powerswitch
        SignalFailure = False
        if self.Signal.S1 == 'alternate' and self.Signal.S2 == 'alternate':
            self.project_calc()
            self.LocalPSF.workload = 5
            SignalFailure = True
        if not self.any_faults():
            if SignalFailure == False or self.ComprehendOut.Video == True:
                self.set_mode('nominal')
            else:
                if SignalFailure == False:
                    if self.ComprehendOut.S1 == False and not self.has_fault('failed_turn_left', 'failed_turn_right', 'failed_noturn'):
                        if self.Signal.S1 == 'NoAction':    self.choose_rand_fault(['failed_turn_left', 'failed_turn_right'])
                        elif self.Signal.S1 == 'TurnRight': self.choose_rand_fault(['failed_turn_left', 'failed_noturn'])
                        elif self.Signal.S1 == 'TurnLeft':  self.choose_rand_fault(['failed_turn_right', 'failed_noturn'])
                    if self.ComprehendOut.S2 == False and not self.has_fault('failed_poweron', 'failed_poweroff', 'failed_nopower'):
                        if self.Signal.S2 == 'NoAction':    self.choose_rand_fault(['failed_poweron', 'failed_poweroff'])
                        elif self.Signal.S2 == 'TurnOn':    self.choose_rand_fault(['failed_poweroff', 'failed_nopower'])
                        elif self.Signal.S2 == 'TurnOff':   self.choose_rand_fault(['failed_poweron', 'failed_nopower'])
                else:
                    if self.Signal.S1 == 'NoAction' and not self.has_fault('failed_turn_left', 'failed_turn_right', 'failed_noturn'):
                        self.choose_rand_fault(['failed_turn_left', 'failed_turn_right'])
                    elif self.Signal.S1 == 'TurnRight' and not self.has_fault('failed_turn_left', 'failed_turn_right', 'failed_noturn'):
                        self.choose_rand_fault(['failed_turn_left', 'failed_noturn'])
                    elif self.Signal.S1 == 'TurnLeft' and not self.has_fault('failed_turn_left', 'failed_turn_right', 'failed_noturn'):
                        self.choose_rand_fault(['failed_turn_right', 'failed_noturn'])
                    if self.Signal.S2 == 'NoAction' and not self.has_fault('failed_poweron', 'failed_poweroff', 'failed_nopower'):
                        self.choose_rand_fault(['failed_poweron', 'failed_poweroff'])
                    elif self.Signal.S2 == 'TurnOn' and not self.has_fault('failed_poweron', 'failed_poweroff', 'failed_nopower'):
                        self.choose_rand_fault(['failed_poweroff', 'failed_nopower'])
                    elif self.Signal.S2 == 'TurnOff' and not self.has_fault('failed_poweron', 'failed_poweroff', 'failed_nopower'):
                        self.choose_rand_fault(['failed_poweron', 'failed_nopower'])

        self.ProjectOut.assign(self.Signal, 'S1', 'S2')

        if self.has_fault('failed_turn_left'):
            self.ProjectOut.S1 = 'TurnLeft'
            if self.Signal.S1 == 'TurnLeft':
                self.remove_fault('failed_turn_left', opermode='nominal', warnmessage='The Signal requires a left turn.')
        elif self.has_fault('failed_turn_right'):
            self.ProjectOut.S1 = 'TurnRight'
            if self.Signal.S1 == 'TurnRight':
                self.remove_fault('failed_turnright', opermode='nominal', warnmessage='The Signal requires a right turn.')
        elif self.has_fault('failed_noturn'):
            self.ProjectOut.S1 = 'NoAction'
            if self.Signal.S1 == 'NoAction':
                self.remove_fault('failed_noturn', opermode='nominal', warnmessage= 'The Signal requires no turns.')

        if self.has_fault('failed_poweron'):
            self.ProjectOut.S2 = 'TurnOn'
            if self.Signal.S2 == 'TurnOn':
                self.remove_fault('failed_poweron', opermode='nominal', warnmessage='The Signal requires to turn on.')
        elif self.has_fault('failed_poweroff'):
            self.ProjectOut.S2 = 'TurnOff'
            if self.Signal.S2 == 'TurnOff':
                self.remove_fault('failed_poweroff', opermode='nominal', warnmessage='The Signal requires to turn off.')
        elif self.has_fault('failed_nopower'):
            self.ProjectOut.S2 = 'NoAction'
            if self.Signal.S2 == 'NoAction':
                self.remove_fault('failed_nopower', opermode='nominal', warnmessage='The Signal requires no power on/off actions.')
    def complete(self):
        if self.has_fault("failed_noaction"):   return False
        else:                                   return True


class Decide(Action):
    def __init__(self, name, flows):
        super().__init__(name, flows)
        self.assoc_modes({'failed_turn_right', 'failed_turn_left', 'failed_poweron', 'failed_poweroff', 'failed_noturn', 'failed_nopower', 'failed_noaction'},
                         ['nominal'], initmode='nominal', exclusive=False, name='decide_')

    def behavior(self, t):
        if not self.any_faults():
            self.set_mode('nominal')
            if self.Signal.S1 != self.ProjectOut.S1:
                if self.ProjectOut.S1 == 'NoAction':
                    self.add_fault('failed_noturn')
                elif self.ProjectOut.S1 == 'TurnRight':
                    self.add_fault('failed_turn_right')
                elif self.ProjectOut.S1 == 'TurnLeft':
                    self.add_fault('failed_turn_left')

            if self.Signal.S2 != self.ProjectOut.S2:
                if self.ProjectOut.S2 == 'NoAction':
                    self.add_fault('failed_nopower')
                elif self.ProjectOut.S2 == 'TurnOn':
                    self.add_fault('failed_poweron')
                elif self.ProjectOut.S2 == 'TurnOff':
                    self.add_fault('failed_poweroff')

        self.DecideOut.assign(self.ProjectOut, 'S1', 'S2')

        if self.has_fault('failed_turn_left'):
            self.DecideOut.S1 = 'TurnLeft'
            if self.Signal.S1 == 'TurnLeft':
                self.remove_fault('failed_turn_left', opermode='nominal', warnmessage='The Signal requires a left turn.')
        elif self.has_fault('failed_turn_right'):
            self.DecideOut.S1 = 'TurnRight'
            if self.Signal.S1 == 'TurnRight':
                self.remove_fault('failed_turnright', opermode='nominal', warnmessage='The Signal requires a right turn.')
        elif self.has_fault('failed_noturn'):
            self.DecideOut.S1 = 'NoAction'
            if self.Signal.S1 == 'NoAction':
                self.remove_fault('failed_noturn', opermode='nominal', warnmessage='The Signal requires no turns.')

        if self.has_fault('failed_poweron'):
            self.DecideOut.S2 = 'TurnOn'
            if self.Signal.S2 == 'TurnOn':
                self.remove_fault('failed_poweron', opermode='nominal', warnmessage='The Signal requires to turn on.')
        elif self.has_fault('failed_poweroff'):
            self.DecideOut.S2 = 'TurnOff'
            if self.Signal.S2 == 'TurnOff':
                self.remove_fault('failed_poweroff', opermode='nominal', warnmessage='The Signal requires to turn off.')
        elif self.has_fault('failed_nopower'):
            self.DecideOut.S2 = 'NoAction'
            if self.Signal.S2 == 'NoAction':
                self.remove_fault('failed_nopower', opermode='nominal', warnmessage='The Signal requires no power on/off actions.')
    def complete(self):
        if self.has_fault("failed_noaction"):   return False
        else:                                   return True


class Reach(Action):
    def __init__(self, name, flows):
        super().__init__(name, flows)
        self.assoc_modes({'cannot_reach'}, exclusive=True)

    def behavior(self, t):
        if not self.any_faults():
            self.ReachOut.assign(self.DecideOut, 'S1', 'S2')
        if self.in_mode('cannot_reach'):
            self.ReachOut.put(S1='NoAction', S2='NoAction')

    def complete(self):
        if self.ReachOut.S1 == 'NoAction' and self.ReachOut.S2 == 'NoAction':   return False
        else:                                                                   return True


class Press(Action):
    def __init__(self, name, flows):
        super().__init__(name, flows)
        self.assoc_modes({'failed_noaction', 'failed_long', 'failed_short'}, exclusive=True, name='press_')

    def behavior(self, t):
        if self.in_mode('failed_noaction'):
            if self.ReachOut.S1 == 'NoAction' and self.ReachOut.S2 == 'NoAction':
                self.remove_fault('failed_noaction', warnmessage='From the outcome of the action Reach the action Turn cannot be performed.')
        else:
            self.PressOut.assign(self.ReachOut, 'S1', 'S2')
            if self.in_mode('failed_long'):
                self.PressOut.time = 1
            elif self.in_mode('failed_short'):
                self.PressOut.time = -1
            else:
                self.PressOut.time = 0

    def complete(self):
        if self.in_mode("no_action", "failed_noaction") or (self.PressOut.S1 == False and self.PressOut.S2 == False):
            return False
        else:                return True

class RoverParam(Parameter, readonly=True):
    """Parameters for rover"""
    period :            float = 1.0             #period of the curve (for sine linetype)
    end :               tuple = (10.0, 10.0)    #end of the curve (requires instantiation)
    initangle :         float = 0.0             #initial rover angle
    linetype :          str = 'sine'            #line type (sine or turn)
    linetype_set = ('sine', 'turn')
    amp :               float = 1.0             #amplitude of sine wave (input for sine linetype)
    wavelength :        float=50.0              #wavelength of sine wave (input for sine linetype)
    radius :            float=20.0              #radius of turn (input for turn linetype)
    start :             float=20.0              #start of turn (input for turn linetype)
    fatigue :           float=0.0
    stress :            float=0.0
    attention :         float=10.0
    degradation :       DegParam = DegParam()
    def __init__(self, *args, **kwargs):
        linetype=self.get_true_field('linetype', *args, **kwargs)
        if linetype=='sine':
            wavelength = self.get_true_field('wavelength', *args, **kwargs)
            amp = self.get_true_field('amp', *args, **kwargs)
            kwargs['period']=2*np.pi/wavelength
            kwargs['initangle'] = sin_angle_func(0.0, amp, kwargs['period'])
            kwargs['end']=(wavelength,0.0)
        elif linetype=='turn':
            radius = self.get_true_field('radius',*args,**kwargs)
            start = self.get_true_field('start',*args,**kwargs)
            kwargs['end']=(radius+start, radius+start)
        super().__init__(*args, **kwargs)



class Controller(FxnBlock):
    def __init__(self, name, flows,params):
        #self.add_params(params)
        super().__init__(name, flows, states={'attention':params.attention})
        self.add_flow('Signal', {'S1': 'NoAction', 'S2': 'NoAction', 'rdiff': 0.0, 'LastState': 'Stop', 'LastPowerswitchState': False})
        self.add_flow('LocalPSF', {'WorkLoad':1})
        self.add_flow('SeeOut', {'O': True})
        self.add_flow('PercieveOut', {'S1': False, 'S2': False, 'Video': False})
        self.add_flow('ComprehendOut', {'S1': False, 'S2': False, 'Video': False})
        self.add_flow('ProjectOut', {'S1': 'NoAction', 'S2': 'NoAction', 'rdiff': 0.0, 'powerswitch': 0})
        self.add_flow('DecideOut', {'S1': 'NoAction', 'S2': 'NoAction'})
        self.add_flow('ReachOut', {'S1': 'NoAction', 'S2': 'NoAction'})
        self.add_flow('PressOut', {'S1': 'NoAction', 'S2': 'NoAction', 'time': 0})

        self.add_act('Look', Look, self.Signal, self.Video,self.Stimulus, self.SeeOut)
        self.add_act("Percieve",  Percieve, self.SeeOut, self.PercieveOut, self.GlobalPSF)
        self.add_act("Comprehend",  Comprehend, self.PercieveOut, self.ComprehendOut, self.GlobalPSF)
        self.add_act("Project", Project, self.ComprehendOut,self.ProjectOut, self.Video, self.Signal, self.Stimulus, self.GlobalPSF, self.LocalPSF, params=params)
        self.add_act("Decide", Decide, self.ProjectOut, self.Signal, self.DecideOut, self.GlobalPSF)
        self.add_act("Reach",  Reach, self.DecideOut, self.ReachOut)
        self.add_act("Press", Press, self.ReachOut, self.PressOut)

        self.add_cond("Look", "Percieve", 'Done_Look', self.Look.complete)
        self.add_cond("Percieve", "Comprehend", 'Done_Percieve', self.Percieve.complete)
        self.add_cond("Comprehend", "Project", 'Done_Comprehend', self.Comprehend.complete)
        self.add_cond("Project", "Decide", 'Done_Project', self.Project.complete)
        self.add_cond("Decide", "Reach", 'Done_Decide', self.Decide.complete)
        self.add_cond("Reach", "Press", 'Done_Reach', self.Reach.complete)

        asg_pos ={'Press': [0.942, 0.076], 'ComprehendOut': [0.302, 0.452], 'SeeOut': [0.028, 0.732],
                 'Project': [0.499, 0.46], 'PressOut': [0.958, -0.07], 'Reach': [0.797, 0.175],
                 'ProjectOut': [0.482, 0.302], 'ReachOut': [0.806, 0.041], 'Comprehend': [0.317, 0.595],
                 'Look': [0.038, 0.877], 'Decide': [0.667, 0.308], 'DecideOut': [0.652, 0.189],
                 'Signal': [0.721, 0.736], 'Percieve': [0.165, 0.738], 'PercieveOut': [0.102, 0.594],
                 'Video': [0.418, 0.889], 'Stimulus': [0.026, 1.034], 'LocalPSF': [0.717, 0.484],
                 'GlobalPSF': [0.521, 0.836]}

        self.assoc_modes({'stuck_turn', 'stuck_power'}, exclusive=False) #'failed_right', 'failed_left', 'failed_long', 'failed_noturn', 'failed_short', 'failed_on', 'failed_off', 'failed_nopower'
        self.build_ASG(per_timestep=True, mode_rep="independent", asg_pos=asg_pos)
    def dynamic_behavior(self, t):
        if self.Stimulus.powerswitch == -1:
            self.Stimulus.rdiff = self.ProjectOut.rdiff
        if not self.Press.complete():
            self.Stimulus.rdiff = 0
            self.ControllerSignal.power = self.Signal.LastPowerswitchState

        else:
            if self.PressOut.S1 != self.Signal.S1:
                if self.PressOut.S1 == 'NoAction':
                    #self.set_mode('failed_action') modes already determined from actions
                    self.Stimulus.rdiff = 0
                elif  self.PressOut.S1 == 'TurnRight':
                    #self.set_mode('failed_right')
                    if self.Stimulus.rdiff != 0:
                        self.Stimulus.rdiff = -1 * self.Stimulus.rdiff
                    else:
                        self.Stimulus.rdiff = 0.3
                elif  self.PressOut.S1 == 'TurnLeft':
                    #self.set_mode('failed_left')
                    if self.Stimulus.rdiff != 0:
                        self.Stimulus.rdiff = -1 * self.Stimulus.rdiff
                    else:
                        self.Stimulus.rdiff = -0.3
            if self.PressOut.S2 == 'TurnOn':
                self.ControllerSignal.power = 1
                #if self.PressOut.S1 != self.Signal.S1:
                #    self.set_mode('failed_on')
            elif self.PressOut.S2 == 'TurnOff':
                self.ControllerSignal.power = 0
                #if self.PressOut.S1 != self.Signal.S1:
                #    self.set_mode('failed_off')
            else:
                self.ControllerSignal.power = self.Signal.LastPowerswitchState
                #if self.PressOut.S1 != self.Signal.S1:
                #    self.set_mode('failed_nopower')
        if self.PressOut.time == 1:
            self.Stimulus.rdiff = 1.1 * self.Stimulus.rdiff
            self.PressOut.time = 0
        elif self.PressOut.time == -1:
            self.Stimulus.rdiff = 0.9 * self.Stimulus.rdiff
            self.PressOut.time = 0

        if not self.has_fault('stuck_turn'):
            vel_adj = max(0.2, 1- 0.9*abs(self.Stimulus.rdiff*20))
            self.ControllerSignal.put(rpower = vel_adj*(1+self.Stimulus.rdiff), lpower = vel_adj*(1-self.Stimulus.rdiff))
            self.ControllerSignal.limit(rpower=(-1,2), lpower=(-1,2))

        if self.Stimulus.powerswitch == -1 and self.Stimulus.rdiff == 0 and (self.Signal.LastPowerswitchState == self.ControllerSignal.power):
            if self.GlobalPSF.fatigue < 5:
                self.GlobalPSF.attention = self.GlobalPSF.attention -1
            else:
                self.GlobalPSF.attention = self.GlobalPSF.attention - 2

            if self.GlobalPSF.attention < 0:
                self.GlobalPSF.attention = 0
        else:
                self.GlobalPSF.attention = self.attention

        if self.has_fault('stuck_power'):
           self.ControllerSignal.power  = self.Signal.LastPowerswitchState

        self.Signal.LastPowerswitchState = bool(self.ControllerSignal.power)

    def complete(self):
        return True

    def time_up(self):
        if self.time >= 5:
            return True
        else:
            return False


class Rover(Model):
    def __init__(self, params=RoverParam(),
                 modelparams=ModelParam(times=(0, 80),phases=(('start',0, 30), ('end', 31, 80))),
                 valparams={'end_rad':1.0}):
        super().__init__(params, modelparams, valparams)

        self.add_flow('GlobalPSF', {'fatigue': params.fatigue, 'stress': params.stress,'attention': params.attention})
        self.add_flow('Ground', {'x':0.0,'y':0.0,'liney':0.0,'linex':0.0, 'lbx':0.0, 'lby':-1.5,\
                                 'ubx':0.0,'uby':1.5, 'vel':0.0, 'line':0.0, 'angle':params.initangle, 'ang':-params.initangle})
        # self.add_flow('Pos_Signal', {'x':0.0,'y':0.0,'liney':0.0,'linex':0.0, 'heading':0.0, 'vel':0.0, 'line':0, 'angle':0.0})
        #self.add_flow('Signal', {'S1': 'NoAction', 'S2': 'NoAction'})
        self.add_flow('EE_12', {'v': 0.0, 'a': 0.0})
        self.add_flow('EE_15', {'v': 0.0, 'a': 0.0})
        self.add_flow('Video', {'x': 0.0, 'y': 0.0, 'liney': 0.0, 'linex': 0.0, 'heading': 0.0,
                      'vel': 0.0, 'line': 0.0, 'angle': params.initangle, 'quality': 1})
        # self.add_flow('Signal', {'rpower':0.0, 'lpower':0.0})
        self.add_flow('MotorControl', {'rpower': 0.0, 'lpower': 0.0})
        self.add_flow('Control', {'power': 0.0})
        self.add_flow('ControllerSignal', {
                      'rpower': 0.0, 'lpower': 0.0, 'power': 0.0})
        self.add_flow('Stimulus',  {'powerswitch': 0, 'rdiff': 0.0})
        # self.add_flow('Example_Disconnect')

        self.add_fxn("Power", ["EE_15", 'EE_12', "Control"], Power)
        # self.add_fxn("Operator", ["Comms", "OverrideComms", "Pos_Signal", "Control"], Operator)
        self.add_fxn("GenerateVideo", ["Ground", "EE_12", "Video"], GenerateVideo)
        self.add_fxn("Operations", ['Video', "Stimulus"], fclass=Operations, fparams=params)
        self.add_fxn("Controller", ['Stimulus', 'GlobalPSF','Video', 'ControllerSignal'], Controller, fparams=params)
        self.add_fxn("Communications", ['Video', 'ControllerSignal', "MotorControl", 'Control', "EE_12"], Communications)
        self.add_fxn("Drive", ["Ground", "EE_15","MotorControl"], fclass=Drive, fparams=params.degradation)
        self.add_fxn("Environment", ['Ground'], Environment, fparams=params)

        pos_bip = {'Video': [1.0488760546884315, 0.6236350372168163],
                     'EE_15': [0.35507976090629734, 0.05846736141517972],
                     'EE_12': [0.32641434609597797, 0.3028352957153257],
                     'Control': [0.10139545679964745, 0.4614616216661047],
                     'GenerateVideo': [1.0179376111737946, 0.41643297470523843],
                     'Ground': [1.0045395358519196, 0.16038976577762787],
                     'Operations': [0.9494356604829556, 0.8303368446246009],
                     'Stimulus': [0.7196313684660003, 0.9248998534158376],
                     'Drive': [0.6618619541164777, 0.05391707408586505],
                     'Controller': [0.5738657729899638, 0.7527772311106469],
                     'GlobalPSF': [0.46382738128566847, 0.9093305348993589],
                     'MotorControl': [0.6465487923055856, 0.25459143623023395],
                     'ControllerSignal': [0.338392216167285, 0.6676446332768291],
                     'Communications': [0.4193706492672662, 0.47284425365204774],
                     'Environment': [0.8812338278458673, -0.03270816057506423],
                     'Power': [0.08855773535620892, 0.22730017067603847]}

        self.build_model(fxnflowgraph_pos=pos_bip)
    def find_classification(self,scen,mdlhist):
        modes, modeproperties = self.return_faultmodes()
        classification = str()
        at_finish=True
        if not in_area(self.flows['Ground'].x,self.flows['Ground'].y,self.valparams['end_rad'],self.params.end[0],self.params.end[1]):
                                classification = "incomplete mission"
                                at_finish = False
        if any(modes):          classification = classification +' faulty'
        if not classification:  classification = 'nominal mission'
        #if (np.isnan(self.fxns['Operations'].t_comp) or self.fxns['Operations'].t_comp > 65): at_finish=False
        num_modes = len(modes)
        end_dist = dist(self.flows['Ground'].x,self.flows['Ground'].y,self.params.end[0],self.params.end[1])

        #tot_deviation = np.sum(np.sqrt((mdlhist['nominal']['flows']['Ground']['x']-mdlhist['faulty']['flows']['Ground']['x'])**2 + (mdlhist['nominal']['flows']['Ground']['y']-mdlhist['faulty']['flows']['Ground']['y'])**2))
        line_dist = find_line_dist(self.flows['Ground'].x,self.flows['Ground'].y, mdlhist['nominal']['flows']['Ground']['linex'], mdlhist['nominal']['flows']['Ground']['liney'])

        return {'rate':0,'cost':0, 'prob':scen.prob, 'expected_cost':0,'at_finish':at_finish, 'line_dist':line_dist, 'num_modes':num_modes, 'end_dist':end_dist, 'faults':modes, 'classification':classification, 'x':self.flows['Ground'].x ,'y':self.flows['Ground'].y}




def plot_map(mdl, mdlhist):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_ground = mdlhist['flows']['Ground']['x']
    y_ground = mdlhist['flows']['Ground']['liney']
    plt.plot(x_ground, y_ground, label="Centerline")

    x_rover = mdlhist['flows']['Ground']['x']
    y_rover = mdlhist['flows']['Ground']['y']
    plt.plot(x_rover, y_rover, label="Rover")

    plt.scatter(mdl.params.end[0], mdl.params.end[1],
                label="End Location")
    plt.scatter(mdlhist['flows']['Ground']['x'][-1],
                mdlhist['flows']['Ground']['y'][-1], label="Final Position")

    plt.scatter(0, 0, label="Start Location")

    plt.xlabel("x-distance (meters)")
    plt.ylabel("y-distance (meters)")
    plt.title("Rover Centerline Tracking")
    plt.grid()

    plt.legend()

    fig = plt.figure()
    if mdl.params.linetype == 'sine':
        y_line = [sin_func(x, y_rover[i], mdl.params.amp, mdl.params.period)[
            1] for i, x in enumerate(x_rover)]
    elif mdl.params.linetype == 'turn':
        y_line = [turn_func(x, y_rover[i], mdl.params.radius, mdl.params.start)[
            1] for i, x in enumerate(x_rover)]

    plt.plot(x_rover, y_rover-y_line)
    plt.xlabel("x-distance (meters)")
    plt.ylabel("y-error (meters)")
    plt.title("Rover Centerline Error")



if __name__ == "__main__":
    import matplotlib.pyplot as pyplot


    mdl = Rover(params=RoverParam('sine', amp=4.0))
    endresults, mdlhist = prop.nominal(mdl)
    plot_map(mdl, mdlhist)
    an.plot.mdlhists({'nominal':mdlhist}, fxnflowvals=['Power'])
    an.plot.mdlhists({'nominal':mdlhist}, fxnflowvals={'Ground'})
    

    
   
    
    
    
   
    mdl=Rover()
    behave_endclasses_hum, behave_mdlhists_hum = prop.nominal_approach(mdl, behave_nomapp_hum, run_stochastic=True)   
    
    nom_comp_hum = an.tabulate.nominal_factor_comparison(behave_nomapp_hum, behave_endclasses_hum, ['t_exp','t_stress'], ['at_finish'], percent=False, give_ci=True, return_anyway=True)
    nom_comp_hum
    
    fig = an.plot.nominal_factor_comparison(nom_comp_hum, 'at_finish', maxy=1.1, xlabel='single-day time (hours)', figsize=(10,6), title="", error_bars=True)
    
    
    scendict = behave_nomapp_hum.get_param_scens('behave_nomapp_hum', 't_exp','t_stress')
    late_scens = scendict[13,9]
    early_scens= scendict[13,1]
    
    plt.figure()
    plt.hist([behave_nomapp_hum.scenarios[scen].p['stress'] for scen in early_scens], alpha=0.5, label='early')
    plt.hist([behave_nomapp_hum.scenarios[scen].p['stress'] for scen in late_scens], alpha=0.5, label='late')
    plt.legend()
    
    plt.figure()
    plt.hist([behave_nomapp_hum.scenarios[scen].p['fatigue'] for scen in early_scens], alpha=0.5, label='early')
    plt.hist([behave_nomapp_hum.scenarios[scen].p['fatigue'] for scen in late_scens], alpha=0.5, label='late')
    plt.legend()
    
    plt.figure()
    plt.hist([int(behave_endclasses_hum[scen]['at_finish']) for scen in early_scens], alpha=0.5, label='early')
    plt.hist([float(behave_endclasses_hum[scen]['at_finish']) for scen in late_scens], alpha=0.5, label='late')
    plt.legend()
    
    """
    #gen initial values for stress and experience
    stress_param = np.random.default_rng(seed=101).gamma(2,1.9,101)
    stress_param = list(stress_param)
    experience_param = np.random.default_rng(seed=101).gamma(1,1.9,101)
    experience_param = list(experience_param)

    #long term human degradation
    deg_mdl_hum_long = HumanDegradationLong()
    endresults,  mdlhist_hum_long = prop.nominal(deg_mdl_hum_long)
    an.plot.mdlhists(mdlhist_hum_long)

  #  params_hum_long = get_longhuman_params_from(mdlhist_hum_long)

    #stochastic over replicates
    nomapp = NominalApproach()
    nomapp.add_param_ranges(gen_long_degPSF_param, 'nomapp', experience_param, scen = (1,101,1))
    #nomapp.add_seed_replicates('test', 100)
    endclasses, mdlhists_hum_long = prop.nominal_approach(deg_mdl_hum_long, nomapp, run_stochastic=True)
    an.plot.mdlhists(mdlhists_hum_long, aggregation='mean_std')

    #nominal
    deg_mdl = RoverDegradation()
    endresults,  mdlhist = prop.nominal(deg_mdl)
    an.plot.mdlhists(mdlhist)
    #nominal
    deg_mdl_hum_short = HumanDegradationShort()
    endresults,  mdlhist_hum_short = prop.nominal(deg_mdl_hum_short)
    an.plot.mdlhists(mdlhist_hum_short)


    #stochastic
    deg_mdl = RoverDegradation()
    endresults,  mdlhist = prop.nominal(deg_mdl, run_stochastic=True)
    an.plot.mdlhists(mdlhist)
    #stochastic
    deg_mdl_hum_short = HumanDegradationShort()
    endresults,  mdlhist_hum_short = prop.nominal(deg_mdl_hum_short, run_stochastic=True)
    an.plot.mdlhists(mdlhist_hum_short)

    #stochastic over replicates
    nomapp = NominalApproach()
    nomapp.add_seed_replicates('test', 100)
    endclasses, mdlhists = prop.nominal_approach(deg_mdl, nomapp, run_stochastic=True)
    an.plot.mdlhists(mdlhists, aggregation='mean_std')

    #stochastic over replicates
    nomapp = NominalApproach()
    nomapp.add_param_ranges(gen_short_degPSF_param, 'nomapp', mdlhists_hum_long, stress_param, scen = (0,100,1), t= (1,101,1) )
    #nomapp.add_param_ranges(gen_short_degPSF_param, 'nomapp', mdlhists_hum_long, stress_param, scen = (0,100,1), t= (1,101,20) )
    #nomapp.add_seed_replicates('test', 100)
    endclasses, mdlhists_hum_short = prop.nominal_approach(deg_mdl_hum_short, nomapp, run_stochastic=True)
    an.plot.mdlhists(mdlhists_hum_short, aggregation='mean_std')
    #individual slice
    # params = get_params_from(mdlhist, 10)


    #params_hum_short = get_human_params_from(mdlhist_hum_short, 10)
    #paramdist_hum = get_human_paramdist_from(mdlhists_hum_short, 10)
    #plt.hist(paramdist_hum['fatigue'])
    #plt.hist(paramdist_hum['stress'])

    #question -- how do we sample this:
    #   - all replicates?
    #   - random sample of them?
    #   - what about times?
    #   - what if we get a complementary sample of times and etc?
    #   - if states in one replicate are the same as a different at the next, can we only sample one?

    #temp_params = gen_long_deg_param_list (mdlhists, mdlhists_hum_long, 100, 100)
    #behave_nomapp = NominalApproach()
    #behave_nomapp.add_param_ranges(gen_sample_params, 'behave_nomapp', mdlhists, t=(1,100, 10), scen = (1,100,5))
    #behave_nomapp.add_param_ranges(gen_sample_params, 'behave_nomapp', mdlhists_hum_short, temp_params, t=(0,10, 1), scen = (0,10000,1))
    #behave_nomapp.add_param_ranges(gen_sample_params, 'behave_nomapp', mdlhists_hum_short, temp_params, t=(0,10, 2), scen = (0,500,1))
    
    
    nomapp_human_short = NominalApproach()
    nomapp_human_short.add_seed_replicates('test', 100)
    
    behave_nomapp_hum = NominalApproach()
    behave_nomapp_hum.add_param_ranges(sample_human_params, 'behave_nomapp', mdlhists_hum_short, t=(0,10, 1), scen = (1,100,5))
    
    mdl = Rover()
    behave_endclasses_hum, behave_mdlhists_hum = prop.nominal_approach(mdl, behave_nomapp_hum, run_stochastic=True)
    f = plot_trajectories(behave_mdlhists_hum)
    an.plot.nominal_vals_2d(behave_nomapp_hum, behave_endclasses_hum, 't', 'scen')


    #behave_nomapp = NominalApproach()
    #behave_nomapp.add_param_ranges(gen_sample_params_combined, 'behave_nomapp', mdlhists, mdlhists_hum, t=(1,100, 10), scen = (1,100,5))

    #mdl=Rover()
    #behave_endclasses, behave_mdlhists = prop.nominal_approach(mdl, behave_nomapp)
    #f = plt.figure()
    #f = plot_trajectories(behave_mdlhists)
    #an.plot.nominal_vals_2d(behave_nomapp, behave_endclasses, 't', 'scen')
    """


    """
    mdl = Rover(params=gen_params('turn', amp=4))
    #mdl = Rover(params=gen_params('turn'))

    dot = an.graph.show(mdl, gtype="fxnflowgraph")  # , renderer='graphviz')
    endresults,  mdlhist1 = prop.nominal(mdl)
    phases, modephases = mdlhist.get_modephases()

    endresults,  mdlhist2 = prop.one_fault(mdl, 'Controller', 'perc_failed_S1', time=5)

    endresults,  mdlhist = prop.one_fault(mdl, 'Operations', 'no_con', time=2)

    endresults,  mdlhist = prop.one_fault(mdl, 'Power', 'short', time=2)
    #endresults,  mdlhist = prop.mult_fault(mdl, {20:{'Drive': ['mech_loss','elec_open']}})

    #an.plot.mdlhists(mdlhist, time_slice=20, fxnflowvals={'Drive':'faults'},
    #                 phases=phases, modephases=modephases)
    an.plot.mdlhists(mdlhist, time_slice=2, fxnflowvals={'GlobalPSF':'all', 'Ground':['x','y'],'Controller':'Look', 'Power':'power'},
                     phases=phases, modephases=modephases)
    an.plot.mdlhists(mdlhist1, time_slice=2, fxnflowvals={'Power':'power'},
                     phases=phases, modephases=modephases)
    an.plot.mdlhistvals(mdlhist1, time=2, fxnflowvals={'Power':'power'},
                     phases=phases, modephases=modephases)

    an.plot.mdlhistvals(mdlhist, time=2, fxnflowvals={'GlobalPSF':'all', 'Ground':['x','y'],'Controller':'Look', 'Power':'power'},
                     phases=phases, modephases=modephases)

    an.plot.indiv_mdlhists(mdlhist,fxnflows={'GlobalPSF':'all', 'Ground':['x','y'],'Controller':'Look', 'Power':'power'},
                     phases=phases, modephases=modephases)
    an.plot.mdlhist(mdlhist,fxnflows={'GlobalPSF':'all', 'Ground':['x','y'],'Controller':'Look', 'Power':'power'},
                     phases=phases, modephases=modephases)
    """
    """
    mdl = Rover(params=gen_params('turn', amp=4))
    act_pos={'Percieve':[0.5,0.5]}

    pos = an.graph.set_pos(mdl.fxns['Controller'].flow_graph, pos=act_pos)

    an.graph.exec_order(mdl, show_dyn_arrows=True)
    an.graph.exec_order(mdl.fxns['Controller'], show_dyn_arrows=True, gtype='actions')
    an.graph.exec_order(mdl.fxns['Controller'], show_dyn_arrows=True, gtype='combined')
    an.graph.exec_order(mdl.fxns['Controller'], show_dyn_arrows=True, gtype='flows')
    """
    """
    PROTOTYPE CODE FOR DISPLAYING ACTION GRAPHS

    #reshist
    endresults,  mdlhist2 = prop.one_fault(mdl, 'Operations', 'no_con', time=2)
    reshist, diff, summary = an.process.hist(mdlhist2)
    restab = an.tabulate.hist(reshist)
    # overall - bip
    an.graph.result_from(mdl, reshist, 1)
    an.graph.result_from(mdl, reshist, 2)
    an.graph.result_from(mdl, reshist, 5)
    # function

    #matplotlib
    an.graph.get_asg_plotlabels(mdl.fxns['Controller'].flow_graph, mdl.fxns['Controller'],  reshist, 1)
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='actions')
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='flows')
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='flows')
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='combined', showfaultlabels=False, pos=act_pos)
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='combined', pos=act_pos, scale=0.5)
    #graphviz
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='actions', renderer='graphviz')
    an.graph.result_from(mdl.fxns['Controller'], reshist, 10, gtype='flows', renderer='graphviz')
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='flows', renderer='graphviz')
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='combined', showfaultlabels=False, pos=act_pos, renderer='graphviz')
    an.graph.result_from(mdl.fxns['Controller'], reshist, 1, gtype='combined', pos=act_pos, renderer='graphviz')

    an.graph.animation_from(mdl.fxns['Controller'], reshist, gtype='actions', renderer='netgraph')

    #matplotlib
    an.graph.show(mdl.fxns['Controller'].action_graph, gtype='fxngraph', pos=act_pos, scale=0.4, arrows=True)
    an.graph.show(mdl.fxns['Controller'].flow_graph, pos=act_pos, highlight=[[*mdl.fxns['Controller'].actions],[],[]],seqgraph=mdl.fxns['Controller'].action_graph)
    #netgraph
    an.graph.show(mdl.fxns['Controller'].action_graph, gtype='fxngraph', renderer='netgraph', pos=act_pos, scale=0.3, arrows=True)
    _,_, gra = an.graph.show(mdl.fxns['Controller'].flow_graph, pos=act_pos, scale=0.6, renderer='netgraph', highlight=[[*mdl.fxns['Controller'].actions],[],[]],seqgraph=mdl.fxns['Controller'].action_graph)
    _,_, gra = an.graph.show(mdl.fxns['Controller'].flow_graph, pos=act_pos, scale=0.6, renderer='netgraph', highlight=[[*mdl.fxns['Controller'].actions],[],[]],seqgraph=mdl.fxns['Controller'].action_graph, seqlabels=True)
    #graphviz
    dot = an.graph.show(mdl.fxns['Controller'].action_graph, gtype='fxngraph', renderer='graphviz', arrows=True)
    dot_bip = an.graph.show(mdl.fxns['Controller'].flow_graph, gtype='fxnflowgraph', renderer='graphviz', arrows=True,seqgraph=mdl.fxns['Controller'].action_graph, seqlabels=True)

    #matplotlib
    an.graph.show(mdl.fxns['Controller'], gtype='actions', scale=0.6)
    an.graph.show(mdl.fxns['Controller'], gtype='flows', scale=0.6)
    an.graph.show(mdl.fxns['Controller'], gtype='combined', scale=0.6)


    #graphviz
    dot_act = an.graph.show(mdl.fxns['Controller'], gtype='actions', renderer='graphviz')
    dot_flows = an.graph.show(mdl.fxns['Controller'], gtype='flows',  renderer='graphviz')
    dot_comb = an.graph.show(mdl.fxns['Controller'], gtype='combined', renderer='graphviz')

    #component
    an.graph.show(mdl, gtype='component', scale=0.6)
    """


    #from matplotlib import pyplot as plt
    #a = plt.figure()

    #gr.draw_edges(edge_paths, edge_width, edge_color, edge_alpha, edge_zorder, True, node_size)

    #graphviz

    #netgraph.InteractiveGraph(mdl.fxns['Controller'].flow_graph, node_positions=act_pos)
    #an.plot.mdlhists(mdlhist)
    # , colors='gray',linestyles='dashed' - for phases
    #  ls='--', color='b' - for nominal
    # for

    #an.plot.mdlhistvals(mdlhist, time=15, fxnflowvals={'GlobalPSF':'all', 'Controller':'all', 'Ground':['x','y']})


    #endresults,  mdlhist = prop.one_fault(mdl, 'Controller','failed_turn_right', time=10)

    # #plot_map(mdl, mdlhist)

    # #mdl.fxns['Controller'].updatefxn('dynamic', time=1)
    #
    # app=SampleApproach(mdl, faults='Controller', phases={'drive':phases['Operations']['drive']})
    # endclasses, mdlhists = prop.approach(mdl, app)



    # from rover_model import plot_trajectories

    #fig = plot_trajectories(mdlhists, app=app, faultlabel='Faulty Scenarios', title='', mode_trunc=len('Controller'), mode_trunc_end=4, show_labels=False, xlim=(-1,80), ylim=(-10,10), faultalpha=0.5)

    # #fig = plot_trajectories(mdlhists, app=app, faultlabel='Faulty Scenarios', title='', mode_trunc=len('Controller'), mode_trunc_end=4, show_labels=False)


    # reshists, diffs, summaries = an.process.hists(mdlhists)
    # an.tabulate.fullfmea(endclasses, summaries)
    # an.tabulate.fullfmea(endclasses, summaries)['degraded flows']


    #mdl.flows['Stimulus'].powerswitch=1
    #mdl.fxns['Controller'].updatefxn('dynamic', time=2)
    #mdl.fxns['Controller'].updatefxn('dynamic', time=3)
#     classgraph = mdl.return_typegraph()
#     pos = an.graph.set_pos(classgraph, gtype='typegraph')
#     an.graph.show(classgraph, gtype='typegraph', pos=pos) #, pos=class_tree)

#     an.graph.exec_order(mdl, gtype='fxngraph')
#     an.graph.exec_order(mdl, gtype='fxnflowgraph', renderer='graphviz')

#     an.plot.dyn_order(mdl)
#     phases, modephases = an.process.modephases(mdlhist)
#     an.plot.phases(phases, modephases)
#     plot_map(mdl, mdlhist)

#     endresults,  mdlhist = prop.one_fault(mdl, 'Drive','elec_open', staged=True, time=13, gtype='typegraph')

#     an.plot.mdlhistvals(mdlhist, fxnflowvals={'Power':['charge','power']}, time=13, phases=phases, modephases=modephases)

#     an.plot.mdlhistvals(mdlhist, fxnflowvals={'Ground':['x','y', 'angle','vel', 'liney', 'ang']}, time=13, phases=phases)
#     an.plot.mdlhistvals(mdlhist, fxnflowvals={'Pos_Signal':['x','y', 'angle','vel', 'heading']}, time=13, phases=phases)
#     an.plot.mdlhistvals(mdlhist, fxnflowvals={'MotorControl':['rpower','lpower']}, time=13, phases=phases)

#     app = NominalApproach()
#     app.add_param_ranges(gen_params,'sine', 'sine', amp=(0, 10, 0.2), wavelength=(10,50,10))
#     app.assoc_probs('sine', amp=(stats.uniform.pdf, {'loc':0,'scale':10}), wavelength=(stats.uniform.pdf,{'loc':10, 'scale':40}))
#     #app.add_param_ranges(gen_params,'turn', radius=(5,40,5), start=(0, 20,5))

#     #labels, faultfxns, degnodes, faultlabels
# #    an.graph.plot_bipgraph(classgraph, {node:node for node in classgraph.nodes},[],[],{}, pos=pos)


#     an.graph.show( gtype='typegraph', scale=0.7)

#     endresults,  mdlhist = prop.one_fault(mdl, 'Drive','elec_open', staged=True, time=13, gtype='fxnflowgraph')
#     an.graph.show( gtype='fxnflowgraph', scale=0.7)

#     reshist, _, _ = an.process.hist(mdlhist)
#     typehist = an.process.typehist(mdl, reshist)
#     an.graph.results_from(mdl, reshist, [10,15,20])
#     an.graph.results_from(mdl, typehist, [10,15,20], gtype='typegraph') #), gtype='typegraph')
#     an.graph.result_from(mdl, reshist, 10, gtype='fxnflowgraph', renderer='graphviz')

    # test_actgraph = Controller("guy", {})
    # plt.figure()
    # test_actgraph.show_ASG("conditions")
    # plt.figure()
    # test_actgraph.show_ASG("flows")
    # plt.figure()
    # test_actgraph.show_ASG()
    # plt.figure()
    # test_actgraph.show_ASG("conditions", with_cond_labels=False)

    # test_actgraph.updatefxn('dynamic',time=1)
    # test_actgraph.updatefxn('dynamic',time=2)
    # test_actgraph.updatefxn('dynamic',time=3)

    # test_actgraph.active_actions
    #endclasses, mdlhists= prop.nominal_approach(mdl, app, pool = mp.Pool(5))


    #fig = an.plot.nominal_vals_1d(app, endclasses, 'amp')
    #fig = an.plot.nominal_vals_1d(app, endclasses, 'radius')

    #app = NominalApproach()
    #app.add_param_ranges(gen_params,'sine','sine', amp=(0, 10, 0.2), wavelength=(10,50,10), dummy=(1,10,1))

    #endclasses, mdlhists= prop.nominal_approach(mdl, app, pool = mp.Pool(5))
    #fig = an.plot.nominal_vals_3d(app, endclasses, 'amp', 'wavelength', 'dummy')
    #app = SampleApproach(mdl, phases = phases, modephases = modephases)

    #endclasses, mdlhist = prop.approach(mdl, app)

    #app_joint = SampleApproach(mdl, phases = phases, modephases = modephases, jointfaults={'faults':2})

    #endclasses, mdlhist = prop.approach(mdl, app_joint)

    #tab = an.tabulate.phasefmea(endclasses, app_joint)
    #an.plot.samplecosts(app_joint, endclasses)

    # an.plot.phases(phases)

    #figs = an.plot.phases(phases, modephases, mdl)
    #figs = an.plot.phases(phases, modephases, mdl, singleplot=False)
