# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:22:05 2021

@author: dhulse

Functions:
    - Communications
    - Avionics
    - Camera/Guidance
    - Structures
    - Power
    - Thermal?

Flows:
    - Communications
    - ground
    - Force
    - EE
    - Camera
"""
from recordclass import asdict
from fmdtools.define.parameter import Parameter, SimParam
from fmdtools.define.state import State
from fmdtools.define.mode import Mode
from fmdtools.define.block import FxnBlock
from fmdtools.define.model import Model
from fmdtools.define.flow import Flow
from fmdtools.sim.approach import SampleApproach
import fmdtools.analyze as an
import fmdtools.sim.propagate as prop
import itertools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

## MODEL FLOWS
class DegParam(Parameter, readonly=True):
    """Parameters for rover degradation"""
    friction :          float = 0.0
    drift :             float = 0.0

class GroundState(State):
    x:      float=0.0
    y:      float=0.0
    linex:  float=0.0
    liney:  float=0.0
    lbx:    float=0.0
    lby:    float=-1.5
    ubx:    float=0.0
    uby:    float=1.5
    vel:    float=0.0
    line:   float=0.0
    angle:  float=0.0
    ang:    float=0.0
class Ground(Flow):
    _init_s = GroundState

class Pos_SignalState(State):
    x:          float=0.0
    y:          float=0.0
    linex:      float=0.0
    liney:      float=0.0
    heading:    float=0.0
    vel:        float=0.0
    line:       int=0
    angle:      float=0.0
class Pos_Signal(Flow):
    _init_s = Pos_SignalState

class EEState(State):
    v:  float=0.0
    a:  float=0.0
class EE(Flow):
    _init_s = EEState

class VideoState(State):
    linex:      float=0.0
    liney:      float=0.0
    angle:      float=0.0
    quality:    float=1.0
class Video(Flow):
    _init_s = VideoState

class ControlState(State):
    rpower:     float=0.0
    lpower:     float=0.0
class Control(Flow):
    _init_s = ControlState

class SwitchState(State):
    power:      float=0.0
class Switch(Flow):
    _init_s = SwitchState

class CommsState(State):
    x:          float=0.0
    y:          float=0.0
    heading:    float=0.0
    vel:        float=0.0
class Comms(Flow):
    _init_s = CommsState

class OverrideState(State):
    rpower:     float=0.0
    lpower:     float=0.0
    active:     float=0.0
class OverrideComms(Flow):
    _init_s = OverrideState
    
class FaultStates(State):
    transfer:   float=1.0
    friction:   float=1.0
    drift:      float=0.0
class Fault(Flow):
    _init_s = FaultStates

# MODEL PARAMETERS
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
    ub_f :              float=10.0
    lb_f :              float=-1.0
    ub_t :              float=10.0
    lb_t :              float=-1.0
    ub_d :              float=2.0
    lb_d :              float=-2.0
    cor_d :             float=1.0
    cor_t :             float=1.0
    cor_f :             float=1.0
    degradation :       DegParam = DegParam()
    drive_modes :       dict={'mode_args':'set'}
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
        super().__init__(*args, strict_immutability=False, **kwargs)

# MODEL FUNCTIONS
class AvionicsMode(Mode):
    faultparams={'no_con':(1e-4, 200),
                 'crash':(1e-4,200)}
    opermodes = ('drive','standby', 'em_off', 'finished')
    mode : str='standby'
class Avionics(FxnBlock):
    __slots__=('video', 'pos_signal', 'ground', 'control', 'faultstates')
    _init_m = AvionicsMode
    _init_p = RoverParam
    _init_video = Video 
    _init_pos_signal = Pos_Signal
    _init_ground = Ground 
    _init_control = Control
    _init_faultstates = Fault
    flownames={'avionics_control':'control'}
    def dynamic_behavior(self,time):
        if not self.m.in_mode('no_con'):
            if time == 5:   self.m.set_mode('drive')
            if time == 100: self.m.set_mode('standby')

        if self.m.in_mode('drive'):
            self.pos_signal.s.assign(self.video.s, 'angle', 'linex', 'liney')
            self.pos_signal.s.heading = self.ground.s.ang
            self.pos_signal.s.assign(self.ground.s, 'x', 'y', 'vel')

            if in_area(*self.p.end,1,*self.pos_signal.s.get('x','y')):  self.m.set_mode('finished')
            elif self.video.s.quality==0:                               self.m.set_mode('em_off')
            elif not self.faultstates_in_bounds():                      self.m.set_mode('em_off')
            else:
                ycorrection= np.arctan((self.pos_signal.s.y-self.pos_signal.s.liney)/(self.pos_signal.s.vel*np.cos(np.pi/180 * self.pos_signal.s.heading)+0.001))
                xcorrection= np.arctan((self.pos_signal.s.x-self.pos_signal.s.linex)/(self.pos_signal.s.vel*np.sin(np.pi/180 * self.pos_signal.s.heading)+0.001))
                turn_fault_correction = self.p.cor_d*self.faultstates.s.drift
                if self.video.s.quality==0.5: 
                    ang_diff = np.arctan((self.pos_signal.s.y - self.p.end[1])/(self.pos_signal.s.x - self.p.end[0])) - self.pos_signal.s.heading + turn_fault_correction
                else:                       
                    ang_diff = (self.pos_signal.s.angle - self.pos_signal.s.heading + turn_fault_correction -5.5*(xcorrection+ycorrection))
                rdiff = (translate_angle(ang_diff)/180)
                vel_fault_correction = 1 + self.p.cor_f*(self.faultstates.s.friction) + self.p.cor_t*(self.faultstates.s.transfer-1)
                vel_adj = max(0.2, 1- 0.9*abs(rdiff*20)) *vel_fault_correction
                self.control.s.put(rpower = vel_adj*(1+(rdiff)), lpower = vel_adj*(1-(rdiff)))
                self.control.s.limit(rpower=(-1,2), lpower=(-1,2))
        if self.m.in_mode('standby','em_off','finished'):   self.control.s.put(rpower = 0, lpower = 0)
    def faultstates_in_bounds(self):
        return (self.p.lb_f <= self.faultstates.s.friction <= self.p.ub_f and \
                self.p.lb_d <=self.faultstates.s.drift <= self.p.ub_d and \
                    self.p.lb_t<=self.faultstates.s.transfer <= self.p.ub_t)

def translate_angle(angle):
    if angle <-180:      angle = angle % 180
    elif angle > 180:   angle = angle % -180
    return angle

class DriveMode(Mode):
    """ """
    s:          FaultStates=FaultStates()
    mode_args : tuple = tuple()
    faultparams = dict()
    key_phases_by='global'
    def __init__(self, *args, mode_args=tuple(), **kwargs):
        super().__init__(*args, **kwargs)
        if 'mode_args' in mode_args:
            self.mode_args=mode_args['mode_args']
        else: self.mode_args=mode_args
        if self.mode_args=='degradation':
            self.assoc_faultstates({'friction':{(self.s.friction+0.5), 2*(self.s.friction+0.5), 5*(self.s.friction+0.5)}, 
                                    'transfer':{0.0}, 
                                    'drift':{self.s.drift+0.2, self.s.drift-0.2}}, 'all')
        elif type(self.mode_args)==int:
            self.assoc_faultstates({'friction':{*np.linspace(0.0,20, 100)}, 
                                    'transfer':{*np.linspace(1.0,0.0, 100)}, 
                                    'drift': {*np.linspace(-0.5,0.5, 100)}}, self.mode_args)
        elif type(self.mode_args)==list:
            self.assoc_faultstates(manual_modes={'s_'+str(i):{'friction':mode[0], 
                                                              'transfer':mode[1], 
                                                              'drift':mode[2]} for i,mode in enumerate(self.mode_args)})
        elif  type(self.mode_args)==dict:
            self.assoc_faultstates(manual_modes=self.mode_args)
        else:
            if 'manual' in self.mode_args:
                self.assoc_faultstates(manual_modes={'elec_open':{'transfer':0.0}, 
                                                     'stuck':{'friction':10.0}, 
                                                     'stuck_right':{'friction':3.0, 'drift':0.2},
                                                     'stuck_left':{'friction':3.0, 'drift':-0.2}})
            if  'set' in self.mode_args:
                self.assoc_faultstates({'friction':{1.5,3.0,10.0}, 
                                        'transfer':{0.5,0.0}, 
                                        'drift':{-0.2,0.2}}, 'all')
            if 'range' in self.mode_args:
                if 'all' in kwargs['drive_modes']:
                    self.assoc_faultstates({'friction':np.linspace(0.0,20, 10), 
                                            'transfer':np.linspace(1.0,0.0, 10), 
                                            'drift':   np.linspace(-0.5,0.5, 10)}, 'all')
                else:
                    self.assoc_faultstates({'friction':np.linspace(0.0,20, 100), 
                                            'transfer':np.linspace(1.0,0.0, 100), 
                                            'drift':   np.linspace(-0.5,0.5, 100)}, 1000)

class Drive(FxnBlock):
    __slots__=('ground', 'motor_control', 'ee_in', 'faultstates')
    _init_p = DegParam
    _init_m = DriveMode
    _init_faultstates = Fault
    _init_ground = Ground 
    _init_motor_control=Control
    _init_ee_in = EE 
    _init_faultstates = Fault
    flownames = {'ee_15':'ee_in'}
    def dynamic_behavior(self, time):
        self.faultstates.s.assign(self.m.s, 'friction', 'transfer', 'drift')
        rpower = self.m.s.transfer*self.ee_in.s.v*self.motor_control.s.rpower/15 + self.m.s.drift
        lpower = self.m.s.transfer*self.ee_in.s.v*self.motor_control.s.lpower/15 - self.m.s.drift
        if self.m.has_fault("elec_open"):   self.ee_in.s.a = 0
        else:                               self.ee_in.s.a = (1.0+self.m.s.friction)*(lpower + rpower)/12
        if (lpower + rpower) >100: self.add_fault("elec_open")
        else:
            self.ground.s.vel= (rpower + lpower)/(1.0+self.m.s.friction)
            self.ground.s.inc(ang = 180/np.pi*np.arctan((rpower-lpower)/(rpower+lpower +0.001)))
            self.ground.s.ang = translate_angle(self.ground.s.ang)
            self.ground.s.inc(x = np.cos(np.pi/180 *self.ground.s.ang) * self.ground.s.vel, \
                            y = np.sin(np.pi/180 *self.ground.s.ang) * self.ground.s.vel)

class PerceptionMode(Mode):
    faultparams = ('bad_feed',)
    opermodes = ('off', 'feed')
    mode:   str='off'
    exclusive = True
class Perception(FxnBlock):
    __slots__=('ground', 'ee', 'video')
    rad=1
    _init_m = PerceptionMode
    _init_ground = Ground
    _init_ee = EE
    _init_video = Video
    flownames={'ee_12':'ee'}
    def dynamic_behavior(self,time):
        if self.m.in_mode('off'):
            self.ee.s.a=0
            self.video.s.put(linex = 0, liney = 0, angle = 0, quality = 0)
            if self.ee.s.v == 12: self.m.set_mode("feed")
        elif self.m.in_mode("feed"):
            if self.ee.s.v > 8:
                if in_area(*self.ground.s.get('linex','liney'), self.rad, *self.ground.s.get('x','y')):
                    self.video.s.assign(self.ground.s, 'linex','liney', 'angle')
                    self.video.s.quality = 1
                else:
                    self.video.quality=0
            elif self.ee.s.v == 0: self.m.set_mode("off")
        elif self.m.has_fault('bad_feed'): self.video.quality = 0.5

def in_area(x,y,rad,xc,yc):
    dist = np.sqrt((x-xc)**2+(y-yc)**2)
    return not dist > rad
def dist(x,y,xc,yc):
    return np.sqrt((x-xc)**2+(y-yc)**2)

class PowerState(State):
    charge :    float= 100.0
    power :     float=0.0
class PowerMode(Mode):
    faultparams = {"no_charge": (1e-5, {'standby':1.0}, 100),
                   "short":     (1e-5, {'supply':1.0}, 100)}
    opermodes = ("supply","charge","standby","off")
    mode: str='off'
    exclusive = True
    key_phases_by = 'self'
class Power(FxnBlock):
    __slots__=('ee_15','ee_5','ee_12','switch')
    _init_s = PowerState
    _init_m = PowerMode
    _init_ee_15 = EE
    _init_ee_5 = EE
    _init_ee_12 = EE
    _init_switch = Switch
    def static_behavior(self,time):       
        if self.m.in_mode("off"):
            self.ee_5.s.put(v=0, a=0) 
            self.ee_12.s.put(v=0,a=0)
            self.ee_15.s.put(v=0,a=0)
            if self.switch.s.power==1:   
                self.m.set_mode("supply")
        elif self.m.in_mode("supply"):
            if self.s.charge > 0:         
                self.ee_5.s.v = 5
                self.ee_12.s.v = 12 
                self.ee_15.s.v = 15
            else:                           
                self.m.set_mode("no_charge")
            if self.switch.s.power==0:   
                self.m.set_mode("off")
        elif self.m.in_mode("short"):
                self.ee_5.s.v = 5
                self.ee_12.s.v = 12 
                self.ee_15.s.v = 15
        elif self.m.in_mode("no_charge"): 
            self.ee_5.s.v = 0 
            self.ee_12.s.v = 0; 
            self.ee_15.s.v = 0;
        if self.m.in_mode("charge"):
            self.s.power = - 1
            if self.s.charge==100:
                self.m.set_mode("off")
        else:
            self.s.power=1.0+self.ee_12.s.mul('v','a')+self.ee_5.s.mul('v','a')+self.ee_15.s.mul('v','a')
    def dynamic_behavior(self,time):
        self.s.inc(charge = - self.s.power/100)
        self.s.limit(charge=(0,100))

class OverrideMode(Mode):
    opermodes = ('off','standby','override')
    mode:   str = 'off'
class Override(FxnBlock):
    __slots__=('override_comms', 'ee', 'motor_control', 'avionics_control')
    _init_m = OverrideMode
    _init_override_comms = OverrideComms
    _init_ee = EE 
    _init_motor_control = Control 
    _init_avionics_control = Control
    flownames={'ee_5':'ee'}
    def dynamic_behavior(self,time):
        if self.m.in_mode('off'):
            self.ee.s.a=0
            if self.ee.s.v==5: self.m.set_mode('standby')
        elif self.m.in_mode('standby'):
            self.motor_control.s.assign(self.avionics_control.s, 'rpower','lpower')
            if self.override_comms =='active' and self.EE.s.v>4: self.m.set_mode('override')
        elif self.m.in_mode('override'):
            self.motor_control.s.assign(self.override_comms.s, 'rpower', 'lpower')

class Communications(FxnBlock):
    __slots__=('ee_12', 'comms', 'pos_signal')
    _init_ee_12 = EE 
    _init_comms = Comms 
    _init_pos_signal = Pos_Signal
    def dynamic_behavior(self,time):
        if self.ee_12.s.v == 12:
            self.ee_12.s.a=1
            self.comms.s.assign(self.pos_signal.s, 'x', 'y', 'vel', 'heading')
        else:   self.comms.s.put(x=0, y=0, vel=0, heading=0)

class Operator(FxnBlock):
    __slots__=('switch',)
    _init_switch = Switch
    def dynamic_behavior(self, t):
        if t==1:    self.switch.s.power=1
        elif t==200: self.switch.s.power=0

class EnvStates(State):
    in_bound: int=1
class Environment(FxnBlock):
    __slots__=('ground',)
    _init_s = EnvStates
    _init_p = RoverParam
    _init_ground = Ground
    def __init__(self, name, flows, p={}, **kwargs):
        super().__init__(name,flows, **kwargs)
    def dynamic_behavior(self, t):
        if self.p.linetype=='sine':
            self.ground.s.angle = sin_angle_func(self.ground.s.x, self.p.amp, self.p.period)
            self.ground.s.linex,self.ground.s.liney = sin_func(self.ground.s.x,self.ground.s.y, self.p.amp, self.p.period)
        elif self.p.linetype=='turn':
            self.ground.s.angle = turn_angle_func(self.ground.s.x, self.p.radius, self.p.start)
            self.ground.s.linex, self.ground.s.liney = turn_func(self.ground.s.x, self.ground.s.y, self.p.radius, self.p.start)
        self.ground.s.lbx = self.ground.s.linex + 1.5 * np.sin(self.ground.s.angle*np.pi/180)
        self.ground.s.lby = self.ground.s.liney - 1.5 * np.cos(self.ground.s.angle*np.pi/180)
        self.ground.s.ubx = self.ground.s.linex - 1.5 * np.sin(self.ground.s.angle*np.pi/180)
        self.ground.s.uby = self.ground.s.liney + 1.5 * np.cos(self.ground.s.angle*np.pi/180)
        self.s.in_bound = int(in_bounds(self.ground.s.x, self.ground.s.y, self.ground.s.lbx, self.ground.s.lby, self.ground.s.ubx, self.ground.s.uby))
def sin_func(x,y, amp, period):
    return x, amp * np.sin(period*x)
def sin_angle_func(x, amp, period):
    return  np.arctan(amp*period*np.cos(period*x))*180/np.pi
def in_bounds(x,y,lx,ly,ux,uy, tol=0.001):
    l_slope = ly/(lx+.000001)
    u_slope = uy/(ux+.000001)
    pt_slope = y/(x+.000001)
    if l_slope <= pt_slope <= u_slope:              return True
    elif u_slope - l_slope <tol:
        if l_slope-tol <= pt_slope <= u_slope+tol:  return True
        else:                                       return False
    else:                                           return False
def turn_func(x,y, radius,start, buffer=0.1):
    if   x >= start+radius-buffer:  return start+radius, y
    elif y >= radius:               return start+radius, y
    elif x >= start:                return x, radius - np.sqrt(radius**2 - (x-start)**2)
    elif x < start:                 return x, 0
def turn_angle_func(x, radius, start):
    if list in {type(x), type(radius), type(start)}:
        raise Exception("Invalid x,radius,start: "+str(x)+","+str(radius)+","+str(start))
    if   x >= start+radius: return 90.0
    elif x >= start:        return 90 - np.arccos(((x-start)/radius))*180/np.pi  #np.arctan((x-start)/(radius**2-(start-x)**2))*180/np.pi
    elif x<start:           return 0.0


def gen_model_params(x, scen):
    params = {'drive_modes':{'custom_fault':{'friction':x[scen][0][0],'drift':x[scen][0][1], 'transfer':x[scen][0][2]}}}
    return params
class Rover(Model):
    __slots__=()
    _init_p = RoverParam
    default_sp = dict(times=(0, 100), phases=(('start',0, 30), ('end', 31, 60)), end_condition='indicate_finished')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_flow('ground',             Ground, s={'angle':self.p.initangle})
        self.add_flow('pos_signal',         Pos_Signal)
        self.add_flow('ee_12',              EE)
        self.add_flow('ee_5',               EE)
        self.add_flow('ee_15',              EE)
        self.add_flow('video',              Video)
        self.add_flow('avionics_control',   Control)
        self.add_flow('motor_control',      Control)
        self.add_flow('switch',             Switch)
        self.add_flow('comms',              Comms)
        self.add_flow('override_comms',     OverrideComms)
        self.add_flow('faultstates',        Fault)
        #self.add_flow('Example_Disconnect')

        self.add_fxn("power",           Power,          "ee_15","ee_5",'ee_12', "switch")
        self.add_fxn("operator",        Operator,       "switch")
        self.add_fxn("communications",  Communications, "comms", "ee_12", 'pos_signal')
        self.add_fxn("perception",      Perception,     "ground", "ee_12", "video")
        self.add_fxn("avionics",        Avionics,       "video",'pos_signal',"ground","avionics_control", "faultstates",
                     p=self.p)
        self.add_fxn("override",        Override,       "override_comms", "ee_5", 'motor_control','avionics_control')
        self.add_fxn("drive",           Drive,          "ground","ee_15", "motor_control", "faultstates",
                     m={'mode_args': self.p.drive_modes},     p=self.p.degradation)
        self.add_fxn("environment",     Environment,    'ground',   p=self.p)

        self.build()
    def indicate_finished(self, time):
        if (in_area(self.flows['ground'].s.x,self.flows['ground'].s.y,1,self.p.end[0],self.p.end[1]) or \
            (time > 5 and self.fxns['avionics'].m.in_mode('standby')) or \
                self.fxns['avionics'].m.in_mode('em_off', 'finished')):
            return True
        else:
            return False
    def find_classification(self,scen,mdlhist):
        modes, modeproperties = self.return_faultmodes()
        classification = str()
        at_finish=True
        if not in_area(self.flows['ground'].s.x,self.flows['ground'].s.y,2,self.p.end[0],self.p.end[1]):
                                classification = "incomplete mission"
                                at_finish = False
        if any(modes):          classification = classification +' faulty'
        if not classification:  classification = 'nominal mission'
        num_modes = len(modes)
        end_dist = dist(self.flows['ground'].s.x,self.flows['ground'].s.y,self.p.end[0],self.p.end[1])
        endpt=[self.flows['ground'].s.x ,self.flows['ground'].s.y]

        f_t= min(len(mdlhist.faulty.flows.ground.s.x),len(mdlhist.nominal.flows.ground.s.y))

        tot_deviation = np.sum(np.sqrt((mdlhist.nominal.flows.ground.s.x[:f_t]-mdlhist.faulty.flows.ground.s.x[:f_t])**2 + 
                                       (mdlhist.nominal.flows.ground.s.y[:f_t]-mdlhist.faulty.flows.ground.s.y[:f_t])**2))
        in_bound = all(mdlhist.faulty.fxns.environment.s.in_bound)
        line_dist = find_line_dist(self.flows['ground'].s.x,self.flows['ground'].s.y, mdlhist.nominal.flows.ground.s.linex, mdlhist.nominal.flows.ground.s.liney)

        return {'rate':0,'cost':0, 'prob':scen['properties'].get('prob',1), 
                'expected cost':0, 'in_bound':in_bound, 'at_finish':at_finish, 
                'line_dist':line_dist, 'num_modes':num_modes, 'end_dist':end_dist, 
                'tot_deviation':tot_deviation, 'faults':modes, 'classification':classification, 
                'endpt':endpt}

import fmdtools.analyze as an

def find_line_dist(x, y , linex, liney):
    return np.min(np.sqrt((linex-x)**2 + (liney-y)**2))

def gen_param_space():
    paramspace=[]
    ranges = [x for x in itertools.product(np.arange(0, 10, 0.2), range(10,50,10))]
    for r in ranges:
        params = RoverParam(linetype='sine',amp=r[0],wavelength=r[1])
        paramspace.append(params)
    ranges = [x for x in itertools.product(range(5,40,5), range(0, 5,20))]
    for r in ranges:
        params = RoverParam(linetype='turn',radius=r[0],start=r[1])
        paramspace.append(params)
    return paramspace

def plot_course(hist, label=True, ax=False):
    if not ax:
        fig,ax = plt.subplots()
        ax=fig.axes[0]
    if label==True: nom_lab="Nominal"; bound_lab="Bounds"; center_lab="Center-line"
    else:           nom_lab='_nolegend_'; bound_lab='_nolegend_'; center_lab='_nolegend_'

    ax.plot(hist.flows.ground.s.x,hist.flows.ground.s.y, color='blue')
    ax.scatter(hist.flows.ground.s.x[-1],hist.flows.ground.s.y[-1], color='blue', marker='*', label=nom_lab)

    x_ground = hist.flows.ground.s.lbx
    y_ground = hist.flows.ground.s.lby
    ax.plot(x_ground,y_ground, label=bound_lab, color='grey')
    x_ground = hist.flows.ground.s.ubx
    y_ground = hist.flows.ground.s.uby
    ax.plot(x_ground,y_ground, label=bound_lab, color='grey')
    x_ground = hist.flows.ground.s.x
    y_ground = hist.flows.ground.s.liney
    ax.plot(x_ground,y_ground, label=center_lab, color='grey', linestyle='--')

def plot_trajectories(mdlhists, nomhist=[],  app= [], faultlabel='Faulty', faultalpha=0.1, range_hist={}, rangealpha=0.1, setalpha=0.3, show_labels=True, title="Fault Trajectories", textoffset=2.0,mode_trunc=5,mode_trunc_end=5, xlim=None, ylim=None, figsize=(4,4), ax=False, legend=True):
    in_mdlhists=False
    if not ax:
        fig,ax = plt.subplots(figsize=figsize)
        ax=fig.axes[0]
    else: fig=ax.get_figure()
    for mode, hist in range_hist.items():
        if mode[6:11]=='hmode':
            ax.plot(hist.flows.ground.s.x,hist.flows.ground.s.y, color='yellow', alpha=rangealpha)
            ax.scatter(hist.flows.ground.s.x[-1],hist.flows.ground.s.y[-1], color='yellow', alpha=rangealpha, marker='o', label='Range')
    for mode, hist in mdlhists.items():
        if mode=='nominal':
            plot_course(hist, ax=ax)
            in_mdlhists=True
        elif mode[6:11]=='hmode':
            ax.plot(hist.flows.ground.s.x,hist.flows.ground.s.y, color='orange', alpha=setalpha)
            ax.scatter(hist.flows.ground.s.x[-1],hist.flows.ground.s.y[-1], color='orange', alpha=setalpha, marker='o', label='Set')
        else:
            ax.plot(hist.flows.ground.s.x,hist.flows.ground.s.y, color='red', alpha=faultalpha)
            ax.scatter(hist.flows.ground.s.x[-1],hist.flows.ground.s.y[-1], color='red', alpha=faultalpha, marker='*', label=faultlabel)
            if show_labels:
                label = mode[mode_trunc:]
                label = label[:-mode_trunc_end]
                randang = np.pi*np.random.rand()
                ax.annotate(label, xy=(hist.flows.ground.s.x[-1],hist.flows.ground.s.y[-1]), fontsize=8, xytext=(textoffset*np.sin(randang), textoffset*np.cos(randang)), textcoords='offset points')
    if app: ax.scatter(hist.flows.ground.s.x[int(app.times[0])-1],hist.flows.ground.s.y[int(app.times[0])-1], color='black', marker='X', s=5, label='fault time')
    if not in_mdlhists and nomhist: plot_course(nomhist, ax=ax)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    ax.set_title(title)
    ax.set_xlabel("x-distance (m)")
    ax.set_ylabel("y-distance (m)")
    ax.grid()
    return fig

def compare_trajectories(mdlhist1, mdlhist2, mdlhist1_name='fault trajectories', mdlhist2_name='comparison trajectories', faulttimes = [], nomhist=[]):
    for mode, hist in mdlhist1.items():
        plt.plot(hist.flows.ground.s.x,hist.flows.ground.s.y, color='grey', alpha=0.2, zorder=1)
        plt.scatter(hist.flows.ground.s.x[-1],hist.flows.ground.s.y[-1], color='grey', alpha=0.3, marker='o', label=mdlhist1_name, zorder=2)
    for mode, hist in mdlhist2.items():
        plt.plot(hist.flows.ground.s.x,hist.flows.ground.s.y, color='tab:orange', alpha=0.2, zorder=1)
        plt.scatter(hist.flows.ground.s.x[-1],hist.flows.ground.s.y[-1], color='tab:orange', alpha=0.3, marker='o', label=mdlhist2_name, zorder=2)
    if faulttimes:
        xfaults = [x for ind, x in enumerate(nomhist.flows.ground.s.x) if ind+1<len(nomhist.time) and nomhist.time[ind+1] in faulttimes]
        yfaults = [y for ind, y in enumerate(nomhist.flows.ground.s.y) if ind+1<len(nomhist.time) and nomhist.time[ind+1] in faulttimes]
        plt.scatter(xfaults, yfaults, marker = 'x', color='black',  label='fault times', zorder=3)
    if nomhist: plot_course(nomhist)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Fault Trajectories")
    plt.xlabel("x-distance (m)")
    plt.ylabel("y-distance(m)")
    plt.grid()
    return plt.gcf()

def plot_map(mdl, mdlhist):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(0,0, label="Start Location", marker = 's', color='grey')
    plt.scatter(mdl.p.end[0],mdl.p.end[1], label="End Location", marker='X', color='grey')

    plot_course(mdlhist, ax=ax)

    plt.xlabel("x-distance (meters)")
    plt.ylabel("y-distance (meters)")
    plt.title("Rover Centerline Tracking")
    plt.grid()

    plt.legend()

def plot_centerline_err(mdl, mdlhist):
    fig = plt.figure()
    x_rover = mdlhist.flows.ground.s.x
    y_rover = mdlhist.flows.ground.s.y
    if mdl.p['linetype']=='sine':
        y_line = [sin_func(x,y_rover[i], mdl.p['amp'], mdl.p['period'])[1] for i,x in enumerate(x_rover)]
    elif mdl.p['linetype']=='turn':
        y_line = [turn_func(x,y_rover[i], mdl.p['radius'], mdl.p['start'])[1] for i,x in enumerate(x_rover)]

    plt.plot(x_rover, y_rover-y_line)
    plt.xlabel("x-distance (meters)")
    plt.ylabel("y-error (meters)")
    plt.title("Rover Centerline Error")

class_tree = {'rover': [-0.07367185100835244, 0.6410710936138487],
 'drive': [0.23684893559545173, 0.36072825626681254],
 'environment': [-0.5104502883293205, 0.38994116849132504],
 'avionics': [-0.20614969706316938, 0.3834687620416293],
 'override': [0.03237968031217919, 0.3759790214222677],
 'perception': [-0.8350044759810874, 0.40176785861898856],
 'power': [0.4699926035926743, 0.35927158639441575],
 'operator': [0.6783281961222181, 0.4006550001329594],
 'communications': [1.0406903861169736, 0.37725113274488953],
 'motor_control': [0.02456213706952065, -0.17456190594455168],
 'ee_5': [0.20244547912639976, 0.06448327387605401],
 'override_comms': [0.1966879650681026, -0.4239713648342337],
 'ee_15': [0.3596916288553037, 0.1191691659491266],
 'comms': [0.7930664677006409, 0.2234018617977897],
 'ground': [-0.5636731555210449, -0.024506034628188678],
 'avionics_control': [-0.22848556112602958, -0.0821525471741435],
 'pos_signal': [0.4189018653841845, -0.27023016168716785],
 'control': [0.6801636335956871, 0.08587826440540502],
 'video': [-0.6095250803472407, -0.32199506812858303],
 'ee_12': [0.6762136362997685, -0.26803553393104507]}
pos_bip = {'power': [-0.684772948203272, -0.2551613615446115],
         'operator': [-0.798933011500376, 0.565156755693186],
         'communications': [-0.5566050878414673, 0.14159180700630447],
         'perception': [0.996672509613648, 0.2507215448302319],
         'avionics': [0.28027473355741117, 0.47255264233968597],
         'override': [0.28987624783062627, -0.17144760874154652],
         'drive': [0.6671719569482308, -0.571646956655247],
         'environment': [1.1329643169383754, -0.6375225566564033],
         'ground': [1.108432946123935, -0.3228541151507237],
         'pos_signal': [-0.256557435572734, 0.5411037985681082],
         'faultstates': [0.75997843863482324, -0.04522869632581994],
         'ee_12': [-0.3676879520509888, -0.04754907961317867],
         'ee_5': [-0.2181352416728437, -0.2015320865756482],
         'ee_15': [-0.5352906801304353, -0.5288715575154177],
         'video': [0.6726175830840695, 0.396008366729458],
         'avionics_control': [0.45997843863482324, 0.04522869632581994],
         'motor_control': [0.6350063940085445, -0.3013633829278297],
         'switch': [-0.9857988678463686, 0.07960895587242012],
         'comms': [-0.642370284813957, 0.35285736707043763],
         'override_comms': [-0.14607433032593392, 0.2981956996230818]}

if __name__=="__main__":
    import multiprocessing as mp
    
    from fmdtools.sim import search
    
    from pymoo.optimize import minimize
    
    from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
    import numpy as np
    mdl = Rover(sp=SimParam(times=(0, 100), phases=(('start',0, 30), ('end', 31, 60))))
    track={'functions':{"Environment":"in_bound"},'flows':{"ground":"all"}}
    rover_prob = search.ProblemInterface("rover_problem", mdl, pool=mp.Pool(5), staged=True, track=track)
    app_drive = SampleApproach(mdl, faults='drive', phases={'global':[0,39]}, defaultsamp={'samp':'evenspacing','numpts':3})
    rover_prob.add_simulation("drive_faults", "multi", app_drive.scenlist)
    rover_prob.add_variables("drive_faults", ("cor_f", (-10,100)), ("cor_d", (-100, 100)), ("cor_t", (-10,100)), vartype='param')
    rover_prob.add_objectives("drive_faults", end_dist="end_dist", tot_deviation="tot_deviation")
    
    p = RoverParam(linetype="turn")
    
    pymoo_prob = rover_prob.to_pymoo_problem(objectives="end_dist")
    algorithm=PatternSearch(x0=np.array([0.0,0.0,0.0])) 
    #res = minimize(pymoo_prob, algorithm, verbose=True)

    #dot = an.graph.show(mdl, gtype="bipartite", renderer='graphviz')

    mdl = Rover(p=p)

    #mdl = Rover(p)
    endresults,  mdlhist = prop.nominal(mdl)
    phases, modephases = mdlhist.get_modephases()
    plot_map(mdl, mdlhist)

    mdl_id = Rover(p={'drive_modes':{'mode_args':'set'}})
    app_id = SampleApproach(mdl_id, faults='drive', phases={'drive':phases['avionics']['drive']}, defaultsamp={'samp':'evenspacing', 'numpts':3})
    endclasses_id, mdlhists_id = prop.approach(mdl_id, app_id)  #pool=mp.Pool(4))

    #behave_endclasses_nested, behave_mdlhists_nested = prop.nested_approach(mdl, behave_nomapp, pool=mp.Pool(5), faults='drive')

    #res_comp = an.tabulate.resilience_factor_comparison(behave_nomapp, behave_endclasses_nested, ['t'], 'at_finish', percent=False)
    #fig = an.plot.resilience_factor_comparison(res_comp, stack=True)

    #endresults,  mdlhist = prop.one_fault(mdl, 'drive','hmode_34', time=1, staged=False)
    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift']})

    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift', 'transfer']})

    mdl = Rover(p=p.copy_with_vals(drive_modes={'mode_args':'manual'}))
    endresults,  mdlhist = prop.one_fault(mdl, 'drive','elec_open', time=1, staged=False)
    
    mdl = Rover(p=p.copy_with_vals(drive_modes={'mode_args':100}))
    endresults,  mdlhist = prop.one_fault(mdl, 'drive','hmode_34', time=1, staged=False)
    
    
    x = [1.0,0.0,1.0]
    mdl = Rover(p=p.copy_with_vals(drive_modes={'custom_fault':{'friction':x[0],'drift':x[1], 'transfer':x[2]}}))

    _, mdlhist = prop.nominal(mdl)

    endresults, reshist = prop.one_fault(mdl,'drive','custom_fault', time=15, staged=True)
    
    
    line_dist = endresults['line_dist']
    end_loc = (reshist.faulty.flows.ground.s.x[-1],reshist.faulty.flows.ground.s.y[-1])
    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift']})

   # app = NominalApproach()
   # app.add_param_ranges(gen_model_p, 'app', x, scen = (0,len(scen)-1,1))
   # endclasses, mdlhists = prop.approach(mdl, app)

    #plot_trajectories(mdlhist, app=app, faultlabel='Faulty Scenarios')


    #fig, ax =an.graph.show(mdl, gtype='bipartite')
    #fig.savefig('bipartite_rover.pdf', format="pdf", bbox_inches = 'tight', pad_inches = 0.0)

    """
    mdl = Rover(p=gen_p('turn'))
    #dot = an.graph.show(mdl, gtype="bipartite", renderer='graphviz')
    #p = gen_p('sine')
    #mdl = Rover(p)
    endresults,  mdlhist = prop.nominal(mdl)
    phases, modephases = mdlhist.get_modephases()
    plot_map(mdl, mdlhist)

    mdl_id = Rover(valp={'drive_modes':'else'})
    app_id = SampleApproach(mdl_id, faults='drive', phases={'drive':phases['avionics']['drive']})
    endclasses_id, mdlhists_id = prop.approach(mdl_id, app_id, staged=True)


    p = gen_p('sine',cor_d=-180, cor_f=1)
    mdl_thing = Rover(p=p)
    _,_, reshist = prop.one_fault(mdl,'drive','stuck_right', time=15, staged=True)
    plt.figure()
    f = plot_trajectories({'nominal':mdlhist}, reshist,  faultalpha=0.6)

    an.plot.mdlhistvals(reshist, time=15, fxnflowvals={'drive':['friction','drift', 'transfer'], 'power':'all'})

    app_opt = SampleApproach(mdl, faults='drive', phases={'drive':phases['avionics']['drive']}, defaultsamp={'samp':'evenspacing','numpts':4})

    #endresults,  mdlhist = prop.one_fault(mdl, 'drive','elec_open', time=1, staged=False)
    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift', 'transfer']})

    #endresults,  mdlhist = prop.one_fault(mdl, 'drive','hmode_34', time=1, staged=False)
    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift']})


    x = [100,0,2,0,2,-2,0,0,0]
    x_p = gen_p('sine', ub_f=x[0], lb_f=x[1], ub_t=x[2],lb_t=x[3], ub_d=x[4], lb_d=x[5], cor_f=x[6], cor_d=x[7], cor_t=x[8])
    mdl_0 = Rover(p=x_p)


    _,_, nomhist = prop.nominal(mdl_0)
    phases, modephases = mdlhist.get_modephases()
    app_0 = SampleApproach(mdl, faults='drive', phases={'drive':phases['avionics']['drive']}, defaultsamp={'samp':'evenspacing','numpts':4})
    endclasses_0, mdlhists_0 = prop.approach(mdl_0, app_0, staged=True)
    plt.figure()
    f = plot_trajectories(mdlhists_0, app=app_0,  faultalpha=0.6)


    x = [100,0,2,0,2,-2,1,-180,1]
    x_p = gen_p('sine', ub_f=x[0], lb_f=x[1], ub_t=x[2],lb_t=x[3], ub_d=x[4], lb_d=x[5], cor_f=x[6], cor_d=x[7], cor_t=x[8])
    mdl_1 = Rover(p=x_p)
    endclasses_1, mdlhists_1 = prop.approach(mdl_1, app_0, staged=True)

    #compare_trajectories(mdlhists_0, mdlhists_1, mdlhist1_name='fault trajectories', mdlhist2_name='comparison trajectories', faulttimes = app.times, nomhist=nomhist)


    #an.graph.show(  scale=0.7)

    #an.plot.mdlhistvals(mdlhist, legend=False)
    #an.plot.mdlhistvals(mdlhist)

    plot_map(mdl, mdlhist)

    #endresults,  mdlhist = prop.one_fault(mdl, 'drive','elec_open', staged=True, time=13, gtype='typegraph')
    endresults,  mdlhist_feed = prop.one_fault(mdl, 'perception', 'bad_feed', staged=True, time=7, gtype='typegraph')
    plot_trajectories(mdlhist_feed, mdlhist, faultalpha=1.0)


    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'power':['charge','power']}, time=7, phases=phases, modephases=modephases)
    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'ground':['x','y', 'angle','vel', 'liney', 'ang']}, time=7, phases=phases)
    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'pos_signal':['x','y', 'angle','vel', 'heading']}, time=7, phases=phases)
    #an.plot.mdlhistvals(mdlhist, fxnflowvals={'motor_control':['rpower','lpower']}, time=7, phases=phases)
    an.plot.mdlhistvals(mdlhist, fxnflowvals={'avionics':['mode']}, time = 13, phases=phases, modephases=modephases)
    an.plot.mdlhistvals(mdlhist, fxnflowvals={'perception':['mode']}, time = 13, phases=phases, modephases=modephases)
    #an.plot.mdlhistvals(mdlhist, fxnflowvals={}, time = 7, phases=phases, modephases=modephases)
    app = NominalApproach()
    app.add_param_ranges(gen_p,'sine', 'sine', amp=(0, 10, 0.2), wavelength=(10,50,10))
    app.assoc_probs('sine', amp=(stats.uniform.pdf, {'loc':0,'scale':10}), wavelength=(stats.uniform.pdf,{'loc':10, 'scale':40}))
    #app.add_param_ranges(gen_p,'turn', radius=(5,40,5), start=(0, 20,5))

    #labels, faultfxns, degnodes, faultlabels
#    an.graph.plot_bipgraph(classgraph, {node:node for node in classgraph.nodes},[],[],{}, pos=pos)
    an.graph.show( gtype='typegraph', scale=0.7)

    #endresults,  mdlhist = prop.one_fault(mdl, 'drive','elec_open', staged=True, time=13, gtype='bipartite')
    endresults,  mdlhist = prop.one_fault(mdl, 'perception', 'bad_feed', staged=True, time=13, gtype='bipartite')
    an.graph.show( gtype='bipartite', scale=0.7)

    reshist, _, _ = an.process.hist(mdlhist)
    typehist = an.process.typehist(mdl, reshist)
    an.graph.results_from(mdl, reshist, [10,15,20])
    an.graph.results_from(mdl, typehist, [10,15,20], gtype='typegraph') #), gtype='typegraph')
    an.graph.result_from(mdl, reshist, 10, gtype='bipartite', renderer='graphviz')

    #endclasses, mdlhists= prop.nominal_approach(mdl, app, pool = mp.Pool(5))

    #fig = an.plot.nominal_vals_1d(app, endclasses, 'amp')
    #fig = an.plot.nominal_vals_1d(app, endclasses, 'radius')

    #app = NominalApproach()
    #app.add_param_ranges(gen_p,'sine','sine', amp=(0, 10, 0.2), wavelength=(10,50,10), dummy=(1,10,1))

    #endclasses, mdlhists= prop.nominal_approach(mdl, app, pool = mp.Pool(5))
    #fig = an.plot.nominal_vals_3d(app, endclasses, 'amp', 'wavelength', 'dummy')
    #app = SampleApproach(mdl, phases = phases, modephases = modephases)

    #endclasses, mdlhist = prop.approach(mdl, app)

    #app_joint = SampleApproach(mdl, phases = phases, modephases = modephases, jointfaults={'faults':2})

    #endclasses, mdlhist = prop.approach(mdl, app_joint)


    #tab = an.tabulate.phasefmea(endclasses, app_joint)
    #an.plot.samplecosts(app_joint, endclasses)

    #an.plot.phases(phases)

    #figs = an.plot.phases(phases, modephases, mdl)
    #figs = an.plot.phases(phases, modephases, mdl, singleplot=False)
    """
    