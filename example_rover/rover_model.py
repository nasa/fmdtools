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
    - Ground
    - Force
    - EE
    - Camera
"""
from fmdtools.modeldef.common import Parameter
from fmdtools.modeldef.block import FxnBlock
from fmdtools.modeldef.model import Model, ModelParam
from fmdtools.modeldef.approach import SampleApproach, NominalApproach
import fmdtools.resultdisp as rd
import fmdtools.faultsim.propagate as prop
import itertools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

class Avionics(FxnBlock):
    def __init__(self,name, flows, params):
        self.p=params
        super().__init__(name, flows, flownames={'AvionicsControl':'Control'})
        self.assoc_modes({'no_con':[1e-4, 200],'CRASH':[1e-4,200]}, ['drive','standby', 'em_off', 'finished'], initmode='standby')
    def dynamic_behavior(self,time):
        if not self.in_mode('no_con'):
            if time == 5: self.set_mode('drive')
            if time == 100: self.set_mode('standby')

        if self.in_mode('drive'):
            self.Pos_Signal.assign(self.Video, 'angle', 'linex', 'liney')
            self.Pos_Signal.heading = self.Ground.ang
            self.Pos_Signal.assign(self.Ground, 'x', 'y', 'vel')

            if in_area(self.p.end[0],self.p.end[1],1,self.Pos_Signal.x,self.Pos_Signal.y):  self.set_mode('finished')
            elif self.Video.quality==0:                                                 self.set_mode('em_off')
            elif not self.faultstates_in_bounds():                                     self.set_mode('em_off')
            else:
                ycorrection= np.arctan((self.Pos_Signal.y-self.Pos_Signal.liney)/(self.Pos_Signal.vel*np.cos(np.pi/180 * self.Pos_Signal.heading)+0.001))
                xcorrection= np.arctan((self.Pos_Signal.x-self.Pos_Signal.linex)/(self.Pos_Signal.vel*np.sin(np.pi/180 * self.Pos_Signal.heading)+0.001))
                turn_fault_correction = self.p.cor_d*self.Faultstates.drift
                if self.Video.quality==0.5: ang_diff = np.arctan((self.Pos_Signal.y - self.end[1])/(self.Pos_Signal.x - self.end[0])) - self.Pos_Signal.heading + turn_fault_correction
                else:                       ang_diff = (self.Pos_Signal.angle - self.Pos_Signal.heading + turn_fault_correction -5.5*(xcorrection+ycorrection))
                rdiff = (translate_angle(ang_diff)/180)
                vel_fault_correction = 1 + self.p.cor_f*(self.Faultstates.friction) + self.p.cor_t*(self.Faultstates.transfer-1)
                vel_adj = max(0.2, 1- 0.9*abs(rdiff*20)) *vel_fault_correction
                self.Control.put(rpower = vel_adj*(1+(rdiff)), lpower = vel_adj*(1-(rdiff)))
                self.Control.limit(rpower=(-1,2), lpower=(-1,2))
        if self.in_mode('standby','em_off','finished'):   self.Control.put(rpower = 0, lpower = 0)
    def faultstates_in_bounds(self):
        return (self.p.lb_f <= self.Faultstates.friction <= self.p.ub_f and \
                self.p.lb_d <=self.Faultstates.drift <= self.p.ub_d and \
                    self.p.lb_t<=self.Faultstates.transfer <= self.p.ub_t)

def translate_angle(angle):
    if angle <-180:      angle = angle % 180
    elif angle > 180:   angle = angle % -180
    return angle


class Drive(FxnBlock):
    def __init__(self,name, flows, params):
        super().__init__(name, flows, flownames={"EE_15":"EE_in"})
        self.p=params[0]
        modeparams = params[1]
        if modeparams['drive_modes']=='degradation':
            base_f, base_d = self.p.friction, self.p.drift
            self.assoc_faultstates({'friction':[base_f, {(base_f+0.5), 2*(base_f+0.5), 5*(base_f+0.5)}], 'transfer':[1.0,{0.0}], 'drift':[base_d, {base_d+0.2, base_d-0.2}]}, 'all')
        elif type(modeparams['drive_modes'])==int:
            self.assoc_faultstates({'friction':[0.0, np.linspace(0.0,20, 100)], 'transfer':np.linspace(1.0,0.0, 100), 'drift':[0.0, np.linspace(-0.5,0.5, 100)]}, params['drive_modes'])
        elif type(modeparams['drive_modes'])==list:
            self.assoc_faultstates({'friction':0.0, 'transfer':1.0, 'drift':0.0}, 'none')
            manual_modes ={'s_'+str(i):{'friction':mode[0], 'transfer':mode[1], 'drift':mode[2]} for i,mode in enumerate(modeparams['drive_modes'])}
            self.assoc_faultstate_modes(manual_modes=manual_modes)
        elif  'set' in modeparams['drive_modes']:
            self.assoc_faultstates({'friction':[0.0, {1.5,3.0,10.0}], 'transfer':{0.5,0.0}, 'drift':[0.0, {-0.2,0.2}]}, 'all')
            if 'manual' in modeparams['drive_modes']:
                self.assoc_faultstate_modes(manual_modes={'elec_open':{'transfer':0.0}, 'stuck':{'friction':10.0}, 'stuck_right':{'friction':3.0, 'drift':0.2},
                                                           'stuck_left':{'friction':3.0, 'drift':-0.2}})
        elif 'range' in modeparams['drive_modes']:
            if 'all' in modeparams['drive_modes']:
                self.assoc_faultstates({'friction':[0.0,np.linspace(0.0,20, 10)], 'transfer':np.linspace(1.0,0.0, 10), 'drift':[0.0, np.linspace(-0.5,0.5, 10)]}, 'all')
            else:
                self.assoc_faultstates({'friction':[0.0,np.linspace(0.0,20, 100)], 'transfer':np.linspace(1.0,0.0, 100), 'drift':[0.0, np.linspace(-0.5,0.5, 100)]}, 1000)
            if 'manual' in modeparams['drive_modes']:
                self.assoc_faultstate_modes(manual_modes={'elec_open':{'transfer':0.0}, 'stuck':{'friction':10.0}, 'stuck_right':{'friction':3.0, 'drift':0.2},
                                                           'stuck_left':{'friction':3.0, 'drift':-0.2}})
        elif modeparams['drive_modes']=='manual':
            self.assoc_faultstates({'friction':0.0, 'transfer':1.0, 'drift':0.0}, 'none')
            self.assoc_faultstate_modes(manual_modes={'elec_open':{'transfer':0.0}, 'stuck':{'friction':10.0}, 'stuck_right':{'friction':3.0, 'drift':0.2},
                                                       'stuck_left':{'friction':3.0, 'drift':-0.2}})
        elif  type(modeparams['drive_modes'])==dict:
            self.assoc_faultstates({'friction':0.0, 'transfer':1.0, 'drift':0.0}, 'none')
            self.assoc_faultstate_modes(manual_modes=modeparams['drive_modes'])
        self.key_phases_by='global'
    def dynamic_behavior(self, time):
        self.Faultstates.assign(self, 'friction', 'transfer', 'drift')
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

class Perception(FxnBlock):
    def __init__(self, name, flows):
        self.set_atts(rad=1)
        super().__init__(name, flows, flownames={'EE_12':'EE'})
        self.assoc_modes({'bad_feed'}, ['off', 'feed'], initmode='off', exclusive=True)
    def dynamic_behavior(self,time):
        if self.in_mode('off'):
            self.EE.a=0
            self.Video.put(linex = 0, liney = 0, angle = 0, quality = 0)
            if self.EE.v == 12: self.set_mode("feed")
        elif self.in_mode("feed"):
            if self.EE.v > 8:
                if in_area(self.Ground.linex, self.Ground.liney, self.rad, self.Ground.x, self.Ground.y):
                    self.Video.assign(self.Ground, 'linex','liney', 'angle')
                    self.Video.quality = 1
                else:
                    self.Video.quality=0
            elif self.EE.v == 0: self.set_mode("off")
        elif self.has_fault('bad_feed'): self.Video.quality = 0.5

def in_area(x,y,rad,xc,yc):
    dist = np.sqrt((x-xc)**2+(y-yc)**2)
    return not dist > rad
def dist(x,y,xc,yc):
    return np.sqrt((x-xc)**2+(y-yc)**2)


class Power(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name,flows, states={"charge": 100.0, 'power':0.0})
        self.assoc_modes({"no_charge":[1e-5, {'standby':1.0}, 100],"short":[1e-5, {'supply':1.0}, 100],}, ["supply","charge","standby","off"], initmode="off", exclusive = True, key_phases_by='self')
    def static_behavior(self,time):
        if self.in_mode("off"):
            self.EE_5.put(v=0, a=0); self.EE_12.put(v=0,a=0);self.EE_15.put(v=0,a=0)
            if self.Control.power==1:   self.set_mode("supply")
        elif self.in_mode("supply"):
            if self.charge > 0:         self.EE_5.v = 5; self.EE_12.v = 12; self.EE_15.v = 15;
            else:                       self.set_mode("no_charge")
            if self.Control.power==0:   self.set_mode("off")
        elif self.in_mode("short"):     self.EE_5.v = 5; self.EE_12.v = 12; self.EE_15.v = 15;
        elif self.in_mode("no_charge"): self.EE_5.v = 0; self.EE_12.v = 0; self.EE_15.v = 0;
        if self.in_mode("charge"):
            self.power = - 1
            if self.charge==100:self.set_mode("off")
        else:
            self.power=1.0+self.EE_12.mul('v','a')+self.EE_5.mul('v','a')+self.EE_15.mul('v','a')
    def dynamic_behavior(self,time):
        self.inc(charge = - self.power/100)
        self.limit(charge=(0,100))

class Override(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows, flownames={'EE_5':'EE'})
        self.assoc_modes({}, ['off','standby','override'], initmode = 'off')
    def dynamic_behavior(self,time):
        if self.in_mode('off'):
            self.EE.a=0
            if self.EE.v==5: self.set_mode('standby')
        elif self.in_mode('standby'):
            self.MotorControl.assign(self.AvionicsControl, 'rpower','lpower')
            if self.OverrideComms =='active' and self.EE.v>4: self.set_mode('override')
        elif self.in_mode('override'):
            self.MotorControl.assign(self.OverrideComms, 'rpower', 'lpower')

class Communications(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name,flows)
    def dynamic_behavior(self,time):
        if self.EE_12.v == 12:
            self.EE_12.a=1
            self.Comms.assign(self.Pos_Signal, 'x', 'y', 'vel', 'heading')
        else:   self.Comms.put(x=0, y=0, vel=0, heading=0)

class Operator(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name,flows)
    def dynamic_behavior(self, t):
        if t==1:    self.Control.power=1
        elif t==200: self.Control.power=0

class Environment(FxnBlock):
    def __init__(self, name, flows, params):
        self.p=params
        super().__init__(name,flows, states={'in_bound':1})
        self.assoc_modes({'obstacle'},['clear'], initmode='clear', exclusive=True)
    def dynamic_behavior(self, t):
        if self.p.linetype=='sine':
            self.Ground.angle = sin_angle_func(self.Ground.x, self.p.amp, self.p.period)
            self.Ground.linex,self.Ground.liney = sin_func(self.Ground.x,self.Ground.y, self.p.amp, self.p.period)
        elif self.p.linetype=='turn':
            self.Ground.angle = turn_angle_func(self.Ground.x, self.p.radius, self.p.start)
            self.Ground.linex, self.Ground.liney = turn_func(self.Ground.x, self.Ground.y, self.p.radius, self.p.start)
        if self.has_fault('obstacle'):
            self.Obstacle.r = 1
            self.Obstacle.x = self.Ground.x
            self.Obstacle.y = self.Ground.y
        self.Ground.lbx = self.Ground.linex + 1.5 * np.sin(self.Ground.angle*np.pi/180)
        self.Ground.lby = self.Ground.liney - 1.5 * np.cos(self.Ground.angle*np.pi/180)
        self.Ground.ubx = self.Ground.linex - 1.5 * np.sin(self.Ground.angle*np.pi/180)
        self.Ground.uby = self.Ground.liney + 1.5 * np.cos(self.Ground.angle*np.pi/180)
        self.in_bound = int(in_bounds(self.Ground.x, self.Ground.y, self.Ground.lbx, self.Ground.lby, self.Ground.ubx, self.Ground.uby))
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

class DegParam(Parameter, readonly=True):
    """Parameters for rover degradation"""
    friction :          float = 0.0
    drift :             float = 0.0

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


def gen_model_params(x, scen):
    params = {'drive_modes':{'custom_fault':{'friction':x[scen][0][0],'drift':x[scen][0][1], 'transfer':x[scen][0][2]}}}
    return params
class Rover(Model):
    def __init__(self, params=RoverParam(),\
                 modelparams=ModelParam(times=(0, 100), phases=(('start',0, 30), ('end', 31, 60))),\
                     valparams={'drive_modes':'set'}):
        super().__init__(params, modelparams, valparams)

        self.add_flow('Ground', {'x':0.0,'y':0.0,'liney':0.0,'linex':0.0, 'lbx':0.0, 'lby':-1.5, 'ubx':0.0,'uby':1.5, 'vel':0.0, 'line':0.0, 'angle':params.initangle, 'ang':0.0})
        self.add_flow('Pos_Signal', {'x':0.0,'y':0.0,'liney':0.0,'linex':0.0, 'heading':0.0, 'vel':0.0, 'line':0, 'angle':0.0})
        self.add_flow('EE_12', {'v':0.0, 'a':0.0})
        self.add_flow('EE_5', {'v':0.0, 'a':0.0})
        self.add_flow('EE_15', {'v':0.0, 'a':0.0})
        self.add_flow('Video', {'liney':0.0,'linex':0.0, 'angle':params.initangle, 'quality':1.0})
        self.add_flow('AvionicsControl', {'rpower':0.0, 'lpower':0.0})
        self.add_flow('MotorControl', {'rpower':0.0, 'lpower':0.0})
        self.add_flow('Control', {'power':0.0})
        self.add_flow('Comms', {'x':0.0,'y':0.0, 'vel':0.0, 'heading':0.0})
        self.add_flow('OverrideComms', {'rpower':0.0, 'lpower':0.0, 'active':0} )
        self.add_flow('Obstacle', {'x':10.0,'y':10.0, 'r':0.0})
        self.add_flow('Faultstates', {'transfer':1.0, 'friction':1.0, 'drift': 0.0})
        #self.add_flow('Example_Disconnect')

        self.add_fxn("Power", ["EE_15","EE_5",'EE_12', "Control"], Power)
        self.add_fxn("Operator", ["Comms", "OverrideComms", "Pos_Signal", "Control"], Operator)
        self.add_fxn("Communications", ["Comms", "EE_12", 'Pos_Signal'], Communications)
        self.add_fxn("Perception", ["Ground", "EE_12", "Video","Obstacle"], Perception)
        self.add_fxn("Avionics",["Video","Comms", "EE_5",'Pos_Signal',"Ground", "AvionicsControl", "Faultstates"], fclass=Avionics, fparams=params)
        self.add_fxn("Override", ["OverrideComms", "EE_5", 'MotorControl','AvionicsControl'], Override)
        self.add_fxn("Drive", ["Ground","EE_15","EE_5", "MotorControl", "Faultstates"], fclass = Drive, fparams=(params.degradation, valparams))
        self.add_fxn("Environment", ['Ground',"Obstacle"], Environment, fparams = params)

        pos_bip = {'Power': [-0.684772948203272, -0.2551613615446115],
                 'Operator': [-0.798933011500376, 0.565156755693186],
                 'Communications': [-0.5566050878414673, 0.14159180700630447],
                 'Perception': [0.996672509613648, 0.2507215448302319],
                 'Avionics': [0.28027473355741117, 0.47255264233968597],
                 'Override': [0.28987624783062627, -0.17144760874154652],
                 'Drive': [0.6671719569482308, -0.571646956655247],
                 'Environment': [1.1329643169383754, -0.6375225566564033],
                 'Ground': [1.108432946123935, -0.3228541151507237],
                 'Pos_Signal': [-0.256557435572734, 0.5411037985681082],
                 'Faultstates': [0.75997843863482324, -0.04522869632581994],
                 'EE_12': [-0.3676879520509888, -0.04754907961317867],
                 'EE_5': [-0.2181352416728437, -0.2015320865756482],
                 'EE_15': [-0.5352906801304353, -0.5288715575154177],
                 'Video': [0.6726175830840695, 0.396008366729458],
                 'AvionicsControl': [0.45997843863482324, 0.04522869632581994],
                 'MotorControl': [0.6350063940085445, -0.3013633829278297],
                 'Control': [-0.9857988678463686, 0.07960895587242012],
                 'Comms': [-0.642370284813957, 0.35285736707043763],
                 'OverrideComms': [-0.14607433032593392, 0.2981956996230818],
                 'Obstacle':[1.195, -0.1]}

        self.build_model(bipartite_pos = pos_bip)
    def end_condition(self, time):
        if (in_area(self.flows['Ground'].x,self.flows['Ground'].y,1,self.params['end'][0],self.params['end'][1]) or \
            (time > 5 and self.fxns['Avionics'].in_mode('standby')) or \
                self.fxns['Avionics'].in_mode('em_off', 'finished')):
            return True
        else:
            return False
    def find_classification(self,scen,mdlhist):
        modes, modeproperties = self.return_faultmodes()
        classification = str()
        at_finish=True
        if not in_area(self.flows['Ground'].x,self.flows['Ground'].y,2,self.params.end[0],self.params.end[1]):
                                classification = "incomplete mission"
                                at_finish = False
        if any(modes):          classification = classification +' faulty'
        if not classification:  classification = 'nominal mission'
        num_modes = len(modes)
        end_dist = dist(self.flows['Ground'].x,self.flows['Ground'].y,self.params.end[0],self.params.end[1])
        endpt=[self.flows['Ground'].x,self.flows['Ground'].y]

        f_t= min(len(mdlhist['faulty']['flows']['Ground']['x']),len(mdlhist['nominal']['flows']['Ground']['y']))

        tot_deviation = np.sum(np.sqrt((mdlhist['nominal']['flows']['Ground']['x'][:f_t]-mdlhist['faulty']['flows']['Ground']['x'][:f_t])**2 + (mdlhist['nominal']['flows']['Ground']['y'][:f_t]-mdlhist['faulty']['flows']['Ground']['y'][:f_t])**2))
        in_bound = all(mdlhist['faulty']['functions']['Environment']['in_bound'])
        line_dist = find_line_dist(self.flows['Ground'].x,self.flows['Ground'].y, mdlhist['nominal']['flows']['Ground']['linex'], mdlhist['nominal']['flows']['Ground']['liney'])

        return {'rate':0,'cost':0, 'prob':scen['properties'].get('prob',1), 'expected cost':0, 'in_bound':in_bound, 'at_finish':at_finish, 'line_dist':line_dist, 'num_modes':num_modes, 'end_dist':end_dist, 'tot_deviation':tot_deviation, 'faults':modes, 'classification':classification, 'endpt':endpt}

import fmdtools.resultdisp as rd

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

    ax.plot(hist['flows']['Ground']['x'],hist['flows']['Ground']['y'], color='blue')
    ax.scatter(hist['flows']['Ground']['x'][-1],hist['flows']['Ground']['y'][-1], color='blue', marker='*', label=nom_lab)

    x_ground = hist['flows']['Ground']['lbx']
    y_ground = hist['flows']['Ground']['lby']
    ax.plot(x_ground,y_ground, label=bound_lab, color='grey')
    x_ground = hist['flows']['Ground']['ubx']
    y_ground = hist['flows']['Ground']['uby']
    ax.plot(x_ground,y_ground, label=bound_lab, color='grey')
    x_ground = hist['flows']['Ground']['x']
    y_ground = hist['flows']['Ground']['liney']
    ax.plot(x_ground,y_ground, label=center_lab, color='grey', linestyle='--')

def plot_trajectories(mdlhists, nomhist=[],  app= [], faultlabel='Faulty', faultalpha=0.1, range_hist={}, rangealpha=0.1, setalpha=0.3, show_labels=True, title="Fault Trajectories", textoffset=2.0,mode_trunc=5,mode_trunc_end=5, xlim=None, ylim=None, figsize=(4,4), ax=False, legend=True):
    in_mdlhists=False
    if not ax:
        fig,ax = plt.subplots(figsize=figsize)
        ax=fig.axes[0]
    else: fig=ax.get_figure()
    for mode, hist in range_hist.items():
        if mode[6:11]=='hmode':
            ax.plot(hist['flows']['Ground']['x'],hist['flows']['Ground']['y'], color='yellow', alpha=rangealpha)
            ax.scatter(hist['flows']['Ground']['x'][-1],hist['flows']['Ground']['y'][-1], color='yellow', alpha=rangealpha, marker='o', label='Range')
    for mode, hist in mdlhists.items():
        if mode=='nominal':
            plot_course(hist, ax=ax)
            in_mdlhists=True
        elif mode[6:11]=='hmode':
            ax.plot(hist['flows']['Ground']['x'],hist['flows']['Ground']['y'], color='orange', alpha=setalpha)
            ax.scatter(hist['flows']['Ground']['x'][-1],hist['flows']['Ground']['y'][-1], color='orange', alpha=setalpha, marker='o', label='Set')
        else:
            ax.plot(hist['flows']['Ground']['x'],hist['flows']['Ground']['y'], color='red', alpha=faultalpha)
            ax.scatter(hist['flows']['Ground']['x'][-1],hist['flows']['Ground']['y'][-1], color='red', alpha=faultalpha, marker='*', label=faultlabel)
            if show_labels:
                label = mode[mode_trunc:]
                label = label[:-mode_trunc_end]
                randang = np.pi*np.random.rand()
                ax.annotate(label, xy=(hist['flows']['Ground']['x'][-1],hist['flows']['Ground']['y'][-1]), fontsize=8, xytext=(textoffset*np.sin(randang), textoffset*np.cos(randang)), textcoords='offset points')
    if app: ax.scatter(hist['flows']['Ground']['x'][int(app.times[0])-1],hist['flows']['Ground']['y'][int(app.times[0])-1], color='black', marker='X', s=5, label='fault time')
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
        plt.plot(hist['flows']['Ground']['x'],hist['flows']['Ground']['y'], color='grey', alpha=0.2, zorder=1)
        plt.scatter(hist['flows']['Ground']['x'][-1],hist['flows']['Ground']['y'][-1], color='grey', alpha=0.3, marker='o', label=mdlhist1_name, zorder=2)
    for mode, hist in mdlhist2.items():
        plt.plot(hist['flows']['Ground']['x'],hist['flows']['Ground']['y'], color='tab:orange', alpha=0.2, zorder=1)
        plt.scatter(hist['flows']['Ground']['x'][-1],hist['flows']['Ground']['y'][-1], color='tab:orange', alpha=0.3, marker='o', label=mdlhist2_name, zorder=2)
    if faulttimes:
        xfaults = [x for ind, x in enumerate(nomhist['flows']['Ground']['x']) if ind+1<len(nomhist['time']) and nomhist['time'][ind+1] in faulttimes]
        yfaults = [y for ind, y in enumerate(nomhist['flows']['Ground']['y']) if ind+1<len(nomhist['time']) and nomhist['time'][ind+1] in faulttimes]
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
    plt.scatter(mdl.params.end[0],mdl.params.end[1], label="End Location", marker='X', color='grey')

    plot_course(mdlhist, ax=ax)

    plt.xlabel("x-distance (meters)")
    plt.ylabel("y-distance (meters)")
    plt.title("Rover Centerline Tracking")
    plt.grid()

    plt.legend()

def plot_centerline_err(mdl, mdlhist):
    fig = plt.figure()
    x_rover = mdlhist['flows']['Ground']['x']
    y_rover = mdlhist['flows']['Ground']['y']
    if mdl.params['linetype']=='sine':
        y_line = [sin_func(x,y_rover[i], mdl.params['amp'], mdl.params['period'])[1] for i,x in enumerate(x_rover)]
    elif mdl.params['linetype']=='turn':
        y_line = [turn_func(x,y_rover[i], mdl.params['radius'], mdl.params['start'])[1] for i,x in enumerate(x_rover)]

    plt.plot(x_rover, y_rover-y_line)
    plt.xlabel("x-distance (meters)")
    plt.ylabel("y-error (meters)")
    plt.title("Rover Centerline Error")

class_tree = {'Rover': [-0.07367185100835244, 0.6410710936138487],
 'Drive': [0.23684893559545173, 0.36072825626681254],
 'Environment': [-0.5104502883293205, 0.38994116849132504],
 'Avionics': [-0.20614969706316938, 0.3834687620416293],
 'Override': [0.03237968031217919, 0.3759790214222677],
 'Perception': [-0.8350044759810874, 0.40176785861898856],
 'Power': [0.4699926035926743, 0.35927158639441575],
 'Operator': [0.6783281961222181, 0.4006550001329594],
 'Communications': [1.0406903861169736, 0.37725113274488953],
 'MotorControl': [0.02456213706952065, -0.17456190594455168],
 'EE_5': [0.20244547912639976, 0.06448327387605401],
 'OverrideComms': [0.1966879650681026, -0.4239713648342337],
 'EE_15': [0.3596916288553037, 0.1191691659491266],
 'Comms': [0.7930664677006409, 0.2234018617977897],
 'Ground': [-0.5636731555210449, -0.024506034628188678],
 'AvionicsControl': [-0.22848556112602958, -0.0821525471741435],
 'Pos_Signal': [0.4189018653841845, -0.27023016168716785],
 'Control': [0.6801636335956871, 0.08587826440540502],
 'Video': [-0.6095250803472407, -0.32199506812858303],
 'EE_12': [0.6762136362997685, -0.26803553393104507],
 'Obstacle': [0,0]}

def get_params_from(mdlhist, t=1):
    friction = mdlhist['functions']['Drive']['friction'][t]
    drift = mdlhist['functions']['Drive']['drift'][t]
    return {'friction':friction, 'drift':drift}

def get_paramdist_from(mdlhists, t):
    friction=[]
    drift=[]
    for rep in mdlhists:
        fdict = get_params_from(mdlhists[rep], t)
        friction.append(fdict['friction'])
        drift.append(fdict['drift'])
    return {'friction':friction, 'drift':drift}

def sample_params(mdlhists, t=1, scen=1):
    mdlhist = [*mdlhists.values()][scen]
    return get_params_from(mdlhist, t)

def gen_sample_params(mdlhists, t=1, scen=1):
    degparams = sample_params(mdlhists, t=t, scen=scen)
    return {'linetype':'turn', 'degradation':DegParam(**degparams)}


if __name__=="__main__":
    import multiprocessing as mp
    
    from fmdtools.faultsim import search
    
    from pymoo.optimize import minimize
    
    from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
    import numpy as np
    mdl = Rover(modelparams=ModelParam(times=(0, 100), phases=(('start',0, 30), ('end', 31, 60)), use_end_condition=False))
    track={'functions':{"Environment":"in_bound"},'flows':{"Ground":"all"}}
    rover_prob = search.ProblemInterface("rover_problem", mdl, pool=mp.Pool(5), staged=True, track=track)
    app_drive = SampleApproach(mdl, faults='Drive', phases={'global':[0,39]}, defaultsamp={'samp':'evenspacing','numpts':3})
    rover_prob.add_simulation("drive_faults", "multi", app_drive.scenlist)
    rover_prob.add_variables("drive_faults", ("cor_f", (-10,100)), ("cor_d", (-100, 100)), ("cor_t", (-10,100)), vartype='param')
    rover_prob.add_objectives("drive_faults", end_dist="end_dist", tot_deviation="tot_deviation")
    
    params = RoverParam(linetype="turn")
    
    pymoo_prob = rover_prob.to_pymoo_problem(objectives="end_dist")
    algorithm=PatternSearch(x0=np.array([0.0,0.0,0.0])) 
    #res = minimize(pymoo_prob, algorithm, verbose=True)

    #dot = rd.graph.show(mdl, gtype="bipartite", renderer='graphviz')

    mdl = Rover(params)

    #mdl = Rover(params)
    endresults,  mdlhist = prop.nominal(mdl)
    phases, modephases = rd.process.modephases(mdlhist)
    plot_map(mdl, mdlhist)

    mdl_id = Rover(valparams={'drive_modes':'set'})
    app_id = SampleApproach(mdl_id, faults='Drive', phases={'drive':phases['Avionics']['drive']}, defaultsamp={'samp':'evenspacing', 'numpts':3})
    endclasses_id, mdlhists_id = prop.approach(mdl_id, app_id, pool=mp.Pool(4))




    #stochastic over replicates
    nomapp = NominalApproach()
    nomapp.add_seed_replicates('test', 100)
    endclasses, mdlhists = prop.nominal_approach(deg_mdl, nomapp, run_stochastic=True, desired_result='endclass')
    rd.plot.mdlhists(mdlhists, aggregation='mean_std')

    #individual slice
    param_slice = get_params_from(mdlhist, 10)
    paramdist = get_paramdist_from(mdlhists, 10)
    plt.hist(paramdist['friction'])
    plt.hist(paramdist['drift'])

    #question -- how do we sample this:
    #   - all replicates?
    #   - random sample of them?
    #   - what about times?
    #   - what if we get a complementary sample of times and etc?
    #   - if states in one replicate are the same as a different at the next, can we only sample one?

    behave_nomapp = NominalApproach()
    behave_nomapp.add_param_ranges(gen_sample_params, 'behave_nomapp', mdlhists, t=(1,100, 10), scen = (1,100,5))

    mdl=Rover()
    behave_endclasses, behave_mdlhists = prop.nominal_approach(mdl, behave_nomapp)
    f = plt.figure()
    f = plot_trajectories(behave_mdlhists)
    rd.plot.nominal_vals_2d(behave_nomapp, behave_endclasses, 't', 'scen')

    comp_groups = {'group_1': [*behave_endclasses][:100],'group_2': [*behave_endclasses][100:]}

    rd.plot.metric_dist(behave_endclasses, metrics=['line_dist', 'end_dist', 'x', 'y'], comp_groups=comp_groups, alpha=0.5, bins=10, metric_bins={'x':20})

    rd.plot.metric_dist_from(behave_mdlhists, times= [0, 10, 20], fxnflowvals = {'Ground':['x', 'y', 'linex', 'ang']}, alpha=0.5, bins=10)

    rd.plot.metric_dist_from(behave_mdlhists, times= 30, fxnflowvals = {'Ground':['x', 'y', 'linex', 'ang']}, comp_groups=comp_groups, alpha=0.5, bins=10)

    #behave_endclasses_nested, behave_mdlhists_nested = prop.nested_approach(mdl, behave_nomapp, pool=mp.Pool(5), faults='Drive')

    #res_comp = rd.tabulate.resilience_factor_comparison(behave_nomapp, behave_endclasses_nested, ['t'], 'at_finish', percent=False)
    #fig = rd.plot.resilience_factor_comparison(res_comp, stack=True)

    #endresults,  mdlhist = prop.one_fault(mdl, 'Drive','hmode_34', time=1, staged=False)
    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Drive':['friction','drift']})

    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Drive':['friction','drift', 'transfer']})

    mdl = Rover(params, valparams={'drive_modes':'manual'})
    endresults,  mdlhist = prop.one_fault(mdl, 'Drive','elec_open', time=1, staged=False)

    endresults,  mdlhist = prop.one_fault(mdl, 'Drive','hmode_34', time=1, staged=False)
    
    
    x = [1.0,0.0,1.0]
    mdl = Rover(params, valparams={'drive_modes':{'custom_fault':{'friction':x[0],'drift':x[1], 'transfer':x[2]}}})

    _, mdlhist = prop.nominal(mdl)

    endresults, reshist = prop.one_fault(mdl,'Drive','custom_fault', time=15, staged=True)
    
    
    line_dist = endresults['line_dist']
    end_loc = (reshist['faulty']['flows']['Ground']['x'][-1],reshist['faulty']['flows']['Ground']['y'][-1])
    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Drive':['friction','drift']})

   # app = NominalApproach()
   # app.add_param_ranges(gen_model_params, 'app', x, scen = (0,len(scen)-1,1))
   # endclasses, mdlhists = prop.approach(mdl, app)

    #plot_trajectories(mdlhist, app=app, faultlabel='Faulty Scenarios')


    #fig, ax =rd.graph.show(mdl, gtype='bipartite')
    #fig.savefig('bipartite_rover.pdf', format="pdf", bbox_inches = 'tight', pad_inches = 0.0)

    """
    mdl = Rover(params=gen_params('turn'))
    #dot = rd.graph.show(mdl, gtype="bipartite", renderer='graphviz')
    #params = gen_params('sine')
    #mdl = Rover(params)
    endresults,  mdlhist = prop.nominal(mdl)
    phases, modephases = rd.process.modephases(mdlhist)
    plot_map(mdl, mdlhist)

    mdl_id = Rover(valparams={'drive_modes':'else'})
    app_id = SampleApproach(mdl_id, faults='Drive', phases={'drive':phases['Avionics']['drive']})
    endclasses_id, mdlhists_id = prop.approach(mdl_id, app_id, staged=True)


    params = gen_params('sine',cor_d=-180, cor_f=1)
    mdl_thing = Rover(params=params)
    _,_, reshist = prop.one_fault(mdl,'Drive','stuck_right', time=15, staged=True)
    plt.figure()
    f = plot_trajectories({'nominal':mdlhist}, reshist,  faultalpha=0.6)

    rd.plot.mdlhistvals(reshist, time=15, fxnflowvals={'Drive':['friction','drift', 'transfer'], 'Power':'all'})

    app_opt = SampleApproach(mdl, faults='Drive', phases={'drive':phases['Avionics']['drive']}, defaultsamp={'samp':'evenspacing','numpts':4})

    #endresults,  mdlhist = prop.one_fault(mdl, 'Drive','elec_open', time=1, staged=False)
    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Drive':['friction','drift', 'transfer']})

    #endresults,  mdlhist = prop.one_fault(mdl, 'Drive','hmode_34', time=1, staged=False)
    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Drive':['friction','drift']})


    x = [100,0,2,0,2,-2,0,0,0]
    x_params = gen_params('sine', ub_f=x[0], lb_f=x[1], ub_t=x[2],lb_t=x[3], ub_d=x[4], lb_d=x[5], cor_f=x[6], cor_d=x[7], cor_t=x[8])
    mdl_0 = Rover(params=x_params)


    _,_, nomhist = prop.nominal(mdl_0)
    phases, modephases = rd.process.modephases(nomhist)
    app_0 = SampleApproach(mdl, faults='Drive', phases={'drive':phases['Avionics']['drive']}, defaultsamp={'samp':'evenspacing','numpts':4})
    endclasses_0, mdlhists_0 = prop.approach(mdl_0, app_0, staged=True)
    plt.figure()
    f = plot_trajectories(mdlhists_0, app=app_0,  faultalpha=0.6)


    x = [100,0,2,0,2,-2,1,-180,1]
    x_params = gen_params('sine', ub_f=x[0], lb_f=x[1], ub_t=x[2],lb_t=x[3], ub_d=x[4], lb_d=x[5], cor_f=x[6], cor_d=x[7], cor_t=x[8])
    mdl_1 = Rover(params=x_params)
    endclasses_1, mdlhists_1 = prop.approach(mdl_1, app_0, staged=True)

    #compare_trajectories(mdlhists_0, mdlhists_1, mdlhist1_name='fault trajectories', mdlhist2_name='comparison trajectories', faulttimes = app.times, nomhist=nomhist)


    #rd.graph.show(  scale=0.7)

    #summhist,_,_ = rd.process.hist(mdlhist)
    #rd.graph.results_from(mdl, summhist, [10,15,20])
    #rd.plot.mdlhistvals(mdlhist, legend=False)
    #rd.plot.mdlhistvals(mdlhist)

    plot_map(mdl, mdlhist)

    #endresults,  mdlhist = prop.one_fault(mdl, 'Drive','elec_open', staged=True, time=13, gtype='typegraph')
    endresults,  mdlhist_feed = prop.one_fault(mdl, 'Perception', 'bad_feed', staged=True, time=7, gtype='typegraph')
    plot_trajectories(mdlhist_feed, mdlhist, faultalpha=1.0)


    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Power':['charge','power']}, time=7, phases=phases, modephases=modephases)
    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Ground':['x','y', 'angle','vel', 'liney', 'ang']}, time=7, phases=phases)
    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Pos_Signal':['x','y', 'angle','vel', 'heading']}, time=7, phases=phases)
    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={'MotorControl':['rpower','lpower']}, time=7, phases=phases)
    rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Avionics':['mode']}, time = 13, phases=phases, modephases=modephases)
    rd.plot.mdlhistvals(mdlhist, fxnflowvals={'Perception':['mode']}, time = 13, phases=phases, modephases=modephases)
    #rd.plot.mdlhistvals(mdlhist, fxnflowvals={}, time = 7, phases=phases, modephases=modephases)
    app = NominalApproach()
    app.add_param_ranges(gen_params,'sine', 'sine', amp=(0, 10, 0.2), wavelength=(10,50,10))
    app.assoc_probs('sine', amp=(stats.uniform.pdf, {'loc':0,'scale':10}), wavelength=(stats.uniform.pdf,{'loc':10, 'scale':40}))
    #app.add_param_ranges(gen_params,'turn', radius=(5,40,5), start=(0, 20,5))

    #labels, faultfxns, degnodes, faultlabels
#    rd.graph.plot_bipgraph(classgraph, {node:node for node in classgraph.nodes},[],[],{}, pos=pos)
    rd.graph.show( gtype='typegraph', scale=0.7)

    #endresults,  mdlhist = prop.one_fault(mdl, 'Drive','elec_open', staged=True, time=13, gtype='bipartite')
    endresults,  mdlhist = prop.one_fault(mdl, 'Perception', 'bad_feed', staged=True, time=13, gtype='bipartite')
    rd.graph.show( gtype='bipartite', scale=0.7)

    reshist, _, _ = rd.process.hist(mdlhist)
    typehist = rd.process.typehist(mdl, reshist)
    rd.graph.results_from(mdl, reshist, [10,15,20])
    rd.graph.results_from(mdl, typehist, [10,15,20], gtype='typegraph') #), gtype='typegraph')
    rd.graph.result_from(mdl, reshist, 10, gtype='bipartite', renderer='graphviz')

    #endclasses, mdlhists= prop.nominal_approach(mdl, app, pool = mp.Pool(5))

    #state_probabilities = rd.process.state_probabilities(endclasses)

    #fig = rd.plot.nominal_vals_1d(app, endclasses, 'amp')
    #fig = rd.plot.nominal_vals_1d(app, endclasses, 'radius')

    #app = NominalApproach()
    #app.add_param_ranges(gen_params,'sine','sine', amp=(0, 10, 0.2), wavelength=(10,50,10), dummy=(1,10,1))

    #endclasses, mdlhists= prop.nominal_approach(mdl, app, pool = mp.Pool(5))
    #fig = rd.plot.nominal_vals_3d(app, endclasses, 'amp', 'wavelength', 'dummy')
    #app = SampleApproach(mdl, phases = phases, modephases = modephases)

    #endclasses, mdlhist = prop.approach(mdl, app)

    #app_joint = SampleApproach(mdl, phases = phases, modephases = modephases, jointfaults={'faults':2})

    #endclasses, mdlhist = prop.approach(mdl, app_joint)


    #tab = rd.tabulate.phasefmea(endclasses, app_joint)
    #rd.plot.samplecosts(app_joint, endclasses)

    #rd.plot.phases(phases)

    #figs = rd.plot.phases(phases, modephases, mdl)
    #figs = rd.plot.phases(phases, modephases, mdl, singleplot=False)
    """
    