# -*- coding: utf-8 -*-
"""
File name: ex_pump.py
Author: Daniel Hulse
Created: October 2019
Description: A simple model for explaining fault model definition

This model constitudes an extremely simple functional model of an electric-powered pump.

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
"""

from fmdtools.define.block import FxnBlock, Mode
from fmdtools.define.flow import Flow 
from fmdtools.define.model import Model, check_model_pickleability
from fmdtools.sim.approach import SampleApproach, NominalApproach
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State 
from fmdtools.define.time import Time
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
    """States for Water Flows"""
    flowrate :  float = 1.0
    pressure :  float = 1.0
    area :      float = 1.0
    level :     float = 1.0
class Water(Flow):
    _init_s=WaterStates

class EEStates(State):
    """States for EE flows"""
    current :   float = 1.0
    voltage  :    float = 1.0
class Electricity(Flow):
    _init_s=EEStates

class SignalStates(State):
    """States of Signal Flows"""
    power :     float = 1.0
class Signal(Flow):
    _init_s=SignalStates

"""
DEFINING RESILIENCE METRICS
Below we define certain functions used in the value function in find_classification
"""
def reseting_accumulate(vec):
    """ Accummulates vector for all positive output (e.g. if input =[1,1,1, 0, 1,1], output = [1,2,3,0,1,2])"""
    newvec = vec
    val=0
    for ind, i in enumerate(vec):
        if i > 0: val = i + val
        else:    val = 0
        newvec[ind] = val
    return newvec

def accumulate(vec):
    """ Accummulates vector (e.g. if input =[1,1,1, 0, 1,1], output = [1,2,3,3,4,5])"""
    return [sum(vec[:i+1]) for i in range(len(vec)) ]

"""
DEFINING MODEL PARAMETERS
Below we define a class that defines the parameter of the model
"""
class PumpParam(Parameter, readonly=True):
    """PumpParam defines the parameters which the pump may be simulated over."""
    cost: tuple = ("repair", "water")   # costs to tabulate in cost model (see find_classification)
    delay: int = 10                     # delay to use in MoveWater function
    delay_lim= (0, 100)                 # valid limits for delay


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
    Mode contains the probability model for faults is associated with each function:
        - failrate = X sets the failure rate for the function (to be distributed over all modes
        - faultparams defines a probability model for each mode, where modes is:
            - {modename: (%of failures, (% at each phase in mdl.phases), repaircosts)
    These failure rates will then be used to generate a list of scenarios for propagate.single_faults and SampleApproach()
    
    Note that these rates are given in occurences/hr by default. To change the units, use the option units='sec'/'min'/'hr'/'day' etc
    """
    failrate = 1e-5
    faultparams = {'no_v':(0.80,[0,1,0], 10000), 
                  'inf_v':(0.20, [0,1,0], 5000)}
class ImportEEState(State):
    effstate : float = 1.0

class ImportEE(FxnBlock):
    __slots__ = ['ee_out']
    _init_m = ImportEEMode
    _init_s = ImportEEState
    _init_ee_out = Electricity
    flownames = {"ee_1":"ee_out"}
    """
    Import EE is the line of electricity going into the pump
    We define it here as a subclass of the FxnBlock superclass (imported from define.py)
    the FxnBlock superclass, which adds the common aspects of the function objects.
    
    Notice how _init_m, _init_s, _init_ee_out variables are assigned to the classes for
    Modes, States, and the Electricity flow used in this FxnBlock. This binds those types
    to the FxnBlock so they are instiantiated and take the `m` (for mode) and `s` (for state) 
    role in the FxnBlock, respectively. For the flow `ee_out`, this defines a Flow variable
    which will be electricity and will additional be held in .flows here. The flownames 
    variable then tells us that `ee_1` at the model level will be `ee_out`
    at the function level (which isn't necessary if they are given the same name).
    """
    
    def condfaults(self,time):
        """
        condfaults() changes the state of the system if there is a change in state in a flow
        Using a condfaults method is optional but helpful for delinating between the determination of a fault and the behavior that results
        During fault propagation condfaults() executes before behavior()
        In this example,  if the current is too high, the line becomes an open circuit (e.g. due to a fuse or line burnout)
        """
        if self.ee_out.s.current>15.0: self.m.add_fault('no_v')
        
    def behavior(self,time):
        """
        behavior() defines the behavior of the function in terms of
        how the system behaves normally and under faults.
        """
        if self.m.has_fault('no_v'):    self.s.effstate=0.0 #an open circuit means no voltage is exported
        elif self.m.has_fault('inf_v'): self.s.effstate=100.0 #a voltage spike means voltage is much higher
        else:                           self.s.effstate=1.0 #normally, voltage is 500 V
        self.ee_out.s.voltage=self.s.effstate * 500

""" 
Import Water Classes
"""

class ImportWaterMode(Mode):
    failrate=1e-5
    faultparams = {'no_wat':(1.0, [1,1,1], 1000)}
    key_phases_by='global'


class ImportWater(FxnBlock):
    __slots__ = ['wat_out']
    _init_m = ImportWaterMode
    _init_wat_out = Water
    flownames = {"wat_1":"wat_out"}

    def behavior(self,time):
        """ The behavior is that if the flow has a no_wat fault, the wate level goes to zero"""
        if self.m.has_fault('no_wat'):  self.wat_out.s.level=0.0
        else:                           self.wat_out.s.level=1.0


""" 
Export Water Classes
"""
class ExportWaterMode(Mode):
    failrate=1e-5
    faultparams = {'block':(1.0, [1.5, 1.0, 1.0], 5000)}
    key_phases_by='global'
    
    
class ExportWater(FxnBlock):
    """ Import Water is the pipe with water going into the pump """
    __slots__ = ['wat_in']
    _init_m = ExportWaterMode 
    _init_wat_in = Water
    flownames = {'wat_2':'wat_in'}
    
    def behavior(self,time):
        """ Here a blockage changes the area the output water flows through """
        if self.m.has_fault('block'): self.wat_in.s.area=0.01

""" 
Import Signal Classes
"""
class ImportSigMode(Mode):
    failrate=1e-6
    faultparams = {'no_sig':(1.0, [1.5, 1.0, 1.0], 10000)}
    key_phases_by='global'

class ImportSig(FxnBlock):
    """ Import Signal is the on/off switch """
    __slots__ = ['sig_out']
    _init_m = ImportSigMode
    _init_sig_out = Signal
    flownames = {'sig_1':'sig_out'}
    
    def behavior(self, time):
        """ This function has time-dependent behavior.
        To have different operational modes depending on the time, use if/else 
        statements on the time variable, which is the simulation time.

        In this case, the power turns on at t=5 and turns back off at t=50.
        """
        if self.m.has_fault('no_sig'): self.sig_out.power=0.0 #an open circuit means no voltage is exported
        else:
            if time<5:      self.sig_out.s.power=0.0
            elif time<50:   self.sig_out.s.power=1.0
            else:           self.sig_out.s.power=0.0

""" 
Move Water Classes
"""
class MoveWatTime(Time):
    timernames = ('pressure_limit',)


class MoveWatStates(State):
    eff:    float = 1.0 #effectiveness state


class MoveWatParams(Parameter, readonly=True):
    delay:  int = 1 #delay parameter


class MoveWatMode(Mode):
    failrate=1e-5
    faultparams = {'mech_break':(0.6, [0.1, 1.2, 0.1], 5000), 'short':(1.0, [1.5, 1.0, 1.0], 10000)}
    key_phases_by ='global'


class MoveWat(FxnBlock):
    """  
    Move Water is the pump itself. While one could decompose this further, one function is used for simplicity.
    
    Note how this Function has more roles being filled
    
    - s (states) by MoveWatStates
    - p (parameter) by MoveWatParams, which lets us parameterize a delay
    - m (mode) by MoveWatMode
    - t (time) by MoveWatTime, which will be used so we can have a timer
    """
    __slots__ = ['ee_in', 'sig_in', 'wat_in', 'wat_out']
    _init_s = MoveWatStates
    _init_p = MoveWatParams
    _init_m = MoveWatMode
    _init_t = MoveWatTime
    _init_ee_in = Electricity
    _init_sig_in = Signal
    _init_wat_in = Water 
    _init_wat_out = Water
    flownames = {"ee_1":"ee_in", "sig_1":"sig_in", "wat_1":"wat_in", "wat_2":"wat_out"}
    def condfaults(self, time):
        """
            Here we use the timer to define a conditional fault that only occurs after a state is present after X seconds.
            We do that by incrementing the timer when the state is present.
            Note that this is done with the internal timestep dt, which we can change locally (for the function) 
            by passing dt=timestep in the super().__init__ method or globally by changing 'tstep' in modelparams
            When the timer exceeds the delay defined by the external variable, the fault is added.
        """
        if self.p.delay:
            if self.indicate_over_pressure(time):
                if time>self.t.time: 
                    self.t.pressure_limit.inc(self.t.dt)
                if self.t.pressure_limit.time>=self.p.delay:   
                    self.m.add_fault('mech_break')
        else:
            if self.indicate_over_pressure(time):        self.m.add_fault('mech_break')
    def indicate_over_pressure(self, time):
        """
        Indicators (methods with names indicate_XXX) can be used to mark individual
        conditions present in the model. Indicators return booleans which are then
        recorded in the .i structure in the model history
        """
        return self.wat_out.s.pressure>15.0
    def behavior(self, time):
        """ here we can define how the function will behave with different faults """
        if self.m.has_fault('short'):
            self.ee_in.s.current=500*10/5000*self.sig_in.s.power*self.ee_in.s.voltage
            self.s.eff=0.0
        elif self.m.has_fault('mech_break'):
            self.ee_in.s.current=0.2*10/5000*self.sig_in.s.power*self.ee_in.s.voltage
            self.s.eff=0.0
        else:
            self.ee_in.s.current=10/5000*self.sig_in.s.power*self.ee_in.s.voltage*min(13.0, self.wat_out.s.pressure)
            self.s.eff=1.0
        
        velocity = self.sig_in.s.power*self.s.eff*min(1000, self.ee_in.s.voltage)*self.wat_in.s.level
        self.wat_out.s.pressure = 10/500 * velocity/self.wat_out.s.area
        self.wat_out.s.flowrate = 0.3/500 * velocity*self.wat_out.s.area

        self.wat_in.s.assign(self.wat_out.s, 'pressure', 'flowrate')


##DEFINE MODEL OBJECT
class Pump(Model):
    __slots__=()
    _init_p = PumpParam
    default_sp = dict(phases=(('start',0,4),('on',5,49),('end',50,55)), times=(0,20, 55), dt=1.0, units='hr')
    default_track = {'flows':{'wat_2':{'s':'flowrate'}, 'ee_1':{'s':{'current'}}}, 'i':'all'}
    """
        This defines the pump model as a Model.

        Models take a dictionary of parameters as input defining any veriables and values to use in the model.
    
        Note that sp is the SimParam defining the simulation. phases in this dictionary are queues for fault 
        sampling which can be used by SampleApproach
        
        We can also chage dt to change the timestep, but note that this can change behavior.  
        In this model, because every time we've entered occurs at a factor of 5,
        and there aren't any complicated controls/dynamics interactions that would need to be
        tuned, we can easily use the timestep t=1 OR t=5.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Here add_flow() is used to instantiate a given flow object with a given type
        to a given name. Non-default values (for s, p, etc) can be passed, and we
        can also pass already-instantiated objects if desired.
        """
        self.add_flow('ee_1',   Electricity)
        self.add_flow('sig_1',  Signal)
        self.add_flow('wat_1',  Water('wat_1'))
        self.add_flow('wat_2',  Water('wat_2'))

        """
        Functions are added to the model using the add_fxn() method, which must be 
        called after add_flow, and needs:
           - a unique function name
           - the class to instantiate the function with (defined above)
           - non-default values (in this case, we are passing p from the model to function level for move_water)
        """
        self.add_fxn('import_ee',    ImportEE,      'ee_1')
        self.add_fxn('import_water', ImportWater,   'wat_1')
        self.add_fxn('import_signal',ImportSig,     'sig_1')
        self.add_fxn('move_water',   MoveWat,       'ee_1', 'sig_1', 'wat_1', 'wat_2', p = {'delay':self.p.delay})
        self.add_fxn('export_water', ExportWater,   'wat_2')
        
        """ __init__ for models ends with a build command to construct the underlying structure"""
        self.build()
    def indicate_finished(self,time):
        """
        Indicators can addtionally be used to stop the simulation when certain conditions are met.
        This method is optional, but helpful when the simulation is expensive and there are 
        defined end conditions (e.g., reaching a destination or failing to do so).
        
        It returns True when the end condition is met and False otherwise. Here 
        a dummy method is provided to demonstrate, in practice this would depend on
        the intended end-states of the model.
        """
        if time>self.sp.times[-1]: return True
        else:                      return False
    def indicate_on(self, time):
        return self.flows['wat_1'].s.flowrate>0
    def find_classification(self,scen, mdlhists):
        """
        Propagation methods use find_classification() to classify the results based on the
        effects of a fault scenario, returning whatever metrics are desired. In this case,
        a dictionary with rate, cost, and expected cost is calculated.

       In this example, there are three costs--water, electrical, and repair costs:
           - repair costs depends on the cost of each mode, while
           - electrical and water costs depend on the lost water in the non-nominal case
        """
        #get fault costs and rates
        if 'repair' in self.p.cost: repcost= self.calc_repaircost()
        else:                               repcost = 0.0
        if 'water' in self.p.cost:
            lostwat = sum(mdlhists['nominal'].flows.wat_2.s.flowrate- mdlhists['faulty'].flows.wat_2.s.flowrate)
            watcost = 750 * lostwat  * self.sp.dt
        elif 'water_exp' in self.p.cost:
            wat = mdlhists['nominal'].flows.wat_2.s.flowrate - mdlhists['faulty'].flows.wat_2.s.flowrate
            watcost =100 *  sum(np.array(accumulate(wat))**2) * self.sp.dt
        else: watcost = 0.0
        if 'ee' in self.p.cost:
            eespike = [spike for spike in mdlhists['faulty'].flows.ee_1.s.current - mdlhists['nominal'].flows.ee_1.s.current if spike >1.0]
            if len(eespike)>0: eecost = 14 * sum(np.array(reseting_accumulate(eespike))) * self.sp.dt
            else: eecost =0.0
        else: eecost = 0.0

        totcost = repcost + watcost + eecost

        rate=scen.rate

        life=1e5
        expcost=rate*life*totcost
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost}

if __name__=="__main__":
    
    mdl = Pump()
    endclass, mdlhist=propagate.nominal(mdl, track='all')
    fig, ax = an.plot.hist(mdlhist ,'flows.wat_1.s', 'i.on')
    
    mdl = Pump()
    newhist2 = mdl.create_hist(range(10), 'default')
    
    newhist2.flows.wat_2.s.flowrate
    
    
    mdl = Pump()
    
    endclass, mdlhist=propagate.nominal(mdl,  mdl_kwargs={'sp':{'end_condition':'indicate_on'}})
    
    endclass, mdlhist=propagate.one_fault(mdl, 'export_water','block', time=29, 
                                          mdl_kwargs={'sp':{'end_condition':'indicate_on'}})
    
    check_model_pickleability(mdl, try_pick=True)
    #from define.common import check_pickleability
    #unpickleable = check_pickleability(mdl, try_pick=True)
    
    #newhist = mdl.create_hist(range(10), 'all')
    
    
    
    #import pickle
    #a = pickle.dumps(newhist)
    #b = pickle.loads(a)
    
    #a = pickle.dumps(mdl.flows)
    #b = pickle.loads(a)
    
    #a = pickle.dumps(mdl.fxns)
    #b = pickle.loads(a)
    
    
    
    #c = pickle.dumps(mdl)
    #d = pickle.loads(c)
    
    mdl = Pump()
    newhist2 = mdl.create_hist(range(10), {'flows':{'ee_1':'all',"wat_1":{'s':('flowrate',)}}})
    mdl = Pump()
    newhist3 = mdl.create_hist(range(10), "all")
    mdl.flows['ee_1'].s
    
    mdl = Pump()
    newhist4 = mdl.create_hist(range(10), {'fxns':{'move_water':['s', 't']}})
    mdl.flows['ee_1'].s
    
    #an.graph.exec_order(mdl)
    endclass, mdlhist=propagate.one_fault(mdl, 'import_water','no_wat', time=29,  staged=True)
    
    #mdlhist.get_faulty_hist(*mdl.fxns)
    endclass, mdlhist=propagate.one_fault(mdl, 'move_water', 'mech_break', time=0, staged=False)
    
    app = NominalApproach()
    app.add_seed_replicates('test', 10)
    
    faultapp = SampleApproach(mdl)
    
    endclasses, mdlhists  = propagate.approach(mdl, faultapp)
    flat = mdlhists.flatten()
    
    gh = mdlhists.get_comp_groups('flows.ee_1.s.current')
    
    endclasses, mdlhists_staged  = propagate.approach(mdl, faultapp, staged=True, track='all')
    flat_staged = mdlhists_staged.flatten()
    
    
    [all(flat[k]==flat_staged[k]) for k in flat]
    all([all(flat[k]==flat_staged[k]) for k in flat])
    
    endclass, mdlhist=propagate.one_fault(mdl, 'import_ee','no_v', time=29,  staged=True, track='all')
    
    deghist = mdlhist.get_degraded_hist(*mdl.fxns, *mdl.flows)
    exp = deghist.get_metrics()
    deghist
    a=deghist.as_table()
    
    b=mdlhist.get_fault_degradation_summary(*mdl.fxns, *mdl.flows)
    
    tab = an.tabulate.result_summary_fmea(endclasses, mdlhists, *mdl.fxns, *mdl.flows)
    
    h = mdlhists.get_expected(app=faultapp, with_nominal=True)
    ec= endclasses.get_expected()
    
    #degsumm = h.get_summary(*mdl.fxns, *mdl.flows)
    
    d=h.get_degraded_hist(*mdl.flows, nomhist=mdlhists.nominal)
    
    
    exp = deghist.get_metrics()
    
    from fmdtools.analyze.graph import ModelGraph
    mg = ModelGraph(mdl)
    mg.set_heatmap(exp)
    mg.draw()
    
    mg = ModelGraph(mdl)
    fig, ax = mg.plot_high_degree_nodes()
    
    mg = ModelGraph(mdl)
    mg.set_exec_order(mdl)
    mg.draw()
    
    
    
    
    