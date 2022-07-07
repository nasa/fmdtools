# -*- coding: utf-8 -*-
"""
File name: pump_stochastic.py
Author: Daniel Hulse
Created: October 2019
Description: A simple model for explaining stochastic behavior modelling

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

import sys, os
sys.path.append(os.path.join('..'))


from fmdtools.modeldef import *
import fmdtools.resultdisp as rd
import fmdtools.faultsim.propagate as propagate

"""
DEFINE MODEL FUNCTIONS
Functions are defined using Python classes that are instantiated as objects
"""

class ImportEE(FxnBlock):
    def __init__(self,name, flows):
        super().__init__(name,flows, flownames = ['EEout'])
        self.failrate=1e-5
        self.assoc_modes({'no_v':[0.80,[0,1,0], 10000], 'inf_v':[0.20, [0,1,0], 5000]}, key_phases_by='global')
        self.assoc_rand_states(('effstate', 1.0), ('grid_noise',1.0))
    def condfaults(self,time):
        if self.EEout.current>15.0: self.add_fault('no_v')
    def behavior(self,time):
        if self.has_fault('no_v'):      self.effstate=0.0 #an open circuit means no voltage is exported
        elif self.has_fault('inf_v'):   self.effstate=100.0 #a voltage spike means voltage is much higher
        else:                           
            if time>self.time: self.set_rand('effstate','triangular',0.9,1,1.1)
        if time>self.time:
            self.set_rand('grid_noise','normal',1, 0.1*(2+np.sin(np.pi/2*time)))
        
        self.EEout.voltage= self.grid_noise*self.effstate * 500

class ImportWater(FxnBlock):
    """ Import Water is the pipe with water going into the pump """
    def __init__(self,name,flows):
        """Here the only flows are the water flowing out"""
        super().__init__(name,flows, flownames=['Watout'])
        self.failrate=1e-5
        self.assoc_modes({'no_wat':[1.0, [1,1,1], 1000]}, key_phases_by='global')
        """
        in this function, no conditional faults are modelled, so it doesn't need to be included
        """
    def behavior(self,time):
        """ The behavior is that if the flow has a no_wat fault, the wate level goes to zero"""
        if self.has_fault('no_wat'): self.Watout.level=0.0
        else:                        self.Watout.level=1.0

class ExportWater(FxnBlock):
    """ Import Water is the pipe with water going into the pump """
    def __init__(self,name,flows):
        #flows going into/out of the function need to be made properties of the function
        super().__init__(name, flows, flownames=['Watin'])
        self.failrate=1e-5
        self.assoc_modes({'block':[1.0, [1.5, 1.0, 1.0], 5000]}, key_phases_by='global')
    def behavior(self,time):
        """ Here a blockage changes the area the output water flows through """
        if self.has_fault('block'): self.Watin.area=0.01

class ImportSig(FxnBlock):
    """ Import Signal is the on/off switch """
    def __init__(self,name,flows):
        """ Here the main flow is the signal"""
        super().__init__(name,flows, flownames=['Sigout'])
        self.failrate=1e-6
        self.assoc_modes({'no_sig':[1.0, [1.5, 1.0, 1.0], 10000]}, key_phases_by='global')
        self.assoc_rand_state('sig_noise',1.0)
    def behavior(self, time):
        if self.has_fault('no_sig'): self.Sigout.power=0.0 #an open circuit means no voltage is exported
        else:
            if time<5:      self.Sigout.power=0.0;self.to_default('sig_noise')
            elif time<50: 
                if not time%5:  self.set_rand('sig_noise', 'choice', [1.0, 0.9, 1.1])
                self.Sigout.power=1.0*self.sig_noise
            else:           self.Sigout.power=0.0; self.to_default()

class MoveWat(FxnBlock):
    """  Move Water is the pump itself. While one could decompose this further, one function is used for simplicity """
    def __init__(self,name, flows, delay):
        flownames=['EEin', 'Sigin', 'Watin', 'Watout']
        states={'total_flow':0.0} #effectiveness state
        self.delay=delay #delay parameter
        super().__init__(name,flows,flownames=flownames,states=states, timers={'timer'})
        self.failrate=1e-5
        self.assoc_modes({'mech_break':[0.6, [0.1, 1.2, 0.1], 5000], 'short':[1.0, [1.5, 1.0, 1.0], 10000]}, key_phases_by='global')
        self.assoc_rand_state("eff",1.0,auto_update=['normal', (1.0, 0.2)])
    def condfaults(self, time):
        if self.delay:
            if self.Watout.pressure>15.0:
                if time>self.time:                  self.timer.inc(self.dt)
                if self.timer.time>self.delay:      self.add_fault('mech_break')
        else:
            if self.Watout.pressure>15.0: self.add_fault('mech_break')

    def behavior(self, time):
        """ here we can define how the function will behave with different faults """
        if self.has_fault('mech_break'):
            self.Watout.pressure = 0.0
            self.Watout.flowrate = 0.0
        else:
            self.Watout.pressure = 10/500 * self.Sigin.power*self.eff*min(1000, self.EEin.voltage)*self.Watin.level/self.Watout.area
            self.Watout.flowrate = 0.3/500 * self.Sigin.power*self.eff*min(1000, self.EEin.voltage)*self.Watin.level*self.Watout.area

        self.Watin.pressure=self.Watout.pressure
        self.Watin.flowrate=self.Watout.flowrate
        if time>self.time: self.total_flow+=self.Watout.flowrate


class Water(Flow):
    def __init__(self):
        attributes={'flowrate':1.0, \
                    'pressure':1.0, \
                    'area':1.0, \
                    'level':1.0}
        super().__init__(attributes, 'Water',ftype='Water')
        self.customattribute='hello'

##DEFINE MODEL OBJECT
class Pump(Model):
    def __init__(self, params={'cost':{'repair', 'water'}, 'delay':10, 'units':'hrs'}, \
                 modelparams = {'phases':{'start':[0,4], 'on':[5, 49], 'end':[50,55]}, 'times':[0,20, 55], 'tstep':1,'seed':1}, \
                     valparams={'flows':{'Wat_2':'flowrate', 'EE_1':'current'}}):
        super().__init__(params=params, modelparams=modelparams, valparams=valparams)
        self.add_flow('EE_1', {'current':1.0, 'voltage':1.0})
        self.add_flow('Sig_1',  {'power':1.0})
        # custom flows which we defined earlier can be added also:
        self.add_flow('Wat_1', Water())
        self.add_flow('Wat_2', Water())

        self.add_fxn('ImportEE',['EE_1'],fclass=ImportEE)
        self.add_fxn('ImportWater',['Wat_1'],fclass=ImportWater)
        self.add_fxn('ImportSignal',['Sig_1'],fclass=ImportSig)
        self.add_fxn('MoveWater', ['EE_1', 'Sig_1', 'Wat_1', 'Wat_2'],fclass=MoveWat, fparams = params['delay'])
        self.add_fxn('ExportWater', ['Wat_2'], fclass=ExportWater)

        self.build_model()
    def find_classification(self,scen, mdlhists):
        #get fault costs and rates
        if 'repair' in self.params['cost']: repcost= self.calc_repaircost()
        else:                               repcost = 0.0
        if 'water' in self.params['cost']:
            lostwat = sum(mdlhists['nominal']['flows']['Wat_2']['flowrate'] - mdlhists['faulty']['flows']['Wat_2']['flowrate'])
            watcost = 750 * lostwat  * self.tstep
        elif 'water_exp' in self.params['cost']:
            wat = mdlhists['nominal']['flows']['Wat_2']['flowrate'] - mdlhists['faulty']['flows']['Wat_2']['flowrate']
            watcost =100 *  sum(np.array(accumulate(wat))**2) * self.tstep
        else: watcost = 0.0
        if 'ee' in self.params['cost']:
            eespike = [spike for spike in mdlhists['faulty']['flows']['EE_1']['current'] - mdlhists['nominal']['flows']['EE_1']['current'] if spike >1.0]
            if len(eespike)>0: eecost = 14 * sum(np.array(reseting_accumulate(eespike))) * self.tstep
            else: eecost =0.0
        else: eecost = 0.0

        totcost = repcost + watcost + eecost

        if scen['properties']['type']=='nominal':   rate=1.0
        else:                                       rate=scen['properties']['rate']

        life=1e5
        expcost=rate*life*totcost
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost}

def paramfunc(delay):
    return {'delay':delay}

if __name__=="__main__":
    import multiprocessing as mp
    
    app = NominalApproach()
    app.add_param_replicates(paramfunc, 'no_delay', 100, (0))
    app.add_param_replicates(paramfunc, 'delay_10', 100, (10))
    
    mdl = Pump(modelparams = {'phases':{'start':[0,4], 'on':[5, 49], 'end':[50,55]}, 'times':[0,20, 55], 'tstep':1,'seed':3})

    
    # endresults, resgraph, mdlhist=propagate.nominal(mdl)
    # rd.plot.mdlhistvals(mdlhist, fxnflowvals={'MoveWater':'eff', 'Wat_1':'flowrate', 'Wat_2':['flowrate','pressure']})
    
    # endresults, resgraph, mdlhist=propagate.nominal(mdl,run_stochastic=True)
    # rd.plot.mdlhistvals(mdlhist, fxnflowvals={'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']})
    
    endresults, resgraph, mdlhist=propagate.one_fault(mdl, 'ExportWater','block', time=20, staged=False, run_stochastic=True, modelparams={'seed':10})
    rd.plot.mdlhistvals(mdlhist, fxnflowvals={'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']}, legend=False)
    
    rd.plot.mdlhists(mdlhist, fxnflowvals={'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']})
    
    app_comp = NominalApproach()
    app_comp.add_param_replicates(paramfunc, 'delay_1', 100, (1))
    app_comp.add_param_replicates(paramfunc, 'delay_10', 100, (10))
    endclasses, mdlhists, apps =propagate.nested_approach(mdl,app_comp, run_stochastic=True, faults=[('ExportWater','block')], staged=True)
    
    comp_mdlhists = {scen:mdlhist['ExportWater block, t=27'] for scen,mdlhist in mdlhists.items()}
    comp_groups = {'delay_1': app_comp.ranges['delay_1']['scenarios'], 'delay_10':app_comp.ranges['delay_10']['scenarios']}
    fig = rd.plot.mdlhists(comp_mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']}, comp_groups=comp_groups, aggregation='percentile', time_slice=27) 
    
    
    tab = rd.tabulate.resilience_factor_comparison(app_comp, endclasses, ['delay'], 'cost')
    # app = NominalApproach()
    # app.add_seed_replicates('test_seeds', 100)
    # endclasses, mdlhists=propagate.nominal_approach(mdl,app, run_stochastic=True)
    # rd.plot.mdlhists(mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']},\
    #                               ylabels={('Wat_2', 'flowrate'):'liters/s'}, color='blue', alpha=0.1, legend_loc=False)
    # rd.plot.mdlhists(mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']}, aggregation='mean_std',\
    #                               ylabels={('Wat_2', 'flowrate'):'liters/s'})
    # rd.plot.mdlhists(mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']}, aggregation='mean_bound',\
    #                               ylabels={('Wat_2', 'flowrate'):'liters/s'})
    # rd.plot.mdlhists(mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']}, aggregation='percentile',\
    #                           ylabels={('Wat_2', 'flowrate'):'liters/s'})     
    
    # rd.plot.mdlhists(mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']}, aggregation='mean_ci',\
    #                               ylabels={('Wat_2', 'flowrate'):'liters/s'}, time_slice=[3,5,7])
        
    # rd.plot.mdlhists(mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure']}, aggregation='mean_ci',\
    #                  comp_groups={'test_1':[*mdlhists.keys()][:50], 'test_2':[*mdlhists.keys()][50:]},\
    #                               ylabels={('Wat_2', 'flowrate'):'liters/s'}, time_slice=[3,5,7])
    # rd.plot.mdlhists(mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure'], 'ImportEE':['effstate', 'grid_noise'], 'EE_1':['voltage','current'], 'Sig_1':['power']},\
    #                               ylabels={('Wat_2', 'flowrate'):'liters/s'}, cols=2, color='blue', alpha=0.1, legend_loc=False)
    # rd.plot.mdlhists(mdlhists, {'MoveWater':['eff','total_flow'], 'Wat_2':['flowrate','pressure'], 'ImportEE':['effstate', 'grid_noise'], 'EE_1':['voltage','current'], 'Sig_1':['power']}, aggregation='percentile',\
    #                               ylabels={('Wat_2', 'flowrate'):'liters/s'}, cols=2, color='blue', alpha=0.1, legend_loc=False)
    #rd.plot.nominal_vals_1d(app, endclasses, 'test_seeds')
    
    
    