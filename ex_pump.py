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


import networkx as nx
import numpy as np

import auxfunctions as aux
import faultprop as fp

#Declare time range to run model over
times=[0,3, 55]

##DEFINE MODEL FLOWS
# Flows are defined using Python classes that are instantiated as objects

# Below defines the class for Electrical Energy
class EE:
    #attributes of the flow are defined during initialization
    def __init__(self):
        #arbitrary states for the flows can be defined. 
        #in this case, rate is the analogue to flow (e.g. current)
        #while effort is the analogue to force (e.g. voltage)
        self.rate=1.0
        self.effort=1.0
    #each flow has a status function that relays the values of important states when queried
    def status(self):
        #each state must be returned in this dictionary to be able to see it in the results
        status={'rate':self.rate, 'effort':self.effort}
        return status.copy()
    
# Defining the class for the flow of Water
class Water:
    def __init__(self):
        self.rate=1.0 # (e.g. velocity)
        self.effort=1.0 # (e.g. pressure)
        #here we will define a different state, viscosity, to illustrate conditional faults
        self.area=1.0 
        #level is the water level--if there is no water, nothing can flow
        self.level=1.0
    def status(self):
        status={'rate':self.rate, 'effort':self.effort, 'area': self.area, 'level': self.level}
        return status.copy()
# Defining the class for the flow of Signal
class Signal:
    def __init__(self):
        #here we define signal as just an on/off state
        self.power=1.0
    def status(self):
        status={'power':self.power}
        return status.copy()


##DEFINE MODEL FUNCTIONS
# Functions are, again, defined using Python classes that are instantiated as objects

# Import EE is the line of electricity going into the pump
class importEE:
    #Initializing the function requires the flows going in and out of the function
    def __init__(self,EEout):
        #flows going into/out of the function need to be made properties of the function
        self.EEout=EEout
        #fault modes that originate in the function are listed here in a dictionary
        #these modes will be used to generate a list of scenarios
        #each fault can be given arbitrary properties in a dictionary
        #in this case, rate and repair cost information
        self.faultmodes={'no_v':{'rate':'moderate', 'rcost':'major'}, \
                         'inf_v':{'rate':'rare', 'rcost':'major'}}
        #this initializes the fault state of the system
        #self.faults will be used to store which faults are in the function
        self.faults=set(['nom'])
    #condfaults changes the state of the system if there is a change in state in a flow
    # using a condfaults method is optional but helpful for delinating between
    # the determination of a fault and the behavior that results
    def condfaults(self):
        #in this case, if the current is too high, the line becomes an open circuit
        # (e.g. due to a fuse or line burnout)
        if self.EEout.rate>5.0:
            self.faults.update(['no_v'])
    #behavior defines the behavior of the function
    def behavior(self):
        #here we can define how the function will behave with different faults
        if self.faults.intersection(set(['no_v'])):
            self.EEout.effort=0.0 #an open circuit means no voltage is exported
        elif self.faults.intersection(set(['inf_v'])):
            self.effstate=100.0 #a voltage spike means voltage is much higher
        else:
            self.effstate=1.0 #normally, voltage is 1.0
    #the updatefxn method defines how the function is seen by the fault propogation code
    #generally, leave this part as-is
    def updatefxn(self,faults=['nom'], time=0): #fxns take faults and time as input
        self.faults.update(faults)  #if there is a fault, it is instantiated in the function
        self.condfaults()           #conditional faults and behavior are then run
        self.behavior()
        return 

# Import Water is the pipe with water going into the pump
class importWater:
    #Initializing the function requires the flows going in and out of the function
    def __init__(self,Watout):
        #flows going into/out of the function need to be made properties of the function
        self.Watout=Watout
        self.faultmodes={'no_wat':{'rate':'moderate', 'rcost':'major'}}
        self.faults=set(['nom'])
    #in this function, no conditional faults are modelled, so we can remove the method
    #as well as the corresponding line in updatefxn()
    def behavior(self):
        #here we can define how the function will behave with different faults
        if self.faults.intersection(set(['no_wat'])):
            self.Watout.level=0.0 #an open circuit means no voltage is exported
        else:
            self.Watout.level=1.0
    def updatefxn(self,faults=['nom'], time=0): 
        self.faults.update(faults)  
        self.behavior()
        return 

# Import Water is the pipe with water going into the pump
class exportWater:
    #Initializing the function requires the flows going in and out of the function
    def __init__(self,Watin):
        #flows going into/out of the function need to be made properties of the function
        self.Watin=Watin
        self.faultmodes={'block':{'rate':'moderate', 'rcost':'major'}}
        self.faults=set(['nom'])
    def behavior(self):
        if self.faults.intersection(set(['block'])): #here the fault is some sort of blockage
            self.Watin.area=0.0

    def updatefxn(self,faults=['nom'], time=0): 
        self.faults.update(faults)  
        self.behavior()
        return 
# Import Signal is the on/off switch
class importSig:
    def __init__(self,Sigout):
        #flows going into/out of the function need to be made properties of the function
        self.Sigout=Sigout
        self.faultmodes={'no_sig':{'rate':'moderate', 'rcost':'major'}}
        self.faults=set(['nom'])
    #when the behavior changes over time (and not just internal state) time must
    # be given as an input
    def behavior(self, time):
        if self.faults.intersection(set(['no_sig'])):
            self.Sigout.power=0.0 #an open circuit means no voltage is exported
        else:
            #Since the signal *generally* defines the operational profile of the system,
            # here we can specify what the system is supposed to do over time
            # in this case, turning on and then off
            if time<5:
                self.Sigout.power=0.0
            elif time<50:
                self.Sigout.power=1.0
            else:
                self.Sigout.power=0.0
    def updatefxn(self,faults=['nom'], time=0): 
        self.faults.update(faults)  
        self.behavior(time)
        return 

# Move Water is the pump itself. While one could decompose this further,
# one function is used for simplicity
class moveWat:
    def __init__(self,EEin, Sigin, Watin, Watout):
        self.EEin=EEin
        self.Sigin=Sigin
        self.Watin=Watin
        self.Watout=Watout
        self.faultmodes={'mech_break':{'rate':'moderate', 'rcost':'major'}, \
                         'short':{'rate':'rare', 'rcost':'major'}}
        self.faults=set(['nom'])
        #timers can be set by adding variables to functions also
        self.t1=0.0
        self.t2=0.0
        self.timer=0.0
    def condfaults(self, time):
        # here we define a conditional fault that only occurs after a state 
        # is present after 10 seconds
        if self.Watout.effort>5.0:
            if self.t1>time:
                self.t1=time
                self.timer+=1
            if self.timer>10.0:
                self.faults.update(['mech_break'])
    #behavior defines the behavior of the function
    def behavior(self, time):
        #here we can define how the function will behave with different faults
        if self.faults.intersection(set(['mech_break'])):
            self.EEin.rate=0.1*self.Sigin.power*self.EEin.effort
            self.effstate=0.0
        if self.faults.intersection(set(['short'])):
            self.EEin.rate=500*self.Sigin.power*self.EEin.effort
            self.effstate=0.0
        else:
            self.EEin.rate=1.0*self.Sigin.power*self.EEin.effort
            self.effstate=1.0 
            
        self.Watout.effort=self.Sigin.power*self.effstate*self.Watin.level/self.Watout.area
        self.Watout.rate=self.Sigin.power*self.effstate*self.Watin.level*self.Watout.area
        
        self.Watin.effort=self.Watout.effort
        self.Watin.rate=self.Watout.rate

    def updatefxn(self,faults=['nom'], time=0): #fxns take faults and time as input
        self.faults.update(faults)  #if there is a fault, it is instantiated in the function
        self.condfaults(time)           #conditional faults and behavior are then run
        self.behavior(time)
        return 
    
#INSTANTIATE MODEL
#the model is initialized using an initialize function
def initialize():
    #INITIALIZE FUNCTION AND FLOW OBJECTS
    
    #Flows must be instantiated prior to the functions that use them, since
    # the functions take them as input
    EE_1=EE()
    Wat_1=Water()
    Wat_2=Water()
    Sig_1=Signal()
    
    #function objects take their respective flows as input 
    Imp_EE=importEE(EE_1)
    Imp_Wat=importWater(Wat_1)
    Imp_Sig=importSig(Sig_1)
    Move_Wat=moveWat(EE_1, Sig_1, Wat_1, Wat_2)
    Exp_Wat=exportWater(Wat_2)
    
    #INITIALIZE AND ASSOCIATE OBJECTS WITH GRAPH
    #initializing the graph
    g=nx.DiGraph()
    
    #add nodes. The first argument gives the node a name. 
    #obj=fxn gives the graph structure the object instantiated for the function
    g.add_node('Import EE', obj=Imp_EE)
    g.add_node('Import Water', obj=Imp_Wat)
    g.add_node('Import Signal', obj=Imp_Sig)
    g.add_node('Move Water', obj=Move_Wat)
    g.add_node('Export Water', obj=Exp_Wat)
    
    #connect the notes with edges
    # flow_1=flow_1 associates a given flow with the edge
    #note that multiple flows can be added to edges using 
    #flow_1=flow_1, flow_2=flow_2 ... etc
    g.add_edge('Import EE', 'Move Water', EE_1=EE_1)
    g.add_edge('Import Signal', 'Move Water', Sig_1=Sig_1)
    g.add_edge('Import Water', 'Move Water', Wat_1=Wat_1)
    g.add_edge('Move Water', 'Export Water', Wat_2=Wat_2)
    
    return g

#PROVIDE MEANS OF CLASSIFYING RESULTS
# this function classifies the faults into severities based on the state of faults
# in this case, we will just use the repair costs and the probability
def findclassification(resgraph, endfaults, endflows, scen):
    
    #get fault costs and rates
    repcosts=fp.listfaultsprops(endfaults, resgraph, 'rcost')
    costs=repcosts.values()
    costkey={'major': 10000, 'minor': 1000}
    totcost=0.0
    
    for cost in costs:
        totcost=totcost+costkey[cost]
    
    life=100
    
    if scen['properties']['type']=='nominal':
        rate=1.0
    else:
        qualrate=scen['properties']['rate']
        ratekey={'rare': 1e-7, 'moderate': 1e-5}
        rate=ratekey[qualrate]
    
    life=1e5
    
    expcost=rate*life*totcost
    
    return {'rate':rate, 'cost': totcost, 'expected cost': expcost}

    

    