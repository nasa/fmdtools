# -*- coding: utf-8 -*-
"""
File name: modeldef.py
Author: Daniel Hulse
Created: October 2019

Description: A module to simplify model definition
"""


#add timers?
class fxnblock(object):
    def __init__(self,flows):
        self.type = 'function'
        for flow in flows.keys():
            setattr(self, flow,flows[flow])
        self.faults=set(['nom'])
    def condfaults(self,time):
        return 0
    def behavior(self,time):
        return 0
    def updatefxn(self,faults=['nom'], time=0): #fxns take faults and time as input
        self.faults.update(faults)  #if there is a fault, it is instantiated in the function
        self.condfaults(time)           #conditional faults and behavior are then run
        self.behavior(time)
        return

#class EE:
#    #attributes of the flow are defined during initialization
#    def __init__(self):
#        #arbitrary states for the flows can be defined. 
#        #in this case, rate is the analogue to flow (e.g. current)
#        #while effort is the analogue to force (e.g. voltage)
#        self.rate=1.0
#        self.effort=1.0
#    #each flow has a status function that relays the values of important states when queried
#    def status(self):
#        #each state must be returned in this dictionary to be able to see it in the results
#        status={'rate':self.rate, 'effort':self.effort}
#        return status.copy()
#    
#def associate(flownames,flowobjs):
#    flows={}
#    for i in length(flownames):
#        flows[flowname[i]]=flowobj
#    return flows
#    
#EE_1=EE()
#EE_2=EE()
#
#class dosomething(fxnblock):
#    def __init__(self,EE1, EE2):
#        self.faults=set('nom')
#        flows={'EE1':EE1,'EE1':EE2}
#        super().__init__(flows)
#    def condfaults(self,time):
#        #A good boss nurtures talent making employees happy!
#        print("The employees feel all warm and fuzzy then put their talents to good use.")
#    def behavior(self,time):
#        #A good boss encourages their employees!
#        print("The team cheers, starts shouting awesome slogans then gets back to work.")
#
#class doless(fxnblock):
#    def __init__(self,EE1, EE2):
#        self.faults=set('nom')
#        flows={'EE1':EE1,'EE1':EE2}
#        super().__init__(flows)
#    def behavior(self,time):
#        #A good boss encourages their employees!
#        print("The team cheers, starts shouting awesome slogans then gets back to work.")
#    
#    
#testfxn=dosomething(EE_1,EE_2)
#
#testfxn2=doless(1,2)