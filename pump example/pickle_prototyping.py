# -*- coding: utf-8 -*-
"""
pickle tests
Created on Wed Mar 24 16:19:59 2021

@author: dhulse
"""

from ex_pump import * 
import pickle
import dill

IE = ImportEE([{'current':1.0, 'voltage':1.0}])
mdl = Pump()

#check if object will pickle
def check_pickleability(obj):
    unpickleable = []
    for name, attribute in vars(obj).items():
        if not dill.pickles(attribute):
            unpickleable = unpickleable + [name]
    if unpickleable: print("The following attributes will not pickle: "+str(unpickleable))
    else:           print("The object is pickleable")
    return unpickleable

def check_model_pickleability(model):
    unpickleable = check_pickleability(model)
    if 'flows' in unpickleable:
        print('Flows: ')
        for flowname, flow in model.flows.items():
            print(flowname)
            check_pickleability(flow)
    if 'fxns' in unpickleable:
        print('Functions: ')
        for fxnname, fxn in model.fxns.items():
            print(fxnname)
            check_pickleability(fxn)


check_pickleability(IE)

check_pickleability(mdl)
check_model_pickleability(mdl)

#pickle.dump( IE, open( "save.p", "wb" ) )

#IE_loaded = pickle.load( open( "save.p", "rb" ) )