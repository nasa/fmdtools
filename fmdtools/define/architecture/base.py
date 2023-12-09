# -*- coding: utf-8 -*-
"""
Description: Module defining the base architecture class.
"""
from fmdtools.define.base import check_pickleability
from fmdtools.define.block.base import Simulable
import time


class Architecture(Simulable):

    def init_architecture(self, *args, **kwargs):
        """Use to initialize architecture."""
        return 0

def inject_faults_internal(obj, faults, compdict):
    """
    Inject faults in the ComponentArchitecture/ASG object obj.

    Parameters
    ----------
    obj : TYPE
        DESCRIPTION.
    faults : TYPE
        DESCRIPTION.
    """
    for fault in faults:
        if fault in obj.faultmodes:
            comp = compdict[obj.faultmodes[fault]]
            comp.m.add_fault(fault[len(comp.name)+1:])


def check_model_pickleability(model, try_pick=False):
    """
    Check to see which attributes of a model object will pickle.

    Provides more detail about functions/flows.
    """
    print('FLOWS ')
    for flowname, flow in model.flows.items():
        print(flowname)
        check_pickleability(flow, try_pick=try_pick)
    print('FUNCTIONS ')
    for fxnname, fxn in model.fxns.items():
        print(fxnname)
        check_pickleability(fxn, try_pick=try_pick)
    time.sleep(0.2)
    print('MODEL')
    unpickleable = check_pickleability(model, try_pick=try_pick)
