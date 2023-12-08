# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:22:12 2023

@author: dhulse
"""

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