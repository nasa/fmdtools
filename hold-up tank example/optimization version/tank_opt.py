# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:01:45 2021

@author: dhulse
"""

def x_to_descost(xdes):
    return (xdes[0]-10)*1000 + xdes[1].^2*10000

def x_to_ocost(xdes):
    
    endresults, resgraph, mdlhist = propagate.nominal(mdl)
    
    return (xdes[0]-10).^2*100  + xdes[1].^2*10000