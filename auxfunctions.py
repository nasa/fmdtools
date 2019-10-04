# -*- coding: utf-8 -*-
"""
File name: auxfunctions.py
Author: Daniel Hulse
Created: December 2018
Description: Misc functions for model definition
"""


import numpy as np
##MISC OPERATIONS TO USE IN MODEL DEFINITION

#m2to1
# multiplies a list of numbers which may take on the values infinity or zero
# in deciding if num is inf or zero, the earlier values take precedence
def m2to1(x):
    if np.size(x)>2:
        x=[x[0], m2to1(x[1:])]
    if x[0]==np.inf:
        y=np.inf
    elif x[1]==np.inf:
        if x[0]==0.0:
            y=0.0
        else:
            y=np.inf
    else:
        y=x[0]*x[1]
    return y

#trunc
# truncates a value to 2 (useful if behavior unchanged by increases)
def trunc(x):
    if x>2.0:
        y=2.0
    else:
        y=x
    return y

#truncn
# truncates a value to n (useful if behavior unchanged by increases)
def truncn(x, n):
    if x>n:
        y=n
    else:
        y=x
    return y

#not sure what this function is for
def dev(x):
    y=abs(abs(x-1.0)-1.0)
    return y

#translates L, R, and C into Left, Right, and Center
def rlc(x):
    y='NA'
    if x=='R':
        y='Right'
    if x=='L':
        y='Left'
    if x=='C':
        y='Center'
    return y

# creates list of corner coordinates for a square, given a center, xwidth, and ywidth
def square(center,xw,yw):
    square=[[center[0]-xw/2,center[1]-yw/2],\
            [center[0]+xw/2,center[1]-yw/2], \
            [center[0]+xw/2,center[1]+yw/2],\
            [center[0]-xw/2,center[1]+yw/2]]
    return square

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#checks to see if a point with x-y coordinates is in the area a
def inrange(area, x, y):
    point=Point(x,y)
    polygon=Polygon(area)
    return polygon.contains(point)

#takes the maximum of a variety of classifications given a list of strings
def textmax(texts):
    if 'major' in texts:
        maxt='major'
    elif 'moderate' in texts:
        maxt='moderate'
    elif 'minor' in texts:
        maxt='minor'
    elif 'replacement' in texts:
        maxt='replacement'
    else:
        maxt='none'
    return maxt
    


