"""
scurvebetatrue

Efficient Python implementation of the smoothest S-curve robot motion planner ever.

This module is optional, it is using true beta functions instead of interpolated.
"""
from __future__ import division # fix python2 1/2 = 0
import scipy
import numpy as np

def f_true(t):
    if(t<=-1):
        return 0.0
    if(t>=1):
        return 1.0
    # scipy incomplete beta function without regularization needs to be multiplied by beta
    if(t<0):
        return 0.5 - 0.5092958178940651 * scipy.special.betainc(0.5, 3.5, t * t) * scipy.special.beta(0.5, 3.5)
    if(t>0):
        return 0.5 + 0.5092958178940651 * scipy.special.betainc(0.5, 3.5, t * t) * scipy.special.beta(0.5, 3.5)
    return 0.5 # t == 0

def sCurve_true(t, motionTime, x0, x1):
    if(isinstance(t, list)):
        return [x0 + (x1 - x0) * f_true(2.0 * t1 / motionTime - 1) for t1 in t]
    if(type(t)==np.ndarray):
        return [x0 + (x1 - x0) * f_true(2.0 * t1 / motionTime - 1) for t1 in list(t)]
    return x0 + (x1 - x0) * f_true(2.0 * t / motionTime - 1)
    