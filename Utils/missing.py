# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 19:42:29 2014

@author: A3940004
"""
import numpy as np

def missing(A):
    # this works fine on arrays
    A = np.asarray(A)
    B = A[~np.isnan(A).any(axis=1)]
    # make it a matrix again
    B = np.asmatrix(B)
    return B
