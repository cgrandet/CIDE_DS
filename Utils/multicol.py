# -*- coding: utf-8 -*-
"""
--------------------
Created on Thu Jul 24 13:37:39 2014
This function checks for perfect multicollineratity
--------------------
Notes:
Input:  Xmat
Output: Xout, ind_x
Xout: matrix with linearly independent columns of Xmat
ind_x: indicator (1/0) for included variables
@author: A3940004
--------------------
"""
import numpy as np

def multicol(Xmat):
    nx,mx = Xmat.shape
    # get a copy of Xmat
    Xmat = np.asarray(Xmat)
    # Store and indicator of surviving: first one always survives
    ind_x = np.zeros((mx))
    ind_x[0] = 1
    # ready to start looping over columns
    # chck that it is indeed the case that not full rank
    # initialize prexmat
    prexmat = Xmat[:,0].copy().reshape((nx,1))
    if np.linalg.matrix_rank(Xmat)==mx:
        # if full-rank: just output same matrix
        ind_x = np.ones((mx,1))
        endmat = Xmat.copy()
    # if not full-rank, continue below    
    else: 
        for vv in np.arange(1,mx):
            testmat = np.concatenate((prexmat,Xmat[:,vv].reshape((nx,1))),axis=1)
            # if full-rank, just save corresponding column to xout
            if np.linalg.matrix_rank(testmat) == testmat.shape[1]:
                ind_x[vv] = 1
                prexmat = testmat.copy()
        # Done with loop:
        endmat = prexmat
    return endmat,ind_x
