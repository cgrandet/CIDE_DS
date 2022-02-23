# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import linalg
from scipy import stats
import missing as mi
import multicol as mc


class ols_dan:
    '''
    This class estimates OLS and returns all relevant parameters plus two
    DataFrames (xmat_df, betahat_df) with variable names
    '''
    def __init__(self,*args):
        # before defining attributes clean data
        nargs   = len(args)
        if nargs == 2: # I only included y,X
            Ymat  = args[0]
            Xmat  = args[1]
            names = np.asarray(['X'+str(i) for i in range(Xmat.shape[1])])
            multi_check = True
        elif nargs == 3:
            Ymat  = args[0]
            Xmat  = args[1]
            names = args[2]
            multi_check = True
        elif nargs == 4:
            Ymat  = args[0]
            Xmat  = args[1]
            names = args[2]
            multi_check = args[3]
        # Ready to start:
        xmat    = np.matrix(Xmat)
        K       = xmat.shape[1]
        T       = xmat.shape[0]
        ymat    = np.matrix(Ymat)
        # it would be nice to have and indicator variable of those excluded
        # because of missing values
        ind_lin = np.arange(T).reshape((T,1))
        #data    = mi.missing(np.concatenate((ymat,xmat,ind_lin), axis=1))
        data    = pd.DataFrame(np.concatenate((ymat,xmat,ind_lin), axis=1)
                            ).dropna().values
        nobs = data.shape[0]
        # get those surviving
        ind_surv = data[:,-1]
        # eliminate this column and continue
        data    = data[:,:-1]
        ymat    = data[:,0].reshape((nobs,1))
        prexmat = data[:,1:].astype('float')
        # check first for multicolinearity
        if multi_check == True:
            xmat, ind_out = mc.multicol(prexmat)
        else:
            xmat, ind_out = prexmat, np.ones((K,1)).T
        # get the names and a data frame of xmat with names
        names   = names[ind_out.flatten()==1]
        xmat_df = pd.DataFrame(xmat, columns=names)
        K       = xmat.shape[1]
        T       = xmat.shape[0]
        dof     = T-K
        xtx     = np.dot(xmat.T,xmat)
        # use the QR decomposition to get the inverse of XtX
        #xtxinv  = linalg.inv(xtx)
        #Q,R     = linalg.qr(xmat)
        #xtxinv  = linalg.inv(np.dot(R.T,R))
        # Use cholesky decomposition: it is faster than QR when K large
        if multi_check==True:
            L      = np.linalg.cholesky(xtx)
            self.L    = L
            Linv   = np.linalg.inv(L)
            xtxinv = np.dot(Linv.T, Linv)
        else:
            xtxinv = np.linalg.inv(xtx)
        xty     = np.dot(xmat.T,ymat)

        self.xmat = xmat
        self.xmat_df = xmat_df
        self.names = names
        self.ymat = ymat
        self.ind_surviving = ind_surv
        # create invXX
        self.invXX = xtxinv
        self.dof   = dof
        self.xtx   = xtx
        self.xty   = xty
        self.nobs  = T
        self.nvar  = K
        self.ind_mc_reg = ind_out
        self.muti_check = multi_check
    # these are the methods of the class
    def nobs(self):
        ''' return number of observations'''
        return self.nobs

    def nvar(self):
        ''' return number of observations'''
        return self.nvar

    def betahat(self):
         '''Returns OLS beta estimates'''
         #return np.asarray(self.invXX*self.xty)
         if self.muti_check==True:
            return np.linalg.solve(self.L.T,
                                      np.linalg.solve(self.L,self.xty))
         else:
            return np.dot(self.invXX, self.xty)

    def betahat_df(self):
         '''Returns OLS beta estimates as a DataFrame with names'''
         betahat_df = pd.DataFrame(np.asarray(self.betahat()).reshape((1,self.nvar))
                                     ,columns=self.names)

         return betahat_df

    def yest(self):
         '''Returns yhat'''
         return np.dot(self.xmat,self.betahat())

    def resids(self):
        ''' Returns residuals'''
        return self.ymat - self.yest()

    def ssr(self):
        '''compute sum of squared residuals'''
        resmat = self.resids()
        return np.dot(resmat.T,resmat)

    def s2hat(self):
        resmat = self.resids()
        return self.ssr()/self.dof

    def vcv_sph(self):
         '''Computes VCV with spherical disturbances'''
         s2hat = self.s2hat()
         return s2hat[0,0]*self.invXX

    def stderr_sph(self):
         '''Standard errors under spherical disturbances'''
         vcv  = self.vcv_sph()
         return np.sqrt(np.diag(vcv))

    #def vcv_white(self):
    #    ''' Computes robust White VCV matrix'''
    #    resids = self.resids()
    #    T      = self.nobs
    #    K      = self.nvar
    #    xmat   = self.xmat
    #    xtxinv = self.invXX
    #    S0 = np.zeros([K,K])
    #    for t in xrange(T):
    #        # ready to start
    #        eps_t = np.asarray(resids[t])
    #        S0 = S0 + (1/float(T))*((eps_t[0,0]**2)*xmat[t,:].T*xmat[t,:])
    #    VCV_white = np.dot(T,xtxinv*S0*xtxinv)
    #    return VCV_white

    def vcv_white(self):
        ''' Computes robust White VCV matrix'''
        resids = self.resids()
        T      = self.nobs
        xmat   = self.xmat
        xtxinv = self.invXX
        res    = np.asarray(resids).reshape((T,1))
        prex   = res*np.asarray(xmat)
        S0     = (1/float(T))*np.dot(prex.T,prex)
        VCV_white= np.dot(T,xtxinv*S0*xtxinv)
        return VCV_white

    def vcv_white_df(self):
         '''Returns OLS vcv_white estimates as a DataFrame with names'''
         vcv_white_df = pd.DataFrame(np.asarray(self.vcv_white())
                                     ,columns=self.names)

         return vcv_white_df

    def stderr_white(self):
         '''Standard errors for White robust vcv matrix'''
         vcv  = self.vcv_white()
         return np.sqrt(np.diag(vcv))


    def stderr_white_df(self):
         '''Returns OLS vcv_white std. error estimates as a DataFrame with names'''
         stderr_white_df = pd.DataFrame(np.asarray(self.stderr_white()).reshape((1,self.nvar))
                                     ,columns=self.names)

         return stderr_white_df


    def tstat_sph(self):
        '''Compute tstatistic for the case of spherical disturbances'''
        vcv_sph = self.vcv_sph()
        K = self.nvar

        se_sph = np.sqrt(np.matrix.reshape(np.matrix.diagonal(vcv_sph),K,1))
        tstat_sph = self.betahat()/se_sph
        return tstat_sph

    def tstat_white(self):
        '''Compute tstatistic for the case of robust VCV (White)'''
        vcv_white = self.vcv_white()
        K = self.nvar
        se_white = np.sqrt(np.matrix.reshape(np.matrix.diagonal(vcv_white),K,1))
        tstat_white = self.betahat()/se_white
        return tstat_white

    def pval_sph(self):
        '''Compute p-values for spherical VCV'''
        tstat_sph = self.tstat_sph()
        dof       = self.dof
        pval_sph = 2*(1-stats.t.cdf(abs(tstat_sph),dof));
        return pval_sph

    def pval_white(self):
        '''Compute p-values for spherical VCV'''
        tstat_white = self.tstat_white()
        dof       = self.dof
        pval_white = 2*(1-stats.t.cdf(abs(tstat_white),dof));
        return pval_white

    def pval_white_df(self):
         '''Returns OLS White pvalues as a DataFrame with names'''
         pval_white_df = pd.DataFrame(np.asarray(self.pval_white()).reshape((1,self.nvar))
                                     ,columns=self.names)

         return pval_white_df

    def R2(self):
        '''Compute R2'''
        resids = self.resids()
        y      = self.ymat
        T      = self.nobs
        R2 =  1 - (resids.T*resids)/(y.T*y - T*np.mean(y)**2)
        R2 = R2[0,0]
        return R2

    def R2_bar(self):
        '''Compute R2_bar'''
        R2  = self.R2()
        T   = self.nobs
        dof = self.dof
        R2_bar = 1- (1-R2)*(T-1)/dof
        return R2_bar

    def Fstat(self):
        '''Compute Fstat '''
        K        = self.nvar
        T        = self.nobs
        ymat     = self.ymat
        xres     = np.ones((T,1))
        betares  = (1/xres.T.dot(xres))*xres.T.dot(ymat)
        rres     = ymat - xres*betares
        R2       = self.R2()
        dof      = self.dof
        R2res    = 1 - (rres.T*rres)/(ymat.T*ymat - T*np.mean(ymat)**2)
        Fstat    = ((R2 - R2res[0,0])/(K-1))/((1-R2)/dof)
        return Fstat

    def Fstat_pval(self):
        '''Compute p-value for F-statistic'''
        Fstat    = self.Fstat()
        K        = self.nvar
        dof      = self.dof
        Fstat_pv = 1 - stats.f.cdf(abs(Fstat),K-1, dof)
        return Fstat_pv
