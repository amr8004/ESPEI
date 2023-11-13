import numpy as np
from matplotlib import pylab as plt
import pandas as pd
from scipy.optimize import leastsq as lsq
from scipy.optimize import curve_fit
import scipy.stats as spst
from scipy import integrate
import os
import json
import espei.pure_element.DB_load as PEDB
from pycalphad.model import Define_Element
#from espei.paramselect import fit_formation_energy
#TODO KNOWNS
#H,S,G autocalculate testing
#HSG magnetic vs nonmagnetic automatic test and calculate
#

#KEY NOTES: parameter_selection/selection.py has model selection code, utilize here to automate model selection.

def imp_data_PE(file):
    """
    This needs to be changed to run through the espei command, run from espei_script.py script
    """
    global df
    #path = os.path.join('.\inst\Example_Data', file)
    df_open = open(file)
    df_raw = json.load(df_open)
    df_df = pd.DataFrame.from_dict(df_raw)
    df = df_df[df_df.Temp > 5]
    return df

def pe_inputJSON(file):
    #path= os.path.join('.\inst\Example_Data',file) #this probably needs changing
    inJSON=open(file)
    ldJSON=json.load(inJSON)
    ele=Define_Element(ldJSON['components'])
    return
def pe_def_model(file):
    inMod=open(file)
    ldMod=json.load(inMod)
    model=ldMod['model']
    return model

def pe_iGuess(file):
    inG=open(file)
    ldG=json.load(inG)
    iGuess=ldG['initialGuess']
    return iGuess

def pe_input(file): #look to nest all this?
    #yaml1=open(os.path.join(".\inst\Example_data",file))
    #yaml2=yaml.load(yaml1,Loader=yaml.FullLoader)
    yaml2 = yaml.load(file, Loader=yaml.FullLoader)
    syspe=yaml2['system']
    sys_pm=syspe['phase_models']
    sys_data=syspe['datasets']
    #print(sys_pm,sys_data)
    pe_inputJSON(sys_pm)
    imp_data(sys_data)
    return

# Get AIC. Add 1 to the df to account for estimation of standard error
def AIC(logLik, nparm,k=2):
    """ Look for built in AIC to replace this"""
    return - 2 * logLik + k * (nparm + 1)

def PE_AICC(nparm, nobs,rss,aicc_factor=None):
    print('CHECKPOINT AICC CALLED')
    k= nparm
    n = nobs
    print('k=',k,
    'n=',n)
    p=aicc_factor if aicc_factor is not None else 1.0
    pk = nparm*p
    aic = n * np.log(rss / n) + 2 * pk
    print('aic2:', aic)
    if pk >= (n-1.0):
        # Prevent the denominator of the proper mAICc from blowing up (pk = n - 1) or negative (pk > n - 1)
        correction = (2.0* p**2 * k**2 + 2.0 * pk) * (-n + pk + 3.0)
    else:
        correction = (2.0 * p**2 * k**2 + 2.0 * pk) / (n - pk - 1.0)
    aicc = aic+correction
    print('CHECKPOINT AICC SOLVED')
    return aicc

def Cp_fit(func, initialGuess, parmNames, data_df):
    """ Should be fine as is"""
    nparm = len(initialGuess)   # number of models parameters
    popt,pcov = curve_fit(func, data_df.Temp, data_df.Cp,initialGuess)  # get optimized parameter values and covariance matrix

    # Get the parameters
    parmEsts = popt
    fvec=func(data_df.Temp,*parmEsts)-data_df.Cp   # residuals

    # Get the Error variance and standard deviation
    RSS = np.sum(fvec**2 )        # RSS = residuals sum of squares
    dof = len(data_df) - nparm     # dof = degrees of freedom
    nobs = len(data_df)            # nobs = number of observation
    MSE = RSS / dof               # MSE = mean squares error
    RMSE = np.sqrt(MSE)           # RMSE = root of MSE

    # Get the covariance matrix
    cov = pcov

    # Get parameter standard errors
    parmSE = np.diag( np.sqrt( cov ) )

    # Calculate the t-values
    tvals = parmEsts/parmSE

    # Get p-values
    pvals = (1 - spst.t.cdf( np.abs(tvals),dof))*2

    # Get goodnes-of-fit criteria
    s2b = RSS / nobs
    logLik = -nobs/2 * np.log(2*np.pi) - nobs/2 * np.log(s2b) - 1/(2*s2b) * RSS

    fit_df=pd.DataFrame(dict( Estimate=parmEsts, StdErr=parmSE, tval=tvals, pval=pvals))

    fit_df.index=parmNames

    print ('Non-linear least squares')
    print ('Model: ' + func.__name__)
    print( '')
    print(fit_df)
    print()
    print ('Residual Standard Error: % 5.4f' % RMSE)
    print ('Df: %i' % dof)
    print('AIC:', AIC(logLik, nparm))
    print('AICC:', PE_AICC(nparm, nobs,RSS))
    return parmEsts