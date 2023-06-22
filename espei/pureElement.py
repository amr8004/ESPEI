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
#from espei.paramselect import fit_formation_energy
#TODO KNOWNS
#H,S,G autocalculate testing
#HSG magnetic vs nonmagnetic automatic test and calculate
#

def pe_dict():
    global RTDB_globals
    global df
RTDB_globals = {}
df = []
# Input parameters
RTDB_globals['element'] = None
RTDB_globals['TM'] = None
RTDB_globals['Th'] = None
RTDB_globals['a_sig'] = None
RTDB_globals['FC'] = None
RTDB_globals['bta'] = None
RTDB_globals['p'] = None
RTDB_globals['Tc'] = None
RTDB_globals['Ph'] = None
RTDB_globals['S_BP1'] = None
RTDB_globals['S_BP2'] = None
RTDB_globals['S_BP3'] = None
RTDB_globals['S_BP4'] = None
RTDB_globals['L_BP1'] = None
RTDB_globals['L_BP2'] = None
RTDB_globals['L_BP3'] = None
RTDB_globals['L_BP4'] = None
RTDB_globals['constituent_1'] = None
RTDB_globals['constituent_2'] = None
RTDB_globals['constituent_3'] = None
# SGTE parameters
# Solid phase
RTDB_globals['S_a1'] = None
RTDB_globals['S_b1'] = None
RTDB_globals['S_c1'] = None
RTDB_globals['S_d1'] = None
RTDB_globals['S_e1'] = None
RTDB_globals['S_f1'] = None
RTDB_globals['S_g1'] = None
RTDB_globals['S_h1'] = None
RTDB_globals['S_a2'] = None
RTDB_globals['S_b2'] = None
RTDB_globals['S_c2'] = None
RTDB_globals['S_d2'] = None
RTDB_globals['S_e2'] = None
RTDB_globals['S_f2'] = None
RTDB_globals['S_g2'] = None
RTDB_globals['S_h2'] = None
RTDB_globals['S_a3'] = None
RTDB_globals['S_b3'] = None
RTDB_globals['S_c3'] = None
RTDB_globals['S_d3'] = None
RTDB_globals['S_e3'] = None
RTDB_globals['S_f3'] = None
RTDB_globals['S_g3'] = None
RTDB_globals['S_h3'] = None
RTDB_globals['S_a4'] = None
RTDB_globals['S_b4'] = None
RTDB_globals['S_c4'] = None
RTDB_globals['S_d4'] = None
RTDB_globals['S_e4'] = None
RTDB_globals['S_f4'] = None
RTDB_globals['S_g4'] = None
RTDB_globals['S_h4'] = None
# Liquid phase
RTDB_globals['L_a1'] = None
RTDB_globals['L_b1'] = None
RTDB_globals['L_c1'] = None
RTDB_globals['L_d1'] = None
RTDB_globals['L_e1'] = None
RTDB_globals['L_f1'] = None
RTDB_globals['L_g1'] = None
RTDB_globals['L_h1'] = None
RTDB_globals['L_a2'] = None
RTDB_globals['L_b2'] = None
RTDB_globals['L_c2'] = None
RTDB_globals['L_d2'] = None
RTDB_globals['L_e2'] = None
RTDB_globals['L_f2'] = None
RTDB_globals['L_g2'] = None
RTDB_globals['L_h2'] = None
RTDB_globals['L_a3'] = None
RTDB_globals['L_b3'] = None
RTDB_globals['L_c3'] = None
RTDB_globals['L_d3'] = None
RTDB_globals['L_e3'] = None
RTDB_globals['L_f3'] = None
RTDB_globals['L_g3'] = None
RTDB_globals['L_h3'] = None
RTDB_globals['L_a4'] = None
RTDB_globals['L_b4'] = None
RTDB_globals['L_c4'] = None
RTDB_globals['L_d4'] = None
RTDB_globals['L_e4'] = None
RTDB_globals['L_f4'] = None
RTDB_globals['L_g4'] = None
RTDB_globals['L_h4'] = None
# Calculated parameters
RTDB_globals['polya'] = None
RTDB_globals['polyb'] = None
RTDB_globals['Td_final'] = None
RTDB_globals['k1_final'] = None
RTDB_globals['k2_final'] = None
RTDB_globals['alfa_final'] = None
RTDB_globals['g_final'] = None
RTDB_globals['Te_final'] = None
RTDB_globals['LCE_A1_final'] = None
RTDB_globals['LCE_A2_final'] = None
RTDB_globals['LCE_TE1_final'] = None
RTDB_globals['LCE_TE2_final'] = None
RTDB_globals['diffSR'] = None
RTDB_globals['a_GL'] = None
RTDB_globals['b_GL'] = None
RTDB_globals['C_GL'] = None
RTDB_globals['d_GL'] = None
# TDB writer parameters
RTDB_globals['El'] = None
RTDB_globals['Ref'] = None
#return


def Define_Element(element, TM=None, Th=None, a_sig=None, FC=None, bta=None, p=None, Tc=None, Ph=None):
    DBU = PEDB.db_unary
    #print(db_PE_Unary)
    #print(os.getcwdb())
    #DBU = json.load(DBU1)
    ele2=str(element[0])
    def_ele = DBU[ele2]
    # print(def_ele)
    RTDB_globals['element'] = element
    vals = {'TM': TM, 'Th': Th, 'a_sig': a_sig, 'FC': FC, 'bta': bta, 'p': p, 'Tc': Tc, 'Ph': Ph}
    # i = 0
    for key, val in vals.items():
        if val:
            RTDB_globals[key] = val
        else:
            RTDB_globals[key] = def_ele[key]
        # i += 1
        # print(def_ele[key])

    others = ['S_BP1', 'S_BP2', 'S_BP3', 'S_BP4', 'L_BP1', 'L_BP2', 'L_BP3', 'L_BP4', 'constituent_1', 'constituent_2',
              'constituent_3']
    for key in others:
        RTDB_globals[key] = def_ele[key]
        # i += 1
    return RTDB_globals

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
    return -2*logLik + k*(nparm + 1)

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
    return parmEsts