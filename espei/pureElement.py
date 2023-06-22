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

#EINSTEIN model
""" This model will be moved to pycalphad later
"""
def Einstein(Te, T):
    # 3R value constant
    koef = 3 * 8.314
    # Einstein function
    Cv_E=[]
    Cv_E = koef * (Te / T) ** 2 * np.exp(Te / T) / (np.exp(Te / T) - 1) ** 2
    return Cv_E

def model_RWE(T,*param):
    Theta_E = param[0]
    a = param[1]
    b = param[2]
    Cp_res = Einstein(Theta_E,T) + a * T + b * T**2
    return Cp_res

def magn_model_RWE(T,*param):
    Theta_E = param[0]
    a = param[1]
    b = param[2]
    Cp_res = Einstein(Theta_E,T) + a * T + b * T**2 + CpMBosse(T)
    return Cp_res

def RWModelE(x, *param):
    RTDB_globals['Te_final'] = param[0]
    RTDB_globals['polya'] = param[1]
    RTDB_globals['polyb'] = param[2]
    if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
        CP_SRME=model_RWE(x, *param)
    else:
        CP_SRME=magn_model_RWE(x, *param)
    return CP_SRME
def model_CS(T,*param):
    Theta_E = param[0]
    a = param[1]
    b = param[2]    
    Cp_res = Einstein(Theta_E,T) + a * T + b * T**4
    return Cp_res

def magn_model_CS(T,*param):
    Theta_E = param[0]
    a = param[1]
    b = param[2]    
    Cp_res = Einstein(Theta_E,T) + a * T + b * T**4 + CpMBosse(T)
    return Cp_res

# Segmented Regression + Einstein
def model_SRE(T,*param):
    Theta_E = param[0]
    k1= param[1]
    k2= param[2]
    alfa= param[3]
    g= param[4]
    #
    CP_E_SR_final = []
    for i in T:
        if i < (alfa - g):
            Cp = k1 * i
        elif i > (alfa + g):
            Cp = (k1 * i) + (k2 * (i - alfa))
        else:
            Cp = k1 * i + k2 * (i - alfa + g)**2/(4*g)
        #E_cp=Einstein(Theta_E,T)
        f1 = np.exp(Theta_E/i)/(np.exp(Theta_E/i)-1)**2.0
        E_cp = 3.0 * 8.314 * (Theta_E/i)**2.0 * f1
        #Cp_final= BentCable(T, k1, k2, alfa, g)+Einstein(Theta_E,T)
        #print(type(E_cp), E_cp, type(Cp), Cp)
        Cp_final=Cp+E_cp
        CP_E_SR_final.append(Cp_final)
    #print(CP_E_SR_final)
    return CP_E_SR_final

# Segmented Regression + Einstein
def magn_model_SRE(T,*param):
    Theta_E = param[0]
    k1= param[1]
    k2= param[2]
    alfa= param[3]
    g= param[4]
    #
    CP_E_SR_final = []
    CP_MAG=CpMBosse(T)
    for i in T:
        if i < (alfa - g):
            Cp = k1 * i
        elif i > (alfa + g):
            Cp = (k1 * i) + (k2 * (i - alfa))
        else:
            Cp = k1 * i + k2 * (i - alfa + g)**2/(4*g)
        #E_cp=Einstein(Theta_E,T)
        f1 = np.exp(Theta_E/i)/(np.exp(Theta_E/i)-1)**2.0
        E_cp = 3.0 * 8.314 * (Theta_E/i)**2.0 * f1
        #Cp_final= BentCable(T, k1, k2, alfa, g)+Einstein(Theta_E,T)
        #print(type(E_cp), E_cp, type(Cp), Cp)
        Cp_final=Cp+E_cp
        CP_E_SR_final.append(Cp_final)
    zipped= zip(CP_E_SR_final,CP_MAG)
    CP_E_SRM_final = [x + y for (x, y) in zipped]
    #print(CP_E_SR_final)
    return CP_E_SRM_final

def integrand(x):
    return (x**4 * np.exp(x))/((np.exp(x) - 1)**2)


def SRModelE(x, *param):
    if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
        CP_SRME=model_SRE(x, *param)
    else:
        CP_SRME=magn_model_SRE(x, *param)
    RTDB_globals['Te_final'] = param[0]
    RTDB_globals['k1_final'] = param[1]
    RTDB_globals['k2_final'] = param[2]
    RTDB_globals['alfa_final'] = param[3]
    RTDB_globals['g_final'] = param[4]
    return CP_SRME

def CSModelE(x, *param):
    if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
        CP_SRME=model_CS(x, *param)
    else:
        CP_SRME=magn_model_CS(x, *param)
    RTDB_globals['Te_final'] = param[0]
    RTDB_globals['polya'] = param[1]
    RTDB_globals['polyb'] = param[2]
    return CP_SRME

#' Magnetic contribution to heat capacity - Stable Solid
#'
#' @param x Temp. range
#'
#' @return Magnetic contribution to heat capacity
#'
def CpMBosse(x):
    Smagn = 8.314 * np.log(RTDB_globals['bta'] + 1)
    Dm = 0.33471979 + 0.49649686 * (1 / RTDB_globals['p'] - 1)
    Cp_magn = []
    # Checks to see if a single value or a list/array is fed which determines whether to iterate or just solve a single value
    try:
        len(x)
    except TypeError:
        Tau = x/RTDB_globals['Tc']
        if Tau < 1:
            Cp_m = Smagn*0.63570895*(1/RTDB_globals['p'] -1)*(2*Tau**3 + 2*Tau**9/3 + 2*Tau**15/5 + 2*Tau**21/7)/Dm
        else:
            Cp_m = Smagn * (2 * Tau ** (- 7) + 2 * Tau ** (- 21) / 3 + 2 * Tau ** (- 35) / 5 + 2 * Tau ** (- 49) / 7) / Dm
        Cp_magn.append(Cp_m)
    else:
        for i in x:
            Tau = i/RTDB_globals['Tc']
            if Tau < 1:
                Cp_m = Smagn*0.63570895*(1/RTDB_globals['p'] -1)*(2*Tau**3 + 2*Tau**9/3 + 2*Tau**15/5 + 2*Tau**21/7)/Dm
            else:
                Cp_m = Smagn * (2 * Tau ** (- 7) + 2 * Tau ** (- 21) / 3 + 2 * Tau ** (- 35) / 5 + 2 * Tau ** (- 49) / 7) / Dm
            Cp_magn.append(Cp_m)
    return Cp_magn

# CP of Melting Temperature JUST FOR RW MODEL
def CpMelt():
    melt= RTDB_globals['TM']
    if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
        if  melt < (RTDB_globals['alfa_final'] -RTDB_globals['g_final']):
            Cp = RTDB_globals['k1_final'] * melt
        elif melt > (RTDB_globals['alfa_final'] +RTDB_globals['g_final']):
            Cp = (RTDB_globals['k1_final'] * melt) + (RTDB_globals['k2_final'] * (melt - RTDB_globals['alfa_final']))
        else:
            Cp = RTDB_globals['k1_final'] *melt+ RTDB_globals['k2_final'] * (melt - RTDB_globals['alfa_final'] +RTDB_globals['g_final'])**2/(4*RTDB_globals['g_final'])
        CPTM = Einstein(RTDB_globals['Te_final'],melt) +  Cp
    else:
        if  melt < (RTDB_globals['alfa_final'] -RTDB_globals['g_final']):
            Cp = RTDB_globals['k1_final'] * melt
        elif melt > (RTDB_globals['alfa_final'] +RTDB_globals['g_final']):
            Cp = (RTDB_globals['k1_final'] * melt) + (RTDB_globals['k2_final'] * (melt - RTDB_globals['alfa_final']))
        else:
            Cp = RTDB_globals['k1_final'] *melt+ RTDB_globals['k2_final'] * (melt - RTDB_globals['alfa_final'] +RTDB_globals['g_final'])**2/(4*RTDB_globals['g_final'])
        CPTM = Einstein(RTDB_globals['Te_final'],melt) + Cp + CpMBosse(melt)
    return CPTM

#Solid phase Cp, takes into account solid melting and then plots SGTE above TM, limited availability implemented model wises
def MSRCpSolid(x):
    CP_TM=CpMelt()
    print('Cp of Melting is ',+ CP_TM)
    Cp = []#* len(x)
    #count = range(0,len(x),1)
    #print(count)
    Tsol=[]
    Cpmelt=[]
    for i in x:
            sigma = ((i-(RTDB_globals['TM']))/50)/(np.sqrt(1+((i - (RTDB_globals['TM'])) /(RTDB_globals['a_sig']))**2))
            if i < RTDB_globals['TM']:
                Tsol.append(i)
                #Cpi = magn_model_SRE(i,RTDB_globals['Te_final'],RTDB_globals['k1_final'],RTDB_globals['k2_final'],RTDB_globals['g_final'],RTDB_globals['alfa_final'])
                #print(i,Cpi)
                #Cp.append(float(Cpi))
            elif i >= RTDB_globals['TM'] and i <= RTDB_globals['Th']:
                Cpi = CP_TM * (1-sigma) + sigma*(RTDB_globals['FC'])                
                #print("Above Tm ", +Cpf)
                Cpmelt.append(float(Cpi))
    #print("Sigma=", sigma)
    Cpsol = SRModelE(Tsol,RTDB_globals['Te_final'],RTDB_globals['k1_final'],RTDB_globals['k2_final'],RTDB_globals['alfa_final'],RTDB_globals['g_final'])
    #print(Cpsol[0])
    #print("Tsol =",Tsol)
    Cp=Cpsol+Cpmelt
    return Cp

# S, H, G Einstein Computations from Cp
def HEin(Te,x):
    # ' Enthalpy_Einstein
    # '
    # ' @param Te Einstein temperature
    # ' @param x Temp range
    # '
    # ' @return Enthalpy_Einstein
    koef = 3*8.314
    he = koef*Te/(np.exp(Te/x)-1)
    return he

def SEin(Te,x):
    # ' Entropy_Einstein
    # '
    # ' @param Te Einstein temperature
    # ' @param x Temp range
    # '
    # ' @return Entropy_Einstein
    e1=np.exp(Te/x)
    e2=e1-1
    koef = 3*8.314
    se = -1*koef* (np.log(e2) - Te * e1/(x * e2))
    return se

def GEin(Te,x):
    # ' Gibbs_Einstein
    # '
    # ' @param Te Einstein temperature
    # ' @param x Temp range
    # '
    # ' @return Gibbs_Einstein
    koef = 3*8.314
    Ge = koef*(x*np.log(np.exp(Te/x)-1)-Te) - HEin(Te, 298.15)
    return Ge

# Bent Cable Model ONLY H, S, G
def HTBCM(x, k1, k2, alfa, g):
    #' Enthalpy SR model
    #
    #  @param x,
    #  @param k1
    #  @param k2
    #  @param alfa
    #  @param g
    #
    H = []
    R = 8.314
    try:
        len(x)
    except TypeError:
        if x < alfa - g:
            Hi = 1/2 * k1 * x**2
            H.append(Hi)
        elif (alfa - g) < x and x <= alfa+g:
            Hi = 1/2 * k1 * x**2 + 1/12 * k2 * (x-alfa+g)**3/g
            H.append(Hi)
        elif x > alfa+g:
            Hi = 1/2*k1*x**2 + k2 * (1/2 * x**2 - alfa * x) + 1/6 * k2 * g ** 2.0 + 1/2 * k2 * alfa ** 2
            H.append(Hi)
    else:
        for i in x:
            if i < alfa - g:
                Hi = 1/2 * k1 * i**2
                H.append(Hi)
            elif (alfa - g) < i and i <= alfa+g:
                Hi = 1/2 * k1 * i**2 + 1/12 * k2 * (i-alfa+g)**3/g
                H.append(Hi)
            elif i > alfa+g:
                Hi = 1/2*k1*i**2 + k2 * (1/2 * i**2 - alfa * i) + 1/6 * k2 * g ** 2.0 + 1/2 * k2 * alfa ** 2
                H.append(Hi)
    return H
def STBCM(x,k1,k2,alfa,g):
    #' Entropy SR model
    #
    #  @param x,
    #  @param k1
    #  @param k2
    #  @param alfa
    #  @param g
    #
    R = 8.314
    try:
        len(x)
    except TypeError:
        if x < alfa - g:
            S = k1*x
        elif alfa-g <= x and x <= alfa +g:
            S = (alfa -g)**2.0*(3.0*k2/(8.0*g) - k2 *np.log(alfa-g)/(4*g))+(k1-k2*(alfa-g)/(2.0*g))*x+k2/(8.0*g)*x**2+k2/(4.0*g)*(alfa-g)**2.0*np.log(x)
        elif x > alfa+g:
            S = (-3.0*k2*alfa/2 - k2/(4.0*g)*((alfa-g)**2 * np.log(alfa-g)-(alfa+g)**2*np.log(alfa+g)))+(k1+k2)*x-k2*alfa*np.log(x)
    else:
        S = []
        for i in x:
            if i < alfa - g:
                Si = k1*i
                S.append(Si)
            elif alfa-g <= i and i <= alfa +g:
                Si = (alfa -g)**2.0*(3.0*k2/(8.0*g) - k2 *np.log(alfa-g)/(4*g))+(k1-k2*(alfa-g)/(2.0*g))*i+k2/(8.0*g)*i**2+k2/(4.0*g)*(alfa-g)**2.0*np.log(i)
                S.append(Si)
            elif i > alfa+g:
                Si = (-3.0*k2*alfa/2 - k2/(4.0*g)*((alfa-g)**2 * np.log(alfa-g)-(alfa+g)**2*np.log(alfa+g)))+(k1+k2)*i-k2*alfa*np.log(i)
                S.append(Si)
    return S

#===============================================================================
#
### --------------------------- Temp * Entropy T*S(T) --------------------------
#
def TSTBCM(x, k1, k2, alfa, g):
    R = 8.314
    try:
        len(x)
    except TypeError:
        if x < alfa - g:
            S = k1*x**2
        elif alfa-g <= x and x <= alfa +g:
            S = (alfa -g)**2.0*(3.0*k2/(8.0*g) - k2 *np.log(alfa-g)/(4*g))*x+(k1-k2*(alfa-g)/(2.0*g))*x**2+k2/(8.0*g)*x**3.0+k2/(4.0*g)*(alfa-g)**2.0*x*np.log(x)
        elif x > alfa+g:
            S = (-3.0*k2*alfa/2 - k2/(4.0*g)*((alfa-g)**2 * np.log(alfa-g)-(alfa+g)**2*np.log(alfa+g)))*x+(k1+k2)*x**2-k2*alfa*x*np.log(x)
    else:
        S = []
        for i in x:
            if i < alfa - g:
                Si = k1*i**2
                S.append(Si)
            elif alfa-g <= i and i <= alfa +g:
                Si = (alfa -g)**2.0*(3.0*k2/(8.0*g) - k2 *np.log(alfa-g)/(4*g))*i+(k1-k2*(alfa-g)/(2.0*g))*i**2+k2/(8.0*g)*i**3.0+k2/(4.0*g)*(alfa-g)**2.0*i*np.log(i)
                S.append(Si)
            elif i > alfa+g:
                Si = (-3.0*k2*alfa/2 - k2/(4.0*g)*((alfa-g)**2 * np.log(alfa-g)-(alfa+g)**2*np.log(alfa+g)))*i+(k1+k2)*i**2-k2*alfa*i*np.log(i)
                S.append(Si)
    return S

def GTBCM(x, k1, k2, alfa, g):
    G=[]
    try:
        len(x)
    except TypeError:
        G2 = HTBCM(x, k1, k2, alfa, g) - TSTBCM(x,k1,k2,alfa,g) - HTBCM(298.15, k1, k2, alfa, g)
    else:
        for i in x:
            Gi = HTBCM(i, k1, k2, alfa, g) - TSTBCM(i,k1,k2,alfa,g) - HTBCM(298.15, k1, k2, alfa, g)
            G.append(Gi)
            G2=[]
            for i in range(len(G)):
                G2.append(G[i][0])
    return G2

#Magnetic S, H, G
def GibbsM_Bosse(x):
    # ' Magnetic contribution to Gibbs energy - Stable Solid
    # '
    # ' @param x Temp. range
    # '
    # ' @return Magnetic contribution to Gibbs energy
    # '
    try:
        len(x)
    except TypeError:
        #print('single point, len(x) caused an error')
        Smagn = 8.314*x*np.log(RTDB_globals['bta']+1)
        Tau = x/RTDB_globals['Tc']
        Dm = 0.33471979 + 0.49649686*(1/RTDB_globals['p']-1)

        if Tau > 1:
            gMagn = -(Tau**(-7.0)/21 + Tau**(-21.0)/630 + Tau**(-35.0)/2975 + Tau**(-49.0)/8232)/Dm
        else:
            gMagn = 1-(0.38438376*Tau**(-1.0)/RTDB_globals['p'] + 0.63570895 *(1/RTDB_globals['p']-1)*(Tau**3/6 + Tau**9/135 + Tau**15/600 +Tau**21/1617))/Dm
        GibbsM = Smagn*gMagn
    else:
        #print('more than 1')
        GibbsM=[]
        for i in x:
            Smagn = 8.314*i*np.log(RTDB_globals['bta']+1)
            Tau = i/RTDB_globals['Tc']
            Dm = 0.33471979 + 0.49649686*(1/RTDB_globals['p']-1)
            if Tau > 1:
                gMagn = -(Tau**(-7.0)/21 + Tau**(-21.0)/630 + Tau**(-35.0)/2975 + Tau**(-49.0)/8232)/Dm
                #print(gMagn)
            else:
                gMagn = 1-(0.38438376*Tau**(-1.0)/RTDB_globals['p'] + 0.63570895 *(1/RTDB_globals['p']-1)*(Tau**3/6 + Tau**9/135 + Tau**15/600 +Tau**21/1617))/Dm
            Gibbs1 = Smagn*gMagn
            #print(type(Gibbs1),Gibbs1)
            GibbsM.append(Gibbs1)
    return GibbsM

def SM(x):
    # ' Magnetic contribution to Entropy - Stable Solid
    # '
    # ' @param x Temp. range
    # '
    # ' @return Magnetic contribution to Entropy
    # '
    # Translated not checked
    GM=GibbsM_Bosse(x)
    npGM=np.array(GM)
    Si=-1*npGM
    #print(Si)
    try:
        len(x)
    except TypeError:
        Sgrad = np.gradient(Si,x)
        return Sgrad
    else:
        Sgrad = np.gradient(Si, x)
        return Sgrad
def HM(x):
    # ' Magnetic contribution to Enthalpy - Stable Solid
    # '
    # ' @param x Temp. range
    # '
    # ' @return Magnetic contribution to Enthalpy
    GM=GibbsM_Bosse(x)
    Hi = GM+x*SM(x)
    return Hi
def GM_Bosse(x):
    # ' Magnetic contribution to Gibbs energy (Corrected) - Stable Solid
    # '
    # ' @param x Temp. range
    # '
    # ' @return Magnetic contribution to Gibbs energy (Corrected)
    gm_res = GibbsM_Bosse(x) - HM(298.15)
    return gm_res

# RW model H, S, G
def HTRWM(x, a, b):
    # Ringwald Workshop model contribution to Enthalpy
    # param x Temp. range
    # param a fit term from model
    # param b fit term from model
    # return RW model contribution to enthalpy
    H = []
    R = 8.314
    try:
        len(x)
    except TypeError:
        Hi = 1/2* a * x**2 + 1/3* b * x**3 #Integrate Cp
        H.append(Hi)
    else:
        for i in x:
            Hi= 1/2* a * i**2 + 1/3* b * i**3 #integrate Cp
            H.append(Hi)
    return H
def STRWM(x,a,b):
    # Ringwald Workshop model contribution to Entropy
    # param x Temp. range
    # param a fit term from model
    # param b fit term from model
    # return RW model contribution to entropy
    S = []
    S0=0
    try:
        len(x)
    except TypeError:
        Si=a*x+1/2*b*x**2+S0
        #S0=???
        S.append(Si)
    else:
        for i in x:
            Si=a*i+1/2*b*i**2
            S.append(Si)
    return S
def TSRWM(x,a,b):
    # Ringwald Workshop model contribution to Entropy * temperature for G calculation
    # param x Temp. range
    # param a fit term from model
    # param b fit term from model
    # return RW model contribution to entropy
    TS=[]
    for i in x:
        TSi=np.array(STRWM(i,a,b))*i
        TSi2=float(TSi)
        TS.append(TSi2)
    return TS
def GTRWM(x,a,b):
    # Ringwald Workshop model contribution to Gibbs Energy
    # param x Temp. range
    # param a fit term from model
    # param b fit term from model
    # return RW model contribution to Gibbs Energy
    H=np.array(HTRWM(x,a,b))-HTRWM(298.15,a,b)
    print(H[0],type(H),len(H))
    TS=np.array(TSRWM(x,a,b))
    print(TS[0],type(TS),len(TS))
    G=H-TS
    return G

# CS model H, S, G
def HTCSM(x,a,b):
    # Chen-Sundman Workshop model contribution to Enthalpy
    # param x Temp. range
    # param a fit term from model
    # param b fit term from model
    # return CS model contribution to enthalpy
    H = []
    try:
        len(x)
    except TypeError:
        Hi = 1/2* a * x**2 + 1/5* b * x**5 #Integrate Cp
        H.append(Hi)
    else:
        for i in x:
            Hi= 1/2* a * i**2 + 1/5* b * i**5 #integrate Cp
            H.append(Hi)
    return H
def STCSM(x,a,b):
    # Chen-Sundman Workshop model contribution to Entropy
    # param x Temp. range
    # param a fit term from model
    # param b fit term from model
    # return CS model contribution to entropy
    S = []
    #S0=0
    try:
        len(x)
    except TypeError:
        Si=a*x+1/4*b*x**4#+S0
        #S0=???
        S.append(Si)
    else:
        for i in x:
            Si=a*i+1/4*b*i**4
            S.append(Si)
    return S
def TSCSM(x,a,b):
    # Chen-Sundman Workshop model contribution to Entropy multiplied by Temp for Gibbs Calculation
    # param x Temp. range
    # param a fit term from model
    # param b fit term from model
    # return CS model contribution to entropy*Temp
    TS=[]
    for i in x:
        TSi=np.array(STCSM(i,a,b))*i
        TSi2=float(TSi)
        TS.append(TSi2)
    return TS
def GTCSM(x,a,b):
    # Chen-Sundman Workshop model contribution to Gibbs
    # param x Temp. range
    # param a fit term from model
    # param b fit term from model
    # return CS model contribution to Gibbs
    H=np.array(HTCSM(x,a,b))-HTCSM(298.15,a,b)
    #print(H[0],type(H),len(H))
    TS=np.array(TSCSM(x,a,b))
    #print(TS[0],type(TS),len(TS))
    G=H-TS
    return G

#' Gibbs_BCM
#'
#' @param x Temp range
#' @param A1 parameter
#' @param A2 parameter
#' @param TE1 parameter
#' @param TE2 parameter
#'
#' @return Gibbs_LCE
def GSRLCMagn(x,TE):
    if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
        G = Gein(TE,x) + GTBCM(x, RTDB_globals['k1_final'],RTDB_globals['k2_final'],RTDB_globals['alfa_final'],RTDB_globals['g_final'])
    else:
        G = Gein(TE,x) + GTBCM(x, RTDB_globals['k1_final'],RTDB_globals['k2_final'],RTDB_globals['alfa_final'],RTDB_globals['g_final']) + GM_Bosse(x)
    return G

# This is named as if it was still LCE but its not
#' @param x Temp range
#' @param TE1 parameter
#' 
#'
#' @return Enthalpy for x < TM
#Translated not checked
def HSRLCMagn(x, TE1):
    #does not work because Einstein, try a debye model?
    if(RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0 ):
        HSR =  HEin(TE1, x) + HTBCM(x, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final'])
    else:
        HSR = HEin(TE1, x) + HTBCM(x, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final']) + HM(x)
    return HSR

#' Entropy for x < TM
#'

# This is named as if it was still LCE but its not
#' @param x Temp range
#' @param TE1 parameter
#' 
#'
#' @return Entropy for x < TM
#Translated not checked
def SSRLCMagn(x, TE1):
    if(RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0 ):
        SSR =  SEin(TE1, x) + STBCM(x, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final'])
    else:
        SSR = HEin(TE1, x) + STBCM(x, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final']) + SM(x)
    return SSR

def H0(x):
    #' Calculate H0
    #'
    #' @param x Temp range
    #'
    #' @return Calculate H0
    #Translated not checked
    CP_TM = float(CpMelt())
    Hz = []
    try:
        len(x)
    except TypeError:
        H_calc = CP_TM*(x-RTDB_globals['a_sig']*np.sqrt((RTDB_globals['a_sig']**2.0+RTDB_globals['TM']**2.0-2.0*RTDB_globals['TM']*x+x**2.0)/RTDB_globals['a_sig']**2.0))+RTDB_globals['FC']*RTDB_globals['a_sig']*np.sqrt((RTDB_globals['a_sig']**2.0+RTDB_globals['TM']**2.0-2.0*RTDB_globals['TM']*x+x**2)/RTDB_globals['a_sig']**2)
        Hz.append(H_calc)
    else:
        for i in x: #[for every value of x]
            H_calc = CP_TM*(i-RTDB_globals['a_sig']*np.sqrt((RTDB_globals['a_sig']**2.0+RTDB_globals['TM']**2.0-2.0*RTDB_globals['TM']*i+i**2.0)/RTDB_globals['a_sig']**2.0))+RTDB_globals['FC']*RTDB_globals['a_sig']*np.sqrt((RTDB_globals['a_sig']**2.0+RTDB_globals['TM']**2.0-2.0*RTDB_globals['TM']*i+i**2)/RTDB_globals['a_sig']**2)
            Hz.append(H_calc)
    return Hz

#HM currently problem child
def autoH(T, def_model):
    if def_model == "CSModelE":
        print('cs')
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
            HCS=HTCSM(T,RTDB_globals['polya'],RTDB_globals['polyb'])
        else:
            HCS=HTCSM(T,RTDB_globals['polya'],RTDB_globals['polyb'])+HM(T)
        H_val=HCS+HEin(CSE,T_plot)
    elif def_model == "RWModelE":
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
        #print('rw',RTDB_globals['polya'],RTDB_globals['polyb'])
            HRW=HTRWM(T,RTDB_globals['polya'],RTDB_globals['polyb'])
        else:
            HRW=HTRWM(T,RTDB_globals['polya'],RTDB_globals['polyb'])+HM(T)
        H_val=HRW+HEin(RTDB_globals['Te_final'],T)
    elif def_model == "SRModelE":
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
            HBCM=HTBCM(T, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final'])
        else:
            HBCM=HTBCM(T, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final'])+HM(T)
        H_val=HEin(RTDB_globals['Te_final'],T)+HBCM

    return H_val

def autoS(T, def_model):
    if def_model == "CSModelE":
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
            SCS=STCSM(T,RTDB_globals['polya'],RTDB_globals['polyb'])
        else:
            SCS=STCSM(T,RTDB_globals['polya'],RTDB_globals['polyb'])+SM(T)
        H_val=HCS+SEin(CSE,T)
    elif def_model == "RWModelE":
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
            SRW=STRWM(T,RTDB_globals['polya'],RTDB_globals['polyb'])
        else:
            SRW=STRWM(T,RTDB_globals['polya'],RTDB_globals['polyb'])+SM(T)
        S_val=SRW+SEin(RTDB_globals['Te_final'],T)
    elif def_model == "SRModelE":
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
            SBCM=STBCM(T_plot, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final'])
        else:
            SBCM=STBCM(T_plot, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final'])+SM(T)
        S_val=SEin(RTDB_globals['Te_final'],T)+SBCM
    return S_val


# +GibbsM_Bosse(T) or GM_Bosse(T) do figure this out homie
def autoG(T, def_model):
    if def_model == "CSModelE":
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
        print('cs')
            GCS=GTCSM(T,RTDB_globals['polya'],RTDB_globals['polyb'])
            G_val=GCS+GEin(RTDB_globals['Te_final'],T)
        else:
            GCS=GTCSM(T,RTDB_globals['polya'],RTDB_globals['polyb'])
            G_val=GCS+GEin(RTDB_globals['Te_final'],T)+GibbsM_Bosse(T)
    elif def_model == "RWModelE":
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
            GRW=GTRWM(T,RTDB_globals['polya'],RTDB_globals['polyb'])
            G_val=GRW+GEin(RTDB_globals['Te_final'],T)
        else:
            GRW=GTRWM(T,RTDB_globals['polya'],RTDB_globals['polyb'])
            G_val=GRW+GEin(RTDB_globals['Te_final'],T)+GibbsM_Bosse(T)
    elif def_model == "SRModelE":
        print('sr')
        if RTDB_globals['bta'] == 0 or RTDB_globals['p'] == 0 or RTDB_globals['Tc'] == 0:
            GBCM=GTBCM(T_plot, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final'])
            GBCM2=[]
            for i in range(len(GBCM)):
                GBCM2.append(GBCM[i][0])
            G_val=GEin(RTDB_globals['Te_final'],T)+GBCM2
        else:
            GBCM=GTBCM(T_plot, RTDB_globals['k1_final'], RTDB_globals['k2_final'], RTDB_globals['alfa_final'], RTDB_globals['g_final'])
            GBCM2=[]
            for i in range(len(GBCM)):
                GBCM2.append(GBCM[i][0])
            G_val=GEin(RTDB_globals['Te_final'],T)+GBCM2+GibbsM_Bosse(T)
    return G_val

##Unclear what Below is for, tbh cant remember
#def fit_PEformation_energy(dbf, comps, phase_name, configuration, symmetry, datasets, ridge_alpha=None, aicc_phase_penalty=None, features=None):
#    """
#    Find suitable linear model parameters for the given phase.
#    We do this by successively fitting heat capacities, entropies and
#    enthalpies of formation, and selecting against criteria to prevent
#    overfitting. The "best" set of parameters minimizes the error
#    without overfitting.#

#    Parameters
#    ----------
#    dbf : Database
#        pycalphad Database. Partially complete, so we know what degrees of freedom to fix.
#    comps : [str]
#        Names of the relevant components.
#    phase_name : str
#        Name of the desired phase for which the parameters will be found.
#    configuration : ndarray
#        Configuration of the sublattices for the fitting procedure.
#    symmetry : [[int]]
#        Symmetry of the sublattice configuration.
#    datasets : PickleableTinyDB
#        All the datasets desired to fit to.
#    ridge_alpha : float
#        Value of the :math:`\\alpha` hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
#        with ordinary least squares regression. For now, the parameter is applied to all features.
#    aicc_feature_factors : dict
#        Map of phase name to feature to a multiplication factor for the AICc's parameter penalty.
#    features : dict
#        Maps "property" to a list of features for the linear model.
#        These will be transformed from "GM" coefficients
#        e.g., {"CPM_FORM": (v.T*symengine.log(v.T), v.T**2, v.T**-1, v.T**3)} (Default value = None)#

#    Returns
#    -------
#    dict
#        {feature: estimated_value}#

#    """
#    aicc_feature_factors = aicc_phase_penalty if aicc_phase_penalty is not None else {}
#    if interaction_test(configuration):
#        _log.debug('ENDMEMBERS FROM INTERACTION: %s', endmembers_from_interaction(configuration))
#        fitting_steps = (["CPM_FORM", "CPM_MIX"], ["SM_FORM", "SM_MIX"], ["HM_FORM", "HM_MIX"])#

#    else:
#        # We are only fitting an endmember; no mixing data needed
#        fitting_steps = (["CPM_FORM"], ["SM_FORM"], ["HM_FORM"])#

#    # create the candidate models and fitting steps
#    if features is None:
#        #Add documentation of change
#        features = OrderedDict([("CPM_FORM", (Einstein(Theta_E,v.T) + a * v.T + b * v.T**2)),
#                                ("SM_FORM", (v.T,)),
#                                ("HM_FORM", (symengine.S.One,)),
#                                ])
#    # dict of {feature, [candidate_models]}
#    candidate_models_features = build_candidate_models(configuration, features)#

#    # All possible parameter values that could be taken on. This is some legacy
#    # code from before there were many candidate models built. For very large
#    # sets of candidate models, this could be quite slow.
#    # TODO: we might be able to remove this initialization for clarity, depends on fixed poritions
#    parameters = {}
#    for candidate_models in candidate_models_features.values():
#        for model in candidate_models:
#            for coef in model:
#                parameters[coef] = 0#

#    # These is our previously fit partial model from previous steps
#    # Subtract out all of these contributions (zero out reference state because these are formation properties)
#    fixed_model = None  # Profiling suggests we delay instantiation
#    fixed_portions = [0]#

#    for desired_props in fitting_steps:
#        feature_type = desired_props[0].split('_')[0]  # HM_FORM -> HM
#        aicc_factor = aicc_feature_factors.get(feature_type, 1.0)
#        solver_qry = (where('solver').test(symmetry_filter, configuration, recursive_tuplify(symmetry) if symmetry else symmetry))
#        desired_data = get_prop_data(comps, phase_name, desired_props, datasets, additional_query=solver_qry)
#        desired_data = filter_configurations(desired_data, configuration, symmetry)
#        desired_data = filter_temperatures(desired_data)
#        _log.trace('%s: datasets found: %s', desired_props, len(desired_data))
#        if len(desired_data) > 0:
#            if fixed_model is None:
#                fixed_model = Model(dbf, comps, phase_name, parameters={'GHSER'+(c.upper()*2)[:2]: 0 for c in comps})
#            config_tup = tuple(map(tuplify, configuration))
#            calculate_dict = get_prop_samples(desired_data, config_tup)
#            sample_condition_dicts = _get_sample_condition_dicts(calculate_dict, list(map(len, config_tup)))
#            weights = calculate_dict['weights']
#            assert len(sample_condition_dicts) == len(weights)#

#            # We assume all properties in the same fitting step have the same
#            # features (all CPM, all HM, etc., but different ref states).
#            # data quantities are the same for each candidate model and can be computed up front
#            data_qtys = get_data_quantities(feature_type, fixed_model, fixed_portions, desired_data, sample_condition_dicts)#

#            # build the candidate model transformation matrix and response vector (A, b in Ax=b)
#            feature_matricies = []
#            data_quantities = []
#            for candidate_coefficients in candidate_models_features[desired_props[0]]:
#                # Map coeffiecients in G to coefficients in the feature_type (H, S, CP)
#                transformed_coefficients = list(map(feature_transforms[feature_type], candidate_coefficients))
#                if interaction_test(configuration, 3):
#                    feature_matricies.append(_build_feature_matrix(sample_condition_dicts, transformed_coefficients))
#                else:
#                    feature_matricies.append(_build_feature_matrix(sample_condition_dicts, transformed_coefficients))
#                data_quantities.append(data_qtys)#

#            # provide candidate models and get back a selected model.
#            selected_model = select_model(zip(candidate_models_features[desired_props[0]], feature_matricies, data_quantities), ridge_alpha, weights=weights, aicc_factor=aicc_factor)
#            selected_features, selected_values = selected_model
#            parameters.update(zip(*(selected_features, selected_values)))
#            # Add these parameters to be fixed for the next fitting step
#            fixed_portion = np.array(selected_features, dtype=np.object_)
#            fixed_portion = np.dot(fixed_portion, selected_values)
#            fixed_portions.append(fixed_portion)
#    return parameters