import numpy as np
import random
from smt.sampling_methods import LHS
import pandas as pd

from indicators import individualContribution, RieszEnergy
from settings import var_keys, oFn_keys, cFn_keys, Fn_keys, nadir, max_presure, e, ref_dirs

"""
==================================================================================================================================================================
Population Management
==================================================================================================================================================================
"""

def flatenPop(Pop):

    flat_pop = pd.concat(Pop, axis=0, ignore_index=True)

    return flat_pop

def dividePop(Pop, i_pop, n_islands):

    # if len(Pop) == (i_pop*n_islands):

    #     div_pop = [Pop.iloc[i:i+i_pop] for i in range(0, len(Pop), i_pop)]
        
    #     return div_pop

    # else:
    #     raise Exception('Pupulation size do not match')
    
    return [Pop.iloc[i:i+i_pop] for i in range(0, len(Pop), i_pop)]

"""
==================================================================================================================================================================
Sorting
==================================================================================================================================================================
"""

def nonDomSort_recursive(F, index):
    fltr = np.logical_not((F[:, None] >= F).all(axis=2).sum(axis=1) == 1)
    F_dom = F[fltr]
    if len(F)==len(F_dom) or len(F_dom)==0:
        return F, index
    else:
        return nonDomSort_recursive(F_dom, index[fltr])

def nonDomSort(Q:pd.DataFrame):
    F = Q[oFn_keys].to_numpy()
    return nonDomSort_recursive(F, np.arange(len(F)))

"""
==================================================================================================================================================================
Dominance
==================================================================================================================================================================
"""

def nonDominated(P, keys=oFn_keys):
    valid = P[keys].values
    fltr = ((valid[:, None] >= valid).all(axis=2).sum(axis=1) == 1)
    return P[fltr]

def validFilter(P):
    return P[(P[cFn_keys]<=0).all(axis=1)]

"""
==================================================================================================================================================================
Evaluate
==================================================================================================================================================================
"""

def evaluate(x, problem):
    oFn, cFn = problem.evaluate(x)

    res = {
        "SpecificEnergy": oFn[0],
        "SEIGrouth": oFn[1],
        "TempIncrease": oFn[2],
        "Price": oFn[3],
        "UpperViolation": cFn[0],
        "LowerViolation": cFn[1],
        "VolFracViolation": cFn[2],
    }
    
    res.update(x)

    return res

"""
==================================================================================================================================================================
Initialize
==================================================================================================================================================================
"""

limits = np.array([[0.2, 4.0],
                [12e-6, 30e-6],
                [40e-6, 250e-6],
                [10e-6, 100e-6],
                [40e-6, 250e-6],
                [12e-6, 30e-6],
                [40e-3, 100e-3],
                [0.2e-6, 20e-6],
                [0.5e-6, 50e-6],
                [4e-3, 25e-3],
                [0.01, 0.6],
                [0.01, 0.6],
                [0.01, 0.6]])

def initializePop(i_pop, n_islands, problem):
    
    pop_size = i_pop*n_islands
    
    sampling = LHS(xlimits=limits)
    smpls = sampling(pop_size)

    smpls2 = np.random.dirichlet((1,1,1,1), pop_size)
    smpls2 = smpls2[:,:3]

    i = 0
    Pop = []
    for _ in range(n_islands):
        pop = []

        for _ in range(i_pop): 

            x = smpls[i]
            x2 = smpls2[i]

            var = {
                "C": x[0],
                "la": x[1],
                "lp": x[2],
                "lo": x[3],
                "ln": x[4],
                "lz": x[5],
                "Lh": x[6],
                "Rp": x[7],
                "Rn": x[8],
                "Rcell": x[9],
                "efp": x2[0],
                "efo": x2[1],
                "efn": x2[2],
                "mat": random.choice(['LCO','LFP']),
                "Ns": random.choice(range(1,101)),
                "Np": random.choice(range(1,101)),
            }

            individual = evaluate(var, problem)
            
            i += 1
            pop.append(individual)
        
        Pop.append(pd.DataFrame(pop))

    return Pop

"""
==================================================================================================================================================================
CSR
==================================================================================================================================================================
"""

def achivement(f, f_star, c, e, presure):

    ref = np.empty((len(e),len(f[0])))
    ref_index = np.empty(len(e))

    for i in range(len(e)):

        ref[i] = f[0]
        best_max = np.inf

        for h in range(len(f)):
            maximum = -np.inf
            for j in range(2):
                maximum = max(maximum, (f[h][j]-f_star[j])/e[i][j])
            if (maximum + presure*np.sum(c[h])**2) < best_max:
                ref[i] = f[h]
                ref_index[i] = h
                best_max = maximum

    ref_index = np.unique(ref_index).astype(int)

    return ref, ref_index

def obtain_aprox(ref, ref_dirs, alpha = 1):
    ak = (((np.sum(ref_dirs**alpha, axis=1))**(1/alpha)).reshape(len(ref_dirs),1))
    ak = np.where(ak==0, 1e-12, ak)
    y = ref_dirs/ak
    return y * (np.max(ref, axis=0) - np.min(ref, axis=0)) + np.min(ref, axis=0)

def n2one_dominates(y, ref):
    truth = y <= ref
    return any(np.logical_and(np.logical_and(truth[:,0],truth[:,1]),np.logical_and(truth[:,2],truth[:,3])))


def referenceCSR(P:pd.DataFrame):

    f = P[oFn_keys].values
    c = P[cFn_keys].values
    c = np.where(c < 0, 0, c)

    f_star = np.min(f, axis=0)

    p = max_presure
    index = []
    while len(index)<len(oFn_keys)+1:
        ref_pnts, index = achivement(f, f_star, c, e, p)
        p *= 0.1
        if p<1:
            break

    alpha = 1
    y = obtain_aprox(ref_pnts, ref_dirs, alpha)

    while n2one_dominates(y, ref_pnts[-1]):
        alpha += 0.05
        if alpha >= 1e3:
            break
        y = obtain_aprox(ref_pnts, ref_dirs, alpha)

    while not(n2one_dominates(y, ref_pnts[-1])):
        alpha -= 0.05
        if alpha <= 0:
            break
        y = obtain_aprox(ref_pnts, ref_dirs, alpha)

    y = y[np.logical_not(np.isnan(y).any(axis=1))]
    y = y[np.logical_not(np.isinf(y).any(axis=1))]

    return y

"""
==================================================================================================================================================================
NDR
==================================================================================================================================================================
"""
rsze = RieszEnergy()

def referenceNDR(P:pd.DataFrame, p_ref=None):

    size = len(P)

    valid = validFilter(P)

    if len(valid)<1:
        valid = P[Fn_keys].values
        fltr = ((valid[:, None] >= valid).all(axis=2).sum(axis=1) == 1)
        valid = valid[fltr]
        valid = valid[:,:-3]

    else:
        valid = P[oFn_keys].values

    if isinstance(p_ref, np.ndarray):
        valid = np.unique(np.append(valid, p_ref, axis=0), axis=0)

    fltr = ((valid[:, None] >= valid).all(axis=2).sum(axis=1) == 1)
    ref = valid[fltr]
    
    if len(ref) < 1:
        raise Exception('Reference has no points')
    
    if len(ref) > size:
        n_delete = int(len(ref)-size)
        total_c = rsze(ref)
        r = np.argsort(individualContribution(rsze,total_c,ref,None))[-n_delete:]
        ref = np.delete(ref,r,axis=0)
    
    return ref

"""
==================================================================================================================================================================
Reference Update
==================================================================================================================================================================
"""

def referenceUpdate(Pop, Ref, ref_mode):

    if ref_mode == 'CSR':
        Ref = referenceCSR(Pop)

    elif ref_mode == 'NDR':
        Ref = referenceNDR(Pop, Ref)
    
    else:
        raise Exception('Undefined reference method')

    return Ref


def referenceGeneration(Pop, ref_mode):
    
    Ref = []
    
    for sub_pop in Pop:
        sub_ref = referenceUpdate(sub_pop, None, ref_mode)
        Ref.append(sub_ref)
    
    return Ref

"""
==================================================================================================================================================================
Contribution
==================================================================================================================================================================
"""
def lessContribution(indicator, Rt, ref, indexes):
    if indicator.name=='HV':
        ref = np.min(Rt, axis=0)
    g_contr = indicator(Rt, ref)
    index = np.argmin(individualContribution(indicator,g_contr,Rt,ref))
    return indexes[index]

"""
==================================================================================================================================================================
Selection
==================================================================================================================================================================
"""

def survivorSelection(P:pd.DataFrame, ref, pop_size, I):

    while len(P) > pop_size:

        if len(P[(P[cFn_keys]<=0).all(axis=1)]) < pop_size:

            r = np.argmax((P[cFn_keys]>0).sum(axis=1))

        else: 
            Rt, Rt_indexes = nonDomSort(P)
            if len(Rt_indexes) > 2:
                r = lessContribution(I,Rt,ref,Rt_indexes)
            else:
                r = Rt_indexes
        
        P = P.drop(r).reset_index(drop=True)

    return P

"""
==================================================================================================================================================================
Migration
==================================================================================================================================================================
"""

def migrate(P, n_islands, i_pop, n_mig):

    Q = [pd.DataFrame() for _ in range(n_islands)]
    for j in range(n_islands):
        for i in range(n_islands):
            if i != j:
                q = random.sample(P[i].index.tolist(),n_mig)
                Q[j] = pd.concat([Q[j], P[i].loc[q]])
                P[i] = P[i].drop(q)

    for i in range(n_islands):
        P[i] = pd.concat([P[i], Q[i]], ignore_index=True)
    
    return P