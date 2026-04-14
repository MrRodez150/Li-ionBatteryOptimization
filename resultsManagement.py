import os
import csv
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

from settings import oFn_keys, cFn_keys, path, nadir, i_pop, n_islands
from IMICA_utils import dividePop

ind_HV = HV(ref_point=np.array(nadir))

def savePopulation(P, expName, gen):

    P.to_csv(path + expName + f'_pop_{gen}.csv', index=False)

def removeEvaluation(gen, samples, expName):
    
    if gen > 0 and np.isin(gen-1, samples, invert=True):
        try:
            os.remove(path + expName + f'_pop_{gen-1}.csv')
        except FileNotFoundError:
            pass

def saveEvaluation(P, gen, n_eval, time, expName):

    f = P[oFn_keys].values
    c = P[cFn_keys].values
    c = np.where(c < 0, 0, c)
    c = np.sum(c, axis=1)
    vfv = P['VolFracViolation'].values
    vfv = np.where(vfv < 0, 0, vfv)
    valid = f[np.where(c==0)]
    fltr = ((valid[:, None] >= valid).all(axis=2).sum(axis=1) == 1)
    n_nds = valid[fltr]

    if len(valid) > 0:
        v_HV = ind_HV(valid)
    else:
        v_HV = 0

    res = [gen, 
           n_eval, 
           ind_HV(f),
           len(valid),
           len(n_nds), 
           v_HV,
           np.min(c), 
           np.mean(c), 
           np.max(c), 
           np.min(vfv), 
           np.mean(vfv), 
           np.max(vfv), 
           time]
    
    with open(path + expName + '_results.csv', 'a') as resf:
        writer = csv.writer(resf)
        writer.writerow(res)

def saveFiles(P, gen, n_eval, time, samples, expName):
    
    savePopulation(P, expName, gen)

    removeEvaluation(gen, samples, expName)

    saveEvaluation(P, gen, n_eval, time, expName)

    return True

"""
==================================================================================================================================================================
Recovery
==================================================================================================================================================================
"""

def recoverEvaluation(expName, pth=path):

    try:
        res = pd.read_csv(pth + expName + '_results.csv')
        print('Checkpoint found, resuming')
        return res, True

    except FileNotFoundError:
        print('No checkpoint found, starting over')
        with open(path + expName + '_results.csv', 'a') as resf:
                writer = csv.writer(resf)
                writer.writerow(["n_Gen", "n_Eval", "g_HV",  "n_valid", "n_nds", "v_HV", "min_CV", "mean_CV", "max_CV", "min_VFV", "mean_VFV", "max_VFV", "time"])
                return None, False

def recoverPop(file, expName):

    s_gen = file['n_Gen'].max()
    if str(s_gen) != 'nan':
        P = pd.read_csv(path + expName + f'_pop_{s_gen}.csv')
        P = dividePop(P, i_pop, n_islands)
    else:
        P = None
        s_gen = 0

    return s_gen, P