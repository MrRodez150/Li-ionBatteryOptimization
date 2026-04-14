import random
import numpy as np
import pandas as pd

from settings import var_keys, cFn_keys, oFn_keys, kappa_parameter, mutation_probability, limits
from IMICA_utils import evaluate

"""
==================================================================================================================================================================
Selection
==================================================================================================================================================================
"""

def calcProbabilities(pop):

    probabilities = []
    
    for _, ind in pop.iterrows():

        volume = 1
        volume2 = kappa_parameter

        for oFn in oFn_keys:
            v = ind[oFn]
            if v < 0:
                volume /= -v
            else:
                volume *= v

        for cFn in cFn_keys:
            v = ind[cFn]
            if v > 0:
                volume2 *= v

        volume += volume2
        probabilities = np.append(probabilities, volume)

    probabilities = np.max(probabilities)*1.1 - probabilities
    probabilities = probabilities/np.sum(probabilities)

    return probabilities 

def rouletteSelection(pop:pd.DataFrame):

    probabilities = calcProbabilities(pop)
    
    parents_index = np.random.choice(range(len(pop)), 2, False, p=probabilities)

    return parents_index

def selectParents(P, method):
    pop_size = len(P)

    if method == 'Rand':
        index = np.random.choice(range(0,pop_size), size=2, replace=False)

    elif method == 'Roulette':
        index = rouletteSelection(P)

    else:
        raise Exception('Undefined selection method')

    parents = P.iloc[index]
    
    return parents

"""
==================================================================================================================================================================
Simulated Binary Crossover
==================================================================================================================================================================
"""

def SBX(p1,p2,rnd=False):
    u = np.random.rand()
    if u <= 0.5:
        beta = (2*u)**(1/21)
    else:
        beta = (1/(2-2*u))**(1/21)

    c1 = ((1+beta)*p1 + (1-beta)*p2)/2
    c2 = ((1-beta)*p1 + (1+beta)*p2)/2

    if rnd:
        return round(c1), round(c2)
    
    else:
        return c1, c2
    

"""
==================================================================================================================================================================
Polinomial Mutation
==================================================================================================================================================================
"""

def PM(p,delta,rnd=False):
    u = np.random.rand()
    if u < 0.5:
        beta = (2*u)**(1/21) - 1
    else:
        beta = 1 - (2-2*u)**(1/21)

    c = p + beta * delta

    if rnd:
        return round(c)
    else:
        return c, beta

"""
==================================================================================================================================================================
Repair
==================================================================================================================================================================
"""

def repair(var,lmts):
    #Lower bound
    if var < lmts[0]:
        var = 2*lmts[0] - var
        var = repair(var,lmts)
    #Upper bound
    if var > lmts[1]:
        var = 2*lmts[1] - var
        var = repair(var,lmts)

    return var

def fixOne (son, parent, beta):
    lam = np.ones(4)
    s = 0
    p = 0
 
    for i, ef in enumerate(['efp', 'efo', 'efn']):
        s += son[ef]
        p += parent[ef]

        if (son[ef] - parent[ef]) < 0:

            lam[i] = parent[ef] / (parent[ef] - son[ef])

    if s - p > 0:

        lam[3] =  (1 - p) / (s - p)

    for i in ['efp', 'efo', 'efn']:

        son[i] = parent[i] + np.min(lam)*(1-beta)*(son[i]-parent[i])
        son[i] = 0.97 * son[i] + 0.01


    return son


def simplexRepair(sons, parents, beta):
    for j in range(2):
        sons[j] = fixOne(sons[j], parents[j], beta[j])
    return sons[0], sons[1] 

"""
==================================================================================================================================================================
Mating
==================================================================================================================================================================
"""

def generateOffspring(parents, problem):
    
    s1 = {}
    s2 = {}
    beta1 = 0
    beta2 = 0

    for var in var_keys:

        p1 = parents[var].iloc[0]
        p2 = parents[var].iloc[1]

        if var == 'mat':

            c1, c2 = random.sample([p1,p2],2)

            if np.random.rand() < mutation_probability:
                c1 = random.choice(limits[var])
            if np.random.rand() < mutation_probability:
                c2 = random.choice(limits[var])

        elif var=='Ns' or var=='Np':

            c1, c2 = SBX(p1,p2,True)

            lim = limits[var]

            if np.random.rand() < mutation_probability:
                c1 = PM(c1,lim[1]-lim[0],True)
            c1 = repair(c1,lim)

            if np.random.rand() < mutation_probability:
                c2 = PM(c2,lim[1]-lim[0],True)
            c2 = repair(c2,lim)

        elif var=='efp'or var=='efo' or var=='efn':

            c1, c2 = SBX(p1,p2,False)
            
            lim = limits[var]

            if np.random.rand() < mutation_probability:
                c1, beta1 = PM(c1,lim[1]-lim[0],False)
            if np.random.rand() < mutation_probability:
                c2, beta2 = PM(c2,lim[1]-lim[0],False)

        else:
            c1, c2 = SBX(p1,p2,False)
            
            lim = limits[var]

            if np.random.rand() < mutation_probability:
                c1, _ = PM(c1,lim[1]-lim[0],False)
            if np.random.rand() < mutation_probability:
                c2, _ = PM(c2,lim[1]-lim[0],False)
            
            c1 = repair(c1,lim)
            c2 = repair(c2,lim)
        
        s1.update({var: c1})
        s2.update({var: c2})

    s1, s2 = simplexRepair([s1, s2], [parents.iloc[0], parents.iloc[1]], [beta1, beta2])
    
    s1 = evaluate(s1, problem)
    s2 = evaluate(s2, problem)

    return pd.DataFrame([s1,s2])
    