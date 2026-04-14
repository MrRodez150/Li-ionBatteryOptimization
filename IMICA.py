import numpy as np
import pandas as pd
import timeit
from threading import Thread

from IMICA_utils import initializePop, referenceGeneration, referenceUpdate, survivorSelection, migrate, flatenPop, validFilter, nonDominated
from mating import selectParents, generateOffspring
from resultsManagement import saveFiles

class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return

"""
==================================================================================================================================================================
IBMOEA
==================================================================================================================================================================
"""

def IBMOEA(problem, P:pd.DataFrame, I, ref, ref_mode, f_mig, pop_size, select_mode):

    for g in range(f_mig):

        parents = selectParents(P, select_mode)
        sons = generateOffspring(parents, problem)
        P = survivorSelection(pd.concat([P, sons], ignore_index=True), ref, pop_size, I)
        ref = referenceUpdate(P, ref, ref_mode)

    return P, ref

def parallelIBMOEA(problem, pop, indicators, ref, n_islands, i_pop, islands, f_mig, ref_mode, select_mode):

    for i in range(1,n_islands):    
        islands[i] = ThreadWithReturnValue(target=IBMOEA, args=(problem, pop[i], indicators[i], ref[i], ref_mode, f_mig, i_pop, select_mode))
        islands[i].start()

    pop[0], ref[0] = IBMOEA(problem=problem, P=pop[0], I=indicators[0], ref=ref[0], f_mig=f_mig, pop_size=i_pop, ref_mode=ref_mode, select_mode=select_mode)

    for i in range(1,n_islands):
        pop[i], ref[i] = islands[i].join()

    return pop, ref


"""
==================================================================================================================================================================
IMICA
==================================================================================================================================================================
"""

def IMICA(ExpName, ref_mode, selct_mode, problem, indicators, Pop, start_gen, i_pop, f_mig, n_mig, f_eval, history_points, verbose):
    
    n_islands = len(indicators)
    islands = [None]*n_islands
    gens = int(np.ceil(f_eval/n_islands/f_mig/2))
    
    samples = np.linspace(0,gens,history_points,dtype=int)
    
    if start_gen == 0:

        start = timeit.default_timer()

        Pop = initializePop(i_pop, n_islands, problem)
    
    Ref = referenceGeneration(Pop, ref_mode)

    if start_gen == 0:     

        end = timeit.default_timer()

        saveFiles(flatenPop(Pop), 0, n_islands*f_mig, end-start, samples, ExpName)

        if verbose:
            print(f'{ExpName} | Init | time: {end-start}')
    
    for g in range(start_gen, gens):

        start = timeit.default_timer()

        Pop, Ref = parallelIBMOEA(problem, Pop, indicators, Ref, n_islands, i_pop, islands, f_mig, ref_mode, selct_mode)

        Pop = migrate(Pop, n_islands, i_pop, n_mig)

        end = timeit.default_timer()

        saveFiles(flatenPop(Pop), g+1, 2*(g+1)*n_islands*f_mig, end-start, samples, ExpName)
        
        if verbose:
            print(f'{ExpName} | Generation: \t {g+1} / {gens} | time: {end-start}')

    # solutions = nonDominated(flatenPop(Pop))

    # valid_solutions = validFilter(solutions)

    if verbose:
            print(f'{ExpName}: Solutions found')

    return Pop