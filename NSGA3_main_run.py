import csv
import timeit
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

from surr_P2D import BatteryP2D_pymoo
from settings import nadir, path, n_islands, i_pop, f_eval, history_points, aplications
from IMICA_main_run import recoverEvaluation

header = ['C','la','lp','lo','ln','lz','Lh','Rcell','Rp','Rn','efp','efo','efn','mat','Ns','Np','SpecificEnergy','SEIGrouth','TempIncrease','Price','UpperViolation','LowerViolation','VolFracViolation']
objFun = ['SpecificEnergy','SEIGrouth','TempIncrease','Price']
constFun = ['UpperViolation','LowerViolation','VolFracViolation']

def stepSolver(algorithm, path):

    while algorithm.has_next():
    
        start = timeit.default_timer()
        algorithm.next()
        end = timeit.default_timer()

        if algorithm.result().F is not None:
            valid = algorithm.result().F
            n_nds = len(valid)
            hv = ind(algorithm.result().F)
            
        else:
            n_nds = 0
            hv = 0
            
        F = algorithm.result().pop.get("F")
        
        res = [ algorithm.n_gen-1, 
                algorithm.evaluator.n_eval, 
                ind(F), 
                sum(algorithm.pop.get("cv") <= 0),
                n_nds, 
                hv, 
                algorithm.pop.get("cv").min(), 
                algorithm.pop.get("cv").mean(), 
                algorithm.pop.get("cv").max(), 
                algorithm.pop.get("G")[2].min(),
                algorithm.pop.get("G")[2].mean(),
                algorithm.pop.get("G")[2].max(),
                end - start]

        with open(path + expName + f'_results.csv', 'a') as resf:
            writer = csv.writer(resf)
            writer.writerow(res)
        
        if algorithm.n_gen in pop_pnts:

            x = algorithm.pop.get("X")
            f = algorithm.pop.get("F")
            g = algorithm.pop.get("G")

            with open(path + expName + f'_pop_{algorithm.n_gen}.csv', 'w') as popf:
                
                writer = csv.DictWriter(popf, fieldnames = header)
                writer.writeheader()

                for i in range(len(x)):
                    xfg_dict = {}
                    xfg_dict.update(x[i])
                    xfg_dict.update(dict(zip(objFun, f[i])))
                    xfg_dict.update(dict(zip(constFun, g[i])))
                    writer.writerow(xfg_dict)



def runNSGA3(exp, app, popul=i_pop*n_islands, func_eval=f_eval, h_p=history_points, pth=path, verbose=True):

    global expName, pop_pnts, ind

    expName = f'NSGA3_{app}_E{exp}'
    print(expName)

    res, skip = recoverEvaluation(expName, pth)

    if not(skip):

        ref_point = np.array(nadir)
        ind = HV(ref_point=ref_point)

        gens = int(func_eval/popul)
        pop_pnts = np.linspace(2, gens, h_p, dtype = "int")


        problem = BatteryP2D_pymoo(aplications[app][0], aplications[app][1])

        ref_dirs = get_reference_directions("energy", 4, popul)

        termination = get_termination("n_gen", gens)

        algorithm = NSGA3(pop_size=popul,
                    sampling=MixedVariableSampling(),
                    mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                    eliminate_duplicates=MixedVariableDuplicateElimination(),
                    ref_dirs=ref_dirs)

        algorithm.setup(problem, termination=termination, verbose=verbose)

        res = stepSolver(algorithm, pth)

    else: 
        print("All done")

    return res


if __name__ == "__main__":
    exp = 1
    app = 'EV'

    runNSGA3(exp, app, pth='Tests/')