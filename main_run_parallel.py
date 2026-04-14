import numpy as np
import random
import time
from multiprocessing import Pool

from NSGA3_main_run import runNSGA3
from IMICA_main_run import runIMICA

TOTAL_SEEDS = 30
PARALLEL_SEEDS = 10
APP = ['CP', 'DR', 'EV']
REFERENCE_MODES = ['NDR','CSR']
SELECTION_MODES = ['Rand','Roulette']

def run_seed(exp):

    random.seed(exp)
    np.random.seed(exp)
    
    for app in APP:
        runNSGA3(exp, app)
    
    for app in APP:
        for refer in REFERENCE_MODES:
            for selct in SELECTION_MODES:
                runIMICA(app, exp, refer, selct)

if __name__ == "__main__":
    start = time.time()

    seeds = list(range(1, TOTAL_SEEDS + 1))

    print(f"Ejecutando {TOTAL_SEEDS} seeds en batches de {PARALLEL_SEEDS}")

    for i in range(0, TOTAL_SEEDS, PARALLEL_SEEDS):

        batch = seeds[i:i + PARALLEL_SEEDS]

        print(f"\n========== NUEVO BATCH: {batch} ==========")

        with Pool(processes=len(batch)) as pool:
            pool.map(run_seed, batch)

        print(f"========== BATCH {batch} COMPLETADO ==========\n")
        time.sleep(2)

    print("TODAS LAS SEEDS HAN TERMINADO")

end = time.time()
print(f"Total runtime: {(end - start)/60:.2f} minutes")