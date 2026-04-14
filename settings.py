import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

aplications = {'CP': (3.7, 3), 
               'DR': (15, 22),
               'EV': (48, 80)}

path = 'Experiments/'

n_islands = 5
i_pop=40 #40
f_mig=40 #40
n_mig=2  #2
f_eval=20_000 #20_000
history_points=20 #20

var_keys = ['C', 'la', 'lp', 'lo', 'ln', 'lz', 'Lh', 'Rp', 'Rn', 'Rcell', 'efp', 'efo', 'efn', 'mat', 'Ns', 'Np']
oFn_keys = ['SpecificEnergy', 'SEIGrouth', 'TempIncrease', 'Price']
cFn_keys = ['UpperViolation', 'LowerViolation', 'VolFracViolation']
Fn_keys = oFn_keys+cFn_keys

limits = {  'C':[0.2, 4.0],
            'la':[12e-6, 30e-6],
            'lp':[40e-6, 250e-6],
            'lo':[10e-6, 100e-6],
            'ln':[40e-6, 250e-6],
            'lz':[12e-6, 30e-6],
            'Lh':[40e-3, 100e-3],
            'Rp':[0.2e-6, 20e-6],
            'Rn':[0.5e-6, 50e-6],
            'Rcell':[4e-3, 25e-3],
            'efp':[0.01, 0.99],
            'efo':[0.01, 0.99],
            'efn':[0.01, 0.99],
            'mat':['LCO','LFP'],
            'Ns':[1, 100],
            'Np':[1, 100],}

nadir = [0.0, 5, 20.0, 4e6]
max_presure = 5e9

e = [[1,      1e-12,  1e-12,  1e-12],
    [1e-12,  1,      1e-12,  1e-12],
    [1e-12,  1e-12,  1,      1e-12],
    [1e-12,  1e-12,  1e-12,  1    ],
    [0.25,   0.25,   0.25,   0.25 ]]

r_dirs = get_reference_directions("energy", 4, 20, seed=1)
ref_dirs = np.where(r_dirs==0, 1e-12, r_dirs)

kappa_parameter = 1e6

mutation_probability = 1/16