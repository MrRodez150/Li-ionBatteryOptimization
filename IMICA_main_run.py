from surr_P2D import BatteryP2D
from indicators import SMS, R2, IGDplus, EpsPlus, DeltaP
from IMICA import IMICA
from settings import path, i_pop, f_mig, n_mig, f_eval, history_points, aplications
from resultsManagement import recoverEvaluation, recoverPop


"""
==================================================================================================================================================================
Single Run
==================================================================================================================================================================
"""

def runIMICA(App, exp, ref_mode, select_mode):

    expName = f'IMICA_{App}_{ref_mode}_{select_mode}_E{exp}'
    print(expName)

    eval, file_found = recoverEvaluation(expName)
  
    if file_found:
        s_gen, P = recoverPop(eval, expName)
    else:
        P = None
        s_gen = 0

    IMICA(  ExpName=expName,
            ref_mode=ref_mode, 
            selct_mode=select_mode, 
            problem=BatteryP2D(aplications[App][0], aplications[App][1]), 
            indicators=[SMS(),R2(),IGDplus(),EpsPlus(),DeltaP()],
            Pop=P, 
            start_gen=s_gen, 
            i_pop=i_pop, 
            f_mig=f_mig, 
            n_mig=n_mig, 
            f_eval=f_eval, 
            history_points=history_points,
            verbose=True)



if __name__ == "__main__":

    app = 'CP'
    refer = 'NDR'
    selct = 'Rand'
    exp = 4

    runIMICA(app, exp, refer, selct)