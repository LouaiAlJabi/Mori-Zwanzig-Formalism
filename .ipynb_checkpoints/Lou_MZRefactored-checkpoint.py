import mori_zwanzig as mz
from sklearn.linear_model import LinearRegression
import numpy as np
import time as tm
import concurrent.futures as cf
import Lou_Bootstrap as lb


    
def GenSamples(num,num_sims,seed):
    """
    Generates a 'num' amount of samples and renum_simsturns the results as a list of all the samples.
    
    args:
    num = The resulting number of samples
    num_sims = number of simulations, determined by the data provided
    seed = The seed for randomizing
    """
    np.random.seed(seed)
    
    list_ = []
    for _ in range(num):
        list_.append((_+1,list(np.random.choice(range(num_sims), size = num_sims, 
                                      replace = True))))
    list_ = np.array(list_,dtype=object)
    return list_

def EvenSplit(list_, chunk):
    """
    The Function takes in a list of even length and splits it into an even number of chunks.
    
    args:
    list_ = The list to split
    chunk = The number of chunks to split into
    """
    if len(list_) % 2 == 0 and chunk % 2 == 0:
        list_ = np.array_split(list_, chunk)
        return [i for i in list_]
    else:
        raise ValueError("The list can't be split evenly into an even number." 
                         + " Check its length and the requested chunk size, both should be even.")

def AutoCorrelationFunctions(velocity,Force):

    """
    This function calculates correlation functions using the provided velocities and forces

    args:
    velocity = the file of velocities
    Froce = the file of forces
    """
    start = tm.time()
    with cf.ProcessPoolExecutor() as exe:
        future_vacf = exe.submit(mz.compute_correlation_functions, velocity,None,True,False,False)
        future_Facf = exe.submit(mz.compute_correlation_functions, Force,None,True,False,False)
        future_vFcross = exe.submit(mz.compute_correlation_functions, velocity,Force,True,False,False)
    
        v_acf = future_vacf.result()
        F_acf = future_Facf.result()
        vF_cross = future_vFcross.result()
    
    print(f"multiprocessing took {(round((tm.time()-start)/60,2))} mins to finish correlation functions")    
    return [v_acf, F_acf, vF_cross]

#Generate samples
#split samples
#set variables
#Calc ACFs
#Calc Diff with Einstein
#Calc Diff directly (np.trapz)
#Calc Diff with MZ
    #boot ACFs and store em
    #boot K and store em
#Calc 