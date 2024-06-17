import mori_zwanzig as mz
from sklearn.linear_model import LinearRegression
import numpy as np
import time as tm
import concurrent.futures as cf
import Lou_Bootstrap as lb


#Generate samples
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

#split samples
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

#Calc ACFs
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

#Load DataFiles
def LoadDatafile(velocity_file,force_file,sqd_file):
    v = np.load(velocity_file)
    F = np.load(force_file)
    sq_displacement = np.load(sqd_file)

#set variables
def VarSetter(part_mass, dt, KB, T, Omega):
    if part_mass < 0:
        raise ValueError("Mass can't be negative")
    return [part_mass, dt, KB, T, Omega]

#Calc Diff with Einstein
def diffusion_constant_from_MSD(start_fit_time, end_fit_time,time,sq_displacement, sample): #24.,28.
    print("Started")
    start_fit_index = np.argmin(np.abs(time - start_fit_time))
    end_fit_index = np.argmin(np.abs(time - end_fit_time))
    dict_ = dict()
    msd = np.mean(sq_displacement[:, sample[1], :], axis=(1,2))
    model = LinearRegression()
    model.fit(time[start_fit_index:end_fit_index].reshape((-1,1)), 
                msd[start_fit_index:end_fit_index])
    m = model.coef_[0]
    D = m/2
    dict_['sample_ID'] = sample[0]
    dict_['Einstein'] = D
    print("Finished")
    return dict_



def DiffConstant_Directly(v_acf_boot,time):
    return np.trapz(v_acf_boot,time)
  
#Calc Diff with MZ
    #boots and calcs K to store em
    #loops through cutoffs and calcs MZ and v_acf_mz
    #returns three dicts,one for mz, one for the Correlation functions, and one for K
    #args:
        #v_acf_boot
        #F_acf_boot
        #vF_cross_boot
        #particle_mass
        #kB
        #T
        #dt
        #num_steps
        #Omega
        #cutoff list
        #time

def DiffConstant_MZ(part_mass, dt, KB, T, Omega, K, C0, time, num_steps, cutoffs,sam_ID,iboot):
    mzDict = dict()
    acfDict = dict()

    for id_ in sam_ID:
        mzDict[id_] =  dict()
        acfDict[id_] = dict()

    for key,value in mzDict.items():  
        for cutoff in cutoffs:
            value[cutoff] = 0.

    for key,value in acfDict.items():  
        for cutoff in cutoffs:
            value[cutoff] = []
    
    for cutoff in cutoffs:
            v_acf_cutoff = mz.integrate_C(dt, num_steps, Omega, K, C0 = C0, cutoff = cutoff)
            
            #results['MZ'][sam_ID[iboot]][cutoff] = np.trapz(v_acf_cutoff, time)
            #results['v_acf_mz'][sam_ID[iboot]][cutoff] = v_acf_cutoff
    
    return None


#bootstrap
    #boots and calcs K to stores em
    #boots and calcs acf_boot to store em
    #