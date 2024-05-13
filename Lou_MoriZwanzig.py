import mori_zwanzig as mz
from sklearn.linear_model import LinearRegression
import numpy as np
import time as tm
from collections import defaultdict
import concurrent.futures as cf
from IPython.display import clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import pickle



def main(sample_splits,seed,num_boots):
    """
    A function that does the bootstrapping in parallel over 16 cores.
    
    args:
    sample_splits = number of sample splits, multiplication of 16
    """
    if sample_splits % 16 != 0:
        raise ValueError("The number of samples must be a multiple of 16.")


    part_mass, dt, KB, T, Omega,seed = VarSetter(80.,0.005,1.,1.,0,seed)
    v,F,sq_displacement = LoadDatafile('velocities.npy','forces.npy','sq_displacement_stat.npy')
    v_acf, F_acf, vF_cross = AutoCorrelationFunctions(v,F,sq_displacement)
    num_steps, num_sims, num_dimensions = v.shape
    steps_to_integrate = num_steps
    time = np.linspace(0, dt*num_steps, num_steps)
    num_bootstraps = num_boots
    cutoffs = np.arange(1,29.)

    samples = GenSamples(sample_splits,num_sims,seed)
    splits = EvenSplit(samples, 16)
    list_ = []
    with cf.ProcessPoolExecutor(16) as exe:
        futures = [exe.submit(Bootstrap, splits[i],cutoffs,v_acf,F_acf,vF_cross,part_mass,dt,KB,T,Omega,num_steps,time) for i in range(len(splits))]
        for future in cf.as_completed(futures):
            list_.append(future.result())
    
    clear_output()
    
    print("finished with multiprocessing, starting Einstein!")
    Einstein = []
    with cf.ThreadPoolExecutor() as exe:
        futures = [exe.submit(diffusion_constant_from_MSD,24., 28.,time,samples[i]) 
                   for i in range(sample_splits)]
    
    for future in cf.as_completed(futures):
        Einstein.append(future.result())
       
    clear_output()
    return list_, Einstein


def VarSetter(part_mass, dt, KB, T, Omega,seed):

    return [part_mass, dt, KB, T, Omega,seed]

def LoadDatafile(velocity_file,force_file,sqd_file):
    v = np.load(velocity_file)
    F = np.load(force_file)
    sq_displacement = np.load(sqd_file)
    
    return [v, F, sq_displacement]

def AutoCorrelationFunctions(v,F):
    # With multiprocessing
    start = tm.time()
    with cf.ProcessPoolExecutor() as exe:
        future_vacf = exe.submit(mz.compute_correlation_functions, v,None,True,False,False)
        future_Facf = exe.submit(mz.compute_correlation_functions, F,None,True,False,False)
        future_vFcross = exe.submit(mz.compute_correlation_functions, v,F,True,False,False)
    
        v_acf = future_vacf.result()
        F_acf = future_Facf.result()
        vF_cross = future_vFcross.result()
    
    print(f"multiprocessing took {(round((tm.time()-start)/60,2))} mins to finish correlation functions")    
    return [v_acf, F_acf, vF_cross]
    
def GenSamples(num,num_sims,seed):
    """
    Generates a 'num' amount of samples and returns the results as a list of all the samples.
    
    args:
    num = The resulting number of samples
    seed = The seed for randomizing
    """
    np.random.seed(seed)
    
    list_ = []
    for _ in range(num):
        list_.append((_+1,list(np.random.choice(range(num_sims), size = num_sims, 
                                      replace = True))))
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

def Fixup(holder,ein):    
    dict_ = defaultdict(list)
    for dic in holder:
        for key, value in dic.items():
            dict_[key].append(value)


    mz_dict = dict()
    for dic,dic2 in zip(dict_['MZ'],dict_['v_acf_mz']):
        for ((key,value),(key2,value2)) in zip(dic.items(),dic2.items()):
            mz_dict[key] = [value,value2]

    mz_cutoff = []
    mz_vacf = []
    for key,value in mz_dict.items():
        mz_cutoff.append(value[0])
        mz_vacf.append(value[1])


    ein_dict = defaultdict(list)
    for dic in ein:
        for key, value in dic.items():
            ein_dict[key].append(value)
    ein_dict = pd.DataFrame(dict(ein_dict))
    ein_list = list(ein_dict.sort_values(by='sample_ID')['Einstein'])


    dict_ = dict(dict_)
    dict_.pop('MZ')
    dict_.pop('v_acf_mz')

    dict_.update({'sample_ID': [value for array in dict_['sample_ID'] for value in array],
                  'Direct': [float(value) for _ in dict_['Direct'] for value in _], 
                  'Einstein': ein_list,       #[value for array in dict_['Einstein'] for value in array],
                  'Correlation_Func': [array for list_ in dict_['Correlation_Func'] for array in list_],
                  'K_Values': [value for inlist in dict_['K_Values'] for array in inlist for value in array],
                  'K1_Values': [array for list_ in dict_['K1_Values'] for array in list_],
                  'K3_Values': [array for list_ in dict_['K3_Values'] for array in list_],
                  'mz_cutoff': mz_cutoff,
                  'v_acf_mz': mz_vacf})
    
    dict_df = pd.DataFrame(dict_).sort_values(by='sample_ID')
    dict_df.reset_index(drop=True,inplace=True)
    return dict_df


def Bootstrap(samples,cutoffs,v_acf,F_acf,vF_cross,particle_mass,dt,kB,T,Omega,num_steps,time): # samples is a list
    """
    
    args:
    samples = sample list
    cutoffs = array of kernel cutoff times
    particle_mass = mass of the big particle (the small particles have unit mass)
    dt = delta time
    seed = seed for rng
    kB = constant Boltsman
    T = temperature
    Omega = Omega value
    """
    
    results = dict()
    start = tm.time()
    array_length = len(samples)
    sam_ID = []
    for id_,sample in samples:
        sam_ID.append(id_)
    
    results['sample_ID'] = np.zeros([array_length])
    results['Direct'] = np.zeros([array_length])
    results['Einstein'] = np.zeros([array_length])
    results['Correlation_Func'] = [[] for i in range(array_length)]
    results['K_Values'] = [[] for i in range(array_length)]
    results['K1_Values'] = [[] for i in range(array_length)]
    results['K3_Values'] = [[] for i in range(array_length)]
    
    
    results['MZ'] = dict()
    results['v_acf_mz'] = dict()
    
    for id_ in sam_ID:
        results['MZ'][id_] =  dict()
        results['v_acf_mz'][id_] = dict()

    for key,value in results['MZ'].items():  
        for cutoff in cutoffs:
            value[cutoff] = 0.

    for key,value in results['v_acf_mz'].items():  
        for cutoff in cutoffs:
            value[cutoff] = []
        
    for iboot in range(len(samples)):
        clear_output()
        print("This is boot number {}".format(iboot+1))
        
        results['sample_ID'][iboot] = sam_ID[iboot]
        sample = samples[iboot][1]
        
        v_acf_boot = np.mean(v_acf[:, sample, :], axis=(1,2))
        F_acf_boot = np.mean(F_acf[:, sample, :], axis=(1,2))
        vF_cross_boot = np.mean(vF_cross[:, sample, :], axis=(1,2))
        results['Correlation_Func'][iboot] = [v_acf_boot, F_acf_boot, vF_cross_boot]
        print("Finished the means")
        
        K1 = -F_acf_boot / (particle_mass * kB * T)
        K3 = vF_cross_boot / (kB * T)
        K = mz.get_K_from_K1_and_K3(dt, K1, K3)
        C0 = v_acf_boot[0]
        results['K_Values'][iboot] = [K]
        results['K1_Values'][iboot] = K1
        results['K3_Values'][iboot] = K3
        print("Finshed K stuff. Check point before mz")
        
        for cutoff in cutoffs:
            v_acf_cutoff = mz.integrate_C(dt, num_steps, Omega, K, C0 = C0, cutoff = cutoff)
            
            results['MZ'][sam_ID[iboot]][cutoff] = np.trapz(v_acf_cutoff, time)
            results['v_acf_mz'][sam_ID[iboot]][cutoff] = v_acf_cutoff
            
        print("Another check point after mz")
    
        results['Direct'][iboot] = np.trapz(v_acf_boot, time)
        print("finished an it")
        
    print(f"Boostrapping took {round((tm.time()-start)/60,2)} mins" 
          + f" over {len(samples)} bootstraps")
    
    return results



if __name__ == '__main__':
    start = tm.time()
    list_, Ein_list = main(16,1098340,1280) 
    result = Fixup(list_,Ein_list)
    print(' The bootstrapping finished in:\n',round(((tm.time()-start)/60 / 60),2) ,'hours')