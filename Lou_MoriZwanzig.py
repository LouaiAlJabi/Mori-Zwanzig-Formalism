import mori_zwanzig as mz
from sklearn.linear_model import LinearRegression
import numpy as np
import time as tm
from collections import defaultdict
import concurrent.futures as cf
from IPython.display import clear_output
import pandas as pd
import pickle


def main(num_boots,cores,seed,part_mass, dt, KB, T, Omega,velocity,force,sq_displacement):
    """
    A function that does the bootstrapping in parallel over 16 cores.
    
    args:
    num_boots = number of bootstraps to do, multiplication of 16. also acts as the number of sample splits
    seed = the randomizing seed to get samples, use for deterministic values
    particle_mass = mass of the big particle (the small particles have unit mass)
    dt = delta time
    seed = seed for rng
    kB = constant Boltsman
    T = temperature
    Omega = Omega value
    velocity = the file of velocities
    Froce = the file of forces
    sq_displacement = the file of squared displacement
    """

    #Use the functions to set up the necessery data
    print("doing autocorrelation")
    v_acf, F_acf, vF_cross = AutoCorrelationFunctions(velocity,force)
    num_steps, num_sims, num_dimensions = velocity.shape
    steps_to_integrate = num_steps
    time = np.linspace(0, dt*num_steps, num_steps)
    #include your own range of cuttoffs
    cutoffs = np.arange(1,29.)

    print("Generating samples")
    #generate and split the sample set
    samples = GenSamples(num_boots,num_sims,seed)
    splits = list(EvenSplit(samples, cores))
    
    print("doing multiprocessing")
    list_ = []
    #With Concurrent futures, split the bootstrapping work to be distributed on 16 cores at a time (e.g. 2 jobs for a 32core computer), adjust accordinge to your device.
    #Recommended not to use all your cores as that might cause the computer to crash
    with cf.ProcessPoolExecutor(cores) as exe:
        futures = [exe.submit(Bootstrap, splits[i],cutoffs,v_acf,F_acf,vF_cross,part_mass,dt,KB,T,Omega,num_steps,time) for i in range(len(splits))]
        for future in cf.as_completed(futures):
            list_.append(future.result())
    
    clear_output()
    
    print("finished with multiprocessing MZ, starting Einstein!")
    Einstein = []
    with cf.ThreadPoolExecutor() as exe:
        futures = [exe.submit(diffusion_constant_from_MSD,24., 28.,time,sq_displacement,samples[i]) 
                   for i in range(num_boots)]
    
    for future in cf.as_completed(futures):
        Einstein.append(future.result())
        
    clear_output()
    return list_, Einstein



def VarSetter(part_mass, dt, KB, T, Omega):
    if part_mass < 0:
        raise ValueError("Mass can't be negative")
    return [part_mass, dt, KB, T, Omega]

def LoadDatafile(velocity_file,force_file,sqd_file):
    v = np.load(velocity_file)
    F = np.load(force_file)
    sq_displacement = np.load(sqd_file)
    
    return [v, F, sq_displacement]

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
    
def GenSamples(num,num_sims,seed):
    """
    Generates a 'num' amount of samples and returns the results as a list of all the samples.
    
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
    """
    This functions takes in the two resulted dictionaries from the "main" function and fixes them to be presented in a single proper dictionary.
    Very specific for this data.

    args:
    holder = the first resulted dictionary from "main" which houses all the results besides the Einstein method
    ein = the second resulted dictionary from "main" that houses only the Einstein method
    """    
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


def Bootstrap(samples,cutoffs,v_acf,F_acf,vF_cross,particle_mass,dt,kB,T,Omega,num_steps,time): 
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

def PickleResults(name,result):
    with open(name,'wb') as file:
        pickle.dump(result,file,protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    start = tm.time()
    print("started")
    velocity, force, sq_displacement = LoadDatafile("velocities.npy", "forces.npy", "sq_displacement_stat.npy")
    part_mass, dt,KB,T,Omega = VarSetter(80.,0.005,1.,1.,0)
    print("mp started")
    list_, Ein_list = main(16,4,1,part_mass,dt,KB,T,Omega,velocity,force,sq_displacement) 
    result = Fixup(list_,Ein_list)
    PickleResults("NewResultTest",result)
    print(' The bootstrapping finished in:\n',round(((tm.time()-start)/60 / 60),2) ,'hours')