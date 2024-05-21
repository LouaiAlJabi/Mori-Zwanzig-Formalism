import numpy as np
import time as tm
from collections import defaultdict
import concurrent.futures as cf
from IPython.display import clear_output
import pandas as pd
import pickle
import os

dataSet = np.array([np.random.rand(25),np.random.rand(25)])
dataStuff = np.random.rand(25)
def UserFunction(dataSet):
    return np.mean(dataSet)

def GenSamples(num,num_sims,seed):
    """
    Generates a 'num' amount of samples and renum_simsturns the results as a list of all the samples.
    Args:
        num (_type_): The resulting number of samples
        num_sims (_type_): number of simulations, determined by the data provided
        seed (_type_): The seed for randomizing

    Returns:
        list: a list of lists, each containing a sample portion
    """    
    np.random.seed(seed)
    
    list_ = []
    for _ in range(num):
        list_.append(np.random.choice(range(num_sims), size = num_sims,replace = True))
        
    return list_

def EvenSplit(list_, chunk):
    """
    The Function takes in a list of even length and splits it into an even number of chunks.
    Args:
        list_ (list): The list to split
        chunk (_type_): The number of chunks to split into

    Raises:
        ValueError: "The list can't be split evenly into an even number. Check its length and the requested chunk size, both should be even."

    Returns:
        list: a list of chunks
    """   
    if len(list_) % 2 == 0 and chunk % 2 == 0:
        list_ = np.array_split(list_, chunk)
        return [i for i in list_]
    else:
        raise ValueError("The list can't be split evenly into an even number." 
                         + " Check its length and the requested chunk size, both should be even.")

def Bootstrap(userFunction,dataSet = np.array,num_boots = int,samplePortion = list,multipleDim = True, sampledAxis=None,*args):
    """
    This function takes in a Function and a dataset and does bootstrapping on it

    Args:
        userFunction (Function): a function that takes in an iterable data structure among its other args
        dataSet (numpy array): an iterable data structure. Defaults to np.array.
        num_boots (int): number of bootstraps. Defaults to int.
        samplePortion (list): a list representing a single chunk of the generated samples. Defaults to list.
        multipleDim (bool, optional): A bool indicating weather the data structure is of a multiple dimensions. Defaults to True.
        sampledAxis (int, optional): The axis the user desires to sample accros. Defaults to None (0).

    Returns:
        Dictionary: contains the results of bootstrap(s)
    """
    boots = len(samplePortion)
    if (not multipleDim):
        sampledAxis = 0

    bootResults = {}
    if (multipleDim):
        for iboot in range(boots):
            bootResults["Results " + str(iboot + 1) + ":"] = []
            bootResults[list(bootResults.keys())[iboot]] = np.apply_along_axis(userFunction,sampledAxis,
                                                                               np.take(dataSet,samplePortion[iboot],axis=sampledAxis),*args)
    else:
        bootResults["Results:"] = np.zeros(boots)
        for iboot in range(boots):
            bootResults["Results:"][iboot] = userFunction(np.take(dataSet,samplePortion[iboot]),*args)

    #the user gives a function with dataset (iterable)
    #the user gives the number of bootstraps they want
    #the user gives a desired axis
    #the user gives the desired confidence interval
    
    #the function takes in a sample portion 
    #the function creates a temp holder of boot
    #the function runs the function with the dataset once
    #the function bootstraps along the desired axis
    #the function returns the output as a list of three elements: function result on whole data, lower and upper confidence bound
    return bootResults
    

def main(userFunction,dataSet = np.array,boots = int,seed = int,cores = int,multipleDim = True, sampledAxis=None,confInt=None,*args):
    """
    The function uses multiprocessing to bootstrap.

    Args:
        userFunction (Function): a function that takes in an iterable data structure among its other args
        dataSet (numpy array): an iterable data structure. Defaults to np.array.
        boots (int): number of bootstraps. Defaults to int.
        seed (int, optional): The seed to randomize with. Defaults to int.
        cores (int, optional): The number of cores the multiprocessing will use. Defaults to int.
        multipleDim (bool, optional): A bool indicating weather the data structure is of a multiple dimensions. Defaults to True.
        sampledAxis (int, optional): The axis the user desires to sample accros. Defaults to None (0).
        confInt (float, optional): The confidence interval. Defaults to 0.95.

    Returns:
        list: A list containing the function result on whole data, lower and upper confidence bound
    """
    np.random.seed(seed)
    finalResults = [[],[]]
    
    if (multipleDim):
        if (sampledAxis):
            num_sims = dataSet.shape[sampledAxis]
        else:
            num_sims = dataSet.shape[0]
            sampledAxis = 0
    else:
        num_sims = dataSet.shape[0]
        sampledAxis = 0
     
    if (not confInt):
        confInt = 0.95
        
    confIdx = int((1 - confInt)/2 * boots) - 1
    samples = GenSamples(boots,num_sims,seed)
    splits = list(EvenSplit(samples, cores))
    finalResults.insert(0,["Results on everything:",np.apply_along_axis(userFunction,sampledAxis,dataSet,*args)])
    
    list_ = []
    with cf.ProcessPoolExecutor(cores) as exe:
        futures = [exe.submit(Bootstrap,userFunction,dataSet,boots,splits[i],multipleDim, sampledAxis,*args) 
                   for i in range(len(splits))]
        for future in cf.as_completed(futures):
            list_.append(future.result())

    if (multipleDim):
        multResults = []
        for i in list_:
            for j in range(len(i)):
                multResults = multResults + list(i["Results " + str(j + 1) + ":"]) 

        finalResults[1] = sorted(multResults)[confIdx]
        finalResults[2] = sorted(multResults)[-(confIdx+1)]
    else:
        singleResults = []
        for i in list_:
            for j in range(len(i)):
                singleResults = singleResults + list(i["Results:"])
        finalResults[1] = sorted(singleResults)[confIdx]
        finalResults[2] = sorted(singleResults)[-(confIdx+1)]
    return finalResults


x = main(UserFunction,dataSet,4000,1,4,True,None,0.95)
print(x)