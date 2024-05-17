import numpy as np
import time as tm
from collections import defaultdict
import concurrent.futures as cf
from IPython.display import clear_output
import pandas as pd
import pickle
import os

dataSet = np.arange(1,100,1,dtype=object)

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

def UserFunction(dataSet):
    return np.square(dataSet)

def Bootstrap(dataSet,function,num_boots,sample):
    baseResult = function(dataSet)
    finalResult = tuple()
    
    #for iboot in num_boots:
        
    
    

baseResult = UserFunction(dataSet)
baseResult.shape

#Bootstrap(dataSet,UserFunction)
print(GenSamples(16,99,1).shape)