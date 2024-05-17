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

def Bootstrap(dataSet,function,num_boots,sampledAxis=None):
    baseResult = function(dataSet)
    finalResult = tuple()
    sample = np.random.choice(range(baseResult))

    
    #for iboot in num_boots:
        
    
    

baseResult = UserFunction(dataSet)
baseResult.shape

#Bootstrap(dataSet,UserFunction)
list_ = []
for _ in range(16):
    list_.append((_+1,list(np.random.choice(range(99), size = 99, 
                                    replace = True))))
print(type(list_[0][1]))