import Lou_Bootstrap as lbs
import time as tm
import Lou_MoriZwanzig as lmz
import Lou_MZRefactored as mz
import numpy as np


dataSet = np.random.rand(25)
x = lbs.main(mz.DiffConstant_Directly,dataSet,16,1,4,None,None,)
