import numpy as np
import pandas as pd

import utils

from front_functions import * 
from utils import * 
import methods

filename = 'datasets/inputdata.xlsx'

def read_model(filename,model_type):
    if (model_type == 'lasso'):
        obj = Lasso
    elif (model_type == 'mlp_reg'):
        obj = MLPRegressor
    elif (model_type == 'mlp_clas'):
        obj = MLPClassifier
    elif (model_type == 'svr'):
        obj = SVR
    else:
        raise ValueError('Invalid model type!')
    return obj

#get X matrix and response vector y (need a function for this) 
df = read_data(filename) 
X,y = molecular_descriptors(df)

#do machine_learning call
#MLP_Regr = MLP_regressor(molecular_descriptors, conductivity)
#MLP_class = MLP_classifier(molecular_descriptors, conductivity)
Lasso = Lasso(molecular_descriptors, conductivity)
#SVR = SVR(molecular_descriptors, conductivity)

#save model to file
utils.save_model(obj,model_type='lasso')

#plot
plot = error_plots()


