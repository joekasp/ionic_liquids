import numpy as np
import pandas as pd
from front_functions import * 
import utils


#get X matrix and response vector y (need a function for this) 
data_frame = data_generate(filename) 

#calculate molecular descriptors
molecular_descriptors, conductivity = molecular_descriptors(data_frame)

#do machine_learning call
MLP_Regr = MLP_regressor(molecular_descriptors, conductivity)
MLP_class = MLP_classifier(molecular_descriptors, conductivity)
Lasso = Lasso(molecular_descriptors, conductivity)
SVR = SVR(molecular_descriptors, conductivity)

#save model to file
utils.save_model(obj,model_type='mlp_reg')


