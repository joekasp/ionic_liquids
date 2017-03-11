import numpy as np
import pandas as pd
<<<<<<< HEAD
import utils

from front_functions import * 

=======
from utils import * 
import methods

filename = 'datasets/inputdata.xlsx'
>>>>>>> fa6a18065105f458292d126a554308418dff6126

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


