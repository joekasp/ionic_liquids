import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils 
from methods import lasso
from visualization import core

filename = 'datasets/inputdata.xlsx'

#get X matrix and response vector y (need a function for this) 
df = utils.read_data(filename) 
X,y = utils.molecular_descriptors(df)

#do machine_learning call
#MLP_Regr = MLP_regressor(molecular_descriptors, conductivity)
#MLP_class = MLP_classifier(molecular_descriptors, conductivity)
obj = lasso.do_lasso(X,y)
#SVR = SVR(molecular_descriptors, conductivity)

#save model to file
#utils.save_model(obj,model_type='lasso')

#plot
my_plot = core.parity_plot(y,obj.predict(X))
plt.show(my_plot)

