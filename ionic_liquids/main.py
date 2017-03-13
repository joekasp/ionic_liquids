import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils 
from methods import methods
from visualization import plots

filename = 'datasets/inputdata.xlsx'

#get X matrix and response vector y (need a function for this) 
df = utils.read_data(filename) 
train, test = train_test_split(data_scaled, Y, test_size=0.1)
X_train,y_train = utils.molecular_descriptors(train)
X_test,y_test = utils.molecular_descriptors(test)

#do machine_learning call
#MLP_Regr = MLP_regressor(molecular_descriptors, conductivity)
#MLP_class = MLP_classifier(molecular_descriptors, conductivity)
obj = methods.do_lasso(X,y)
#SVR = SVR(molecular_descriptors, conductivity)

#save model to file
#utils.save_model(obj,model_type='lasso')

#plot
my_plot = plots.parity_plot(y,obj.predict(X))
plt.show(my_plot)

