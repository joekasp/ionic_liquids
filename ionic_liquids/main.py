import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import utils 
from methods import methods
from visualization import plots


FILENAME = 'datasets/inputdata.xlsx'
MODEL = 'lasso'

#get X matrix and response vector y  

df = utils.read_data(FILENAME) 
train, test = train_test_split(data_scaled, Y, test_size=0.1)
X_train,y_train,X_test,y_test = utils.molecular_descriptors(df)

#do machine_learning call

if (MODEL.lower() == 'mlp_regressor'): 
    obj = methods.do_MLP_regressor(X_train, y_train)
elif (MODEL.lower() == 'mlp_classifier'): 
    obj = methods.do_MLP_classifier(X_train, y_train)
elif (MODEL.lower() == 'lasso'): 
    obj = methods.do_lasso(X_train,y_train)
elif (MODEL.lower() == 'svr'): 
    obj = methods.do_svr(X_train,y_train)
else:
    raise ValueError("Model not supported")

#save model to file

utils.save_model(obj,model_type=MODEL)

#plot

my_plot = plots.parity_plot(y,obj.predict(X))
plt.show(my_plot)

