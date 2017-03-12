import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import * 
from methods import lasso, mlp_classifier, mlp_regressor, svr, errors

filename = inputdata.xlsx

#Read in the filename 'inputdata.xlsx' 
data = read_data(filename)
molecular_descriptors, conductivity = molecular_descriptors(data)

#Read in model
obj = read_model(filename,model_type)

#Save model to file
model = save_model(obj,model_type)

#Plot the electronic conductivities and predictions
error_value = []
#X_train, X_test, y_train, y_test = train_test_split(data_scaled, Y, test_size=0.1)
#error_value.append(errors(molecular_descriptors, y_test)
#plot_errors = make_plot(obj, error_value)
#plot_predictions = make_plot(obj,y_test)
#show(plot_errors,plot_predictions))

