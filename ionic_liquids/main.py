import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import utils
from methods import methods
from visualization import plots


FILENAME = 'datasets/inputdata.xlsx'
MODEL = 'mlp_regressor'
DIRNAME = 'my_test'

# get X matrix and response vector y (need a function for this)
df = utils.read_data(FILENAME)
X, y = utils.molecular_descriptors(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# normalize testing data using training data
X_train, X_mean, X_std = utils.normalization(X_train)
X_test, trash, trash = utils.normalization(X_test, X_mean, X_std)

# do machine_learning call
if (MODEL.lower() == 'mlp_regressor'):
    obj = methods.do_MLP_regressor(X_train, y_train)
elif (MODEL.lower() == 'mlp_classifier'):
    obj = methods.do_MLP_classifier(X_train, y_train)
elif (MODEL.lower() == 'lasso'):
    obj = methods.do_lasso(X_train, y_train)
elif (MODEL.lower() == 'svr'):
    obj = methods.do_svr(X_train, y_train)
else:
    raise ValueError("Model not supported")

# save model to file
utils.save_model(obj, X_mean, X_std, X_train, y_train, dirname=DIRNAME)

# plot
my_plot = plots.parity_plot(y_train, obj.predict(X_train))
plt.show(my_plot)
