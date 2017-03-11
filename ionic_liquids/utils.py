import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV

def save_model(obj,model_type,filename='default'):
    items = []
    if (model_type == 'lasso'):
        pass
    elif (model_type == 'mlp_reg'):
        pass
    elif (model_type == 'mlp_clas'):
        pass
    elif (model_type == 'svr'):
        pass
    else:
        Raise ValueError('Invalid model type!')

    if (filename == 'default'):
        filename = 'model' + model_type + '.txt'

    f = open(filename,'w')
    f.write(model_type + '\n')
    for item in items:
        f.write(item)
        f.write('\n')
    f.close()
 
    return


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
        Raise ValueError('Invalid model type!')    
    return obj

