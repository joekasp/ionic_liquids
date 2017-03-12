import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as Calculator

def save_model(obj,model_type,filename='default'):
    items = []
    if (model_type == 'lasso'):
        items.append(obj.coef_)
        items.append(obj.sparse_coef_)
        items.append(obj.intercept_)
        items.append(obj.n_iter_)
    elif (model_type == 'mlp_reg'):
        items.append(obj.loss_)
        items.append(obj.coefs_)
        items.append(obj.intercepts_)
        items.append(obj.n_iter_)
        items.append(obj.n_layers_)
        items.append(obj.n_outputs_)
        items.append(obj.out_activation_)
    elif (model_type == 'mlp_clas'):
        items.append(obj.classes_)
        items.append(obj.loss_)
        items.append(obj.coefs_)
        items.append(obj.intercepts_)
        items.append(obj.n_iter_)
        items.append(obj.n_layers_)
        items.append(obj.n_outputs_)
        items.append(obj.out_activation_)
    elif (model_type == 'svr'):
        items.append(obj.support_)
        items.append(obj.support_vectors_)
        items.append(obj.dual_coef_)
        items.append(obj.coef_)
        items.append(obj.intercept_)
        items.append(obj.sample_weight_)
    else:
        raise ValueError('Invalid model type!')

    if (filename == 'default'):
        filename = 'model_' + model_type + '.txt'

    f = open(filename,'w')
    f.write(model_type + '\n')
    for item in items:
        f.write(str(item))
        f.write('\n')
    f.close()
 
    return
