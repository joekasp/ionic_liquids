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

def read_model(filename,model_type):
    if (model_type == 'lasso'):
        obj = lasso.do_lasso(X,y)
    elif (model_type == 'mlp_reg'):
        obj = MLP_regressor.do_MLP_regressor(molecular_descriptors, conductivity)
    elif (model_type == 'mlp_clas'):
        obj = mlp_classifier.do_MLP_classifier(molecular_descriptors, conductivity)
    elif (model_type == 'svr'):
        obj = svr.do_SVR(molecular_descriptors, conductivity)
    else:
        raise ValueError('Invalid model type!')
    return obj
