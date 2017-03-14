#import packages
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
from methods import methods


def errors(y_test,y_prediction):
    '''Generate the prediction error'''
    difference = y_prediction - y_test
    j
    return difference


def train_model(model,data_file,test_percent,save=True):
    """
    Choose the regression model
    
    Input
    ------
    model: string, the model to use
    data_file: dataframe, cleaned csv data
    test_percent: float, the percentage of data held for testing
    
    Returns
    ------
    obj: objective, the regressor
    X: dataframe, normlized input feature
    y: targeted electrical conductivity
    
    """
    
    df = read_data(data_file)
    X,y = molecular_descriptors(df)
    X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=(test_percent/100))
    
    #print("training model is ",model)
    if (model.lower() == 'lasso'):
        obj = methods.do_lasso(X_train,y_train)
    elif (model.lower() == 'mlp_regressor'):
        obj = methods.do_MLP_regressor(X_train,y_train)
    elif (model.lower() == 'mlp_classifier'):
        obj = methods.do_MLP_classifier(X_train,y_train)
    elif (model.lower() == 'svr'):
        obj = methods.do_svr(X_train,y_train)
    else:
        raise ValueError('Invalid model type!') 
    
    return obj, X, y


def molecular_descriptors(data):
    """
    Use RDKit to prepare the molecular descriptor
    
    Inputs
    ------
    data: dataframe, cleaned csv data
    
    Returns
    ------
    norm_X: normalized input features
    Y: experimental electrical conductivity
    
    """
    
    n = data.shape[0]
    # Choose which molecular descriptor we want
    list_of_descriptors = ['NumHeteroatoms', 'ExactMolWt',
        'NOCount', 'NumHDonors',
        'RingCount', 'NumAromaticRings', 
        'NumSaturatedRings','NumAliphaticRings']
    # Get the molecular descriptors and their dimension
    calc = Calculator(list_of_descriptors)
    D = len(list_of_descriptors)
    d = len(list_of_descriptors)*2 + 4
    
    Y = data['EC_value']
    X = np.zeros((n,d))
    X[:,-3] = data['T']
    X[:,-2] = data['P']
    X[:,-1] = data['MOLFRC_A']
    
    for i in range(n):
        A = Chem.MolFromSmiles(data['A'][i])
        B = Chem.MolFromSmiles(data['B'][i])
        X[i][:D]    = calc.CalcDescriptors(A)
        X[i][D:2*D] = calc.CalcDescriptors(B)

    prenorm_X = pd.DataFrame(X,columns=['NUM', 'NumHeteroatoms_A', 
        'MolWt_A', 'NOCount_A','NumHDonors_A', 
        'RingCount_A', 'NumAromaticRings_A', 
        'NumSaturatedRings_A',
        'NumAliphaticRings_A', 
        'NumHeteroatoms_B', 'MolWt_B', 
        'NOCount_B', 'NumHDonors_B',
        'RingCount_B', 'NumAromaticRings_B', 
        'NumSaturatedRings_B', 
        'NumAliphaticRings_B',
        'T', 'P', 'MOLFRC_A'])
    # Normalize all the input feature
    norm_X = StandardScaler().fit_transform(prenorm_X)

    return norm_X, Y


def read_data(filename):
    """
    Reads data in from given file to Pandas DataFrame

    Inputs
    -------
    filename : string of path to file

    Returns
    ------ 
    df : Pandas DataFrame

    """
    cols = filename.split('.')
    name = cols[0]
    filetype = cols[1]
    if (filetype == 'csv'):
        df = pd.read_csv(filename)
    elif (filetype in ['xls','xlsx']):
        df = pd.read_excel(filename)
    else:
        raise ValueError('Filetype not supported')

    #clean the data if necessary
    df['EC_value'], df['EC_error'] = zip(*df['ELE_COD'].map(lambda x: x.split('±')))

    return df


def save_model(obj,model_type,filename='default'):
    """
    Save the trained regressor model to the file
    
    Input
    ------
    obj: objective, the regressor objective
    model_type: string, the name of the model
    
    Returns
    ------
    None
    """
    
    items = []
    if (model_type.lower() == 'lasso'):
        items.append(obj.coef_)
        items.append(obj.sparse_coef_)
        items.append(obj.intercept_)
        items.append(obj.n_iter_)
    elif (model_type.lower() == 'mlp_regressor'):
        items.append(obj.loss_)
        items.append(obj.coefs_)
        items.append(obj.intercepts_)
        items.append(obj.n_iter_)
        items.append(obj.n_layers_)
        items.append(obj.n_outputs_)
        items.append(obj.out_activation_)
    elif (model_type.lower() == 'mlp_classifier'):
        items.append(obj.classes_)
        items.append(obj.loss_)
        items.append(obj.coefs_)
        items.append(obj.intercepts_)
        items.append(obj.n_iter_)
        items.append(obj.n_layers_)
        items.append(obj.n_outputs_)
        items.append(obj.out_activation_)
    elif (model_typel.lower() == 'svr'):
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


def read_model(filename,model_type):
    """
    Read the trained regressor to 
    avoid repeating training.
    
    Input
    ------
    model_type: string, the name of the regression model
    
    Returns
    ------
    obj: objective, the regressor objective
    
    """
    if (model_type == 'lasso'):
        obj = Lasso
    elif (model_type == 'mlp_reg'):
        obj = MLPRegressor
    elif (model_type == 'mlp_clas'):
        obj = MLPClassifier
    elif (model_type == 'svr'):
        obj = SVR
    else:
        raise ValueError('Invalid model type!')
    return obj

