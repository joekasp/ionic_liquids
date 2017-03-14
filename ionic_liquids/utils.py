#default python modules
import os
from datetime import datetime
#external packages
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as Calculator
#internal modules
from methods import methods


def errors(y_test,y_prediction):
    '''Generate the prediction error'''
    difference = y_prediction - y_test
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
    
    model = model.replace(' ','_')
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


def save_model(obj,X=None,y=None,dirname='default'):
    """
    Save the trained regressor model to the file
    
    Input
    ------
    obj: model object
    X : Predictor matrix
    y : Response vector
    dirname : the directory to save contents  
 
    Returns
    ------
    None
    """
    if (dirname == 'default'):
        timestamp = str(datetime.now())[:19]
        dirname = 'model_'+timestamp.replace(' ','_')
    else:
        pass
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
    filename = dirname + '/model.pkl'   
    joblib.dump(obj,filename)  

    if (X is not None):
        filename = dirname + '/X_data.pkl'
        joblib.dump(X,filename)
    else:
        pass

    if (y is not None):
        filename = dirname + '/y_data.pkl'
        joblib.dump(y,filename)
    else:
        pass

    return


def read_model(file_dir):
    """
    Read the trained regressor to 
    avoid repeating training.
    
    Input
    ------
    file_dir : the directory containing all model info
    
    Returns
    ------
    obj: model object
    X : predictor matrix (if it exists) otherwise None
    y : response vector (if it exists) otherwise None
    
    """
    filename = file_dir + '/model.pkl'
    obj = joblib.load(filename) 

    try:
        X = joblib.load(file_dir + '/X_data.pkl')
    except:
        X = None
    try:
        y = joblib.load(file_dir + '/y_data.pkl')
    except:
        y = None
    
    return obj, X, y

