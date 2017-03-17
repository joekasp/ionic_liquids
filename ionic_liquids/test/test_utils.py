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
from utils import *
from numpy.random import normal

def test_errors():
    """
    Test calling errors, 
    
    Input
    ------
    y_test: Y for test data, numpy array
    y_prediction: Prediction for X_train, numpy array
    
    Returns
    ------
    difference: difference between two array, type is numpy array

    Checking:
    1. The difference is calculated right
    """
    A = np.array((1,1,1))
    B = np.array((2,2,2))
    assert np.array_equal(errors(A,B),(B-A)), "Error is calculated wrong" 

def test_train_model():  
    """
    Test calling train_model,
    
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
    
    Checking:
    1. X_mean is calculated
    2. X_std is calculated
    """
    obj, X, Y, X_mean, X_std = train_model('lasso','datasets/compounddata.xlsx',0.1)
    assert X_mean is not None, "X mean is not calculated."
    assert X_std is not None, "X standard deviation is not calculated."

def test_normalization():
    """
    Test calling normalization,

    Inputs
    ------
    data : Pandas DataFrame
    means : optional numpy argument of column means
    stdevs : optional numpy argument of column st. devs

    Returns
    ------
    normed : the normalized DataFrame
    means : the numpy row vector of column means
    stdevs : the numpy row vector of column st. devs
    
    Checking:
    1. Normalization is calculated right
    """
    A = pd.DataFrame(normal(10,2,25).reshape(5,5))
    norm, mean, std = normalization(A)
    B = StandardScaler(A).copy
    assert pd.DataFrame.equals(norm,B), "Normalization is not right"

def test_predict_model():
    """
    Test calling predict_model,

    Inputs
    -----
    A_smile : SMILES string for compound A
    B_smile : SMILES string for compound B
    obj : model object
    t : float of temperature
    p : float of pressure
    m : float of mol_fraction
    X_mean : means of columns for normalization
    X_stdev : stdevs fo columns for normalization
    flag : string to designate which variable is on x-axis

    Returns
    ------
    x_vals : x-values chosen by flag
    y_pred : predicted conductivity (y_values)
    
    Checking:
    1. Shape of x_vals is right
    2. Shape of y_pred is right
    """
    obj, X, Y, X_mean, X_std = train_model('lasso','datasets/compounddata.xlsx',0.1)
    X, Y = predict_model('[B-](F)(F)(F)F.CCCCCCn1cc[n+](c1)C','CO',obj,328.15,101,0.0065,X_mean,X_std,flag='m')
    assert X.shape[0] == 101, "Prediction is wrong!"
    assert Y.shape[0] == 101, "Prediction is wrong!"

def test_molecular_descriptors():
    """
    Test calling molecular_descriptors,
    
    Inputs
    ------
    data: dataframe, cleaned csv data
    
    Returns
    ------
    prenorm_X: normalized input features
    Y: experimental electrical conductivity
    
    Checking:
    1. Shape of prenorm_X is right
    """
    
    df = read_data("datasets/compounddata.xlsx")
    X, Y = molecular_descriptors(df)
    assert X.shape[0] == 2523, "Data shape is not right"

def test_read_data():
    """
    Test calling read_data

    Inputs
    -------
    filename : string of path to file

    Returns
    ------ 
    df : Pandas DataFrame
    
    Checking:
    1. Shape of df is right

    """
    df = read_data("datasets/compounddata.xlsx")
    assert df.shape[0] == 2523, "Data shape is not right"

def test_save_model():
    """
    Test calling save_model
    
    Input
    ------
    obj: model object
    X_mean : mean for each column of training X
    X_stdev : stdev for each column of training X
    X : Predictor matrix
    y : Response vector
    dirname : the directory to save contents  
 
    Returns
    ------
    None
    
    Checking:
    1. model is saved as a directory
    """
    obj, X_mean, X_stdev, X, y = read_model('test_model/')
    save_model(obj, X_mean, X_stdev, X, y,'saved_model')
    obj, X_mean, X_stdev, X, y = read_model('saved_model/')
    assert X_mean is not None, "The model is not saved"

def test_read_model():
    """
    Test calling read_model
    
    Input
    ------
    file_dir : the directory containing all model info
    
    Returns
    ------
    obj: model object
    X_mean : mean of columns in training X
    X_stdev : stdev of columns in training X
    X : predictor matrix (if it exists) otherwise None
    y : response vector (if it exists) otherwise None
    
    Checking:
    1. X_mean is read from model
    2. X_stdev is read from model
    
    """
    obj, X_mean, X_stdev, X, y = read_model('test_model/')
    assert X_mean is not None, "X mean from this model is null"
    assert X_stdev is not None, "X standard deviation from this model is null"
    
