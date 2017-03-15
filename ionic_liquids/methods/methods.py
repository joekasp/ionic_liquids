#import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

def do_svr(X,y):
    """
    Call the Support Vector Regressor, 
    Fit the weight on the training set
    
    Input
    ------
    X: dataframe, n*m, n is number of data points, 
        m is number of features
    y: experimental electrical conductivity
    
    Returns
    ------
    svr: objective, the regressor objective 
    """
    
    svr = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001,
        C=1.0, epsilon=0.01, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    svr = GridSearchCV(svr, cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})
    svr.fit(X,y)
    return svr


def do_MLP_regressor(X,y):
    """
    Call the MLP Regressor, 
    Fit the weight on the training set
    
    Input
    ------
    X: dataframe, n*m, n is number of data points, 
        m is number of features
    y: experimental electrical conductivity
    
    Returns
    ------
    mlp_regr : the MLP object with the best parameters
    """    
    alphas = np.array([5,2,5,1.5,1,0.1,0.01,0.001,0.0001,0])
    mlp_regr = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
        solver='adam', alpha=0.0001, max_iter=5000, random_state=None,learning_rate_init=0.01)
    grid_search = GridSearchCV(mlp_regr, param_grid=dict(alpha=alphas))
    grid_search.fit(X,y)
    
    #print(grid_search.best_params_)
    mlp_regr.alpha_ = grid_search.best_params_['alpha']
    mlp_regr.fit(X,y)

    return mlp_regr

def do_MLP_classifier(X,y):
    """
    Call the MLP Classifier, 
    Fit the weight on the training set
    
    Input
    ------
    X: dataframe, n*m, n is number of data points, 
        m is number of features
    y: experimental electrical conductivity
    
    Returns
    ------
    mlp_class: sklearn object with model information 

    """
    
    alphas = np.array([5,2,5,1.5,1,0.1,0.01,0.001,0.0001,0])
    mlp_class = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', 
        solver='adam', alpha=0.0001, max_iter=5000, random_state=None,learning_rate_init=0.01)
    grid_search = GridSearchCV(mlp_class, param_grid=dict(alpha=alphas))
    grid_search.fit(X,y)

    mlp_class.alpha_ = grid_search.best_params_['alpha']    
    mlp_class.fit(X,y)

    return mlp_class


def do_lasso(X,y): 
    """
    Runs a lasso grid search on the input data
    
    Inputs
    ------
    X: dataframe, n*m, n is number of data points, 
        m is number of features
    y: experimental electrical conductivity
    
    Returns
    ------
    lasso : sklearn object with the model information 
    """
    
    alphas = np.array([5,4,3,2,1,0.1,0.01,0.001,0.0001])
    lasso = Lasso(alpha=0.001, fit_intercept=True, normalize=False, precompute=False,
        copy_X=True, max_iter=10000, tol=0.001, positive=False, random_state=None, selection='cyclic')
    gs = GridSearchCV(lasso, param_grid=dict(alpha=alphas))
    gs.fit(X,y)
    
    lasso.alpha_ = gs.best_params_['alpha']

    lasso.fit(X,y)
        
    return lasso

