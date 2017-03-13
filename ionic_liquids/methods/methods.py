import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def do_svr(X,y):
    svr = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001,
        C=1.0, epsilon=0.01, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    svr = GridSearchCV(svr, cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})
    svr.fit(X,y)

	return svr

def do_MLP_regressor(X,y):
    #MLPRegressor
    alphas = np.array([5,2,5,1.5,1,0.1,0.01,0.001,0.0001,0])
    mlp_regr = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
        solver='adam', alpha=0.0001, max_iter=5000, random_state=None,learning_rate_init=0.01)
    grid_search = GridSearchCV(mlp_regr, param_grid=dict(alpha=alphas))
    grid_search.fit(X,y)
    return grid_search


def do_MLP_classifier(X,y):
    #MLPClassifier
    alphas = np.array([5,2,5,1.5,1,0.1,0.01,0.001,0.0001,0])
    mlp_class = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', 
        solver='adam', alpha=0.0001, max_iter=5000, random_state=None,learning_rate_init=0.01)
    grid_search = GridSearchCV(mlp_class, param_grid=dict(alpha=alphas))
    grid_search.fit(X_train,y_train)
    return grid_search


def do_lasso(X,y): 
    """
    Runs a lasso grid search on the input data

    Inputs
    ------
    X : Matrix of predictors
    y : feature of interest

    Returns
    ------
    lasso : sklearn object with the model information

    """

    alphas = np.array([5,4,3,2,1,0.1,0.01,0.001,0.0001])
    lasso = Lasso(alpha=0.001, fit_intercept=True, normalize=False, precompute=False,
        copy_X=True, max_iter=10000, tol=0.001, positive=False, random_state=None, selection='cyclic')
    #gs = GridSearchCV(lasso, param_grid=dict(alpha=alphas))
    #gs.fit(X,y)
    lasso.fit(X,y)
	
    return lasso

