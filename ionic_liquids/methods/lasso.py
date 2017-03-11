import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

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

