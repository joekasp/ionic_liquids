import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

def lasso(dataframe): 
		alphas = np.array([5,4,3,2,1,0.1,0.01,0.001,0.0001])
		lasso = Lasso(alpha=0.001, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=10000, tol=0.001, positive=False, random_state=None, selection='cyclic')
		grid_search = GridSearchCV(lasso, param_grid=dict(alpha=alphas))
		grid_search.fit(X_train,y_train)
		return grid_search
