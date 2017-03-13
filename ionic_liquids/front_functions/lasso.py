#import packages, line with ## can be deleted, need final decision
import pandas as pd
##import matplotlib.pyplot as plt
import numpy as np
##from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV

def lasso(dataframe): 
	'''Calling the LASSO regressor, 
	passing the training data into the regressor 
	'''
	alphas = np.array([5,4,3,2,1,0.1,0.01,0.001,0.0001])
	lasso = Lasso(alpha=0.001, fit_intercept=True, normalize=False,
			precompute=False, copy_X=True, max_iter=10000, 
			tol=0.001, positive=False, random_state=None, 
			selection='cyclic')
	gs = GridSearchCV(lasso, param_grid=dict(alpha=alphas))
	gs.fit(X_train,y_train) # Fit the LASSO weight

	return gs
