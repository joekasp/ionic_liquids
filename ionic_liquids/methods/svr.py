import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def do_svr(X,y):
    svr = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001,
        C=1.0, epsilon=0.01, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    svr = GridSearchCV(svr, cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})
    svr.fit(X,y)

	return svr

