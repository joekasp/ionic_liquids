import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


def MLP_Regressor(dataframe):
		alphas = np.array([5,2,5,1.5,1,0.1,0.01,0.001,0.0001,0])
		mlp_regr = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=5000, random_state=None,learning_rate_init=0.01)
		grid_search = GridSearchCV(mlp_regr, param_grid=dict(alpha=alphas))
		grid_search.fit(X_train,y_train)
		return grid_search
