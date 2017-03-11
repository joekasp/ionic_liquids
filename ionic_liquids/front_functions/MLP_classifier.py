import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def MLP_Classifier(dataframe):
		alphas = np.array([1,0.1,0.01,0.001,0.0001])
		mlp_class = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=5000, random_state=None,learning_rate_init=0.01)
		grid_search = GridSearchCV(mlp_class, param_grid=dict(alpha=alphas))
		grid_search.fit(X_train,y_train)
		return grid_search

