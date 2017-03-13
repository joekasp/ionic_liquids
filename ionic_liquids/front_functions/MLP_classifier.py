#import packages, lines starts with ## can be deleted, need finial decision
import pandas as pd
##import matplotlib.pyplot as plt
import numpy as np
##from rdkit import Chem
##from rdkit.Chem import AllChem
##from rdkit.Chem import Descriptors
##from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
##from sklearn.preprocessing import StandardScaler
##from sklearn.neural_network import MLPRegressor
##from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def MLP_Classifier(dataframe):
	'''Calling the MLPClassifier, passing the data into the NN'''
	alphas = np.array([5,2,5,1.5,1,0.1,0.01,0.001,0.0001,0]) 
	mlp_class = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
				solver='adam', alpha=0.0001, max_iter=5000, 
				random_state=None,learning_rate_init=0.01)
	gs = GridSearchCV(mlp_class, param_grid=dict(alpha=alphas))
	gs.fit(X_train,y_train) # Fit the NN weights for the training set
	
	return gs

