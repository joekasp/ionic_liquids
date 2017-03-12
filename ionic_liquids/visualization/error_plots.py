import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def error_plot(object,X_test,y_test):
		plot = plt.figure(figsize=(4,4))
		plot.scatter(y_test.values.astype(np.float),object.predict(X_test))
		plot.plot([0,12],[0,12],lw=4,c='r')
		plot.show()
		return plot 
