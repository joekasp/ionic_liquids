import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from methods import lasso, mlp_classifier, mlp_regressor, svr
from sklearn.metrics import mean_squared_error

def parity_plot(y_pred,y_act):
    """
    Creates a parity plot

    Input
    -----
    y_pred : predicted values from the model 
    y_act : 'true' (actual) values

    Output
    ------
    fig : matplotlib figure

    """

    fig = plt.figure(figsize=(4,4))
    plt.scatter(y_act,y_pred, alpha=0.5)
    plt.plot([y_act.min(),y_act.max()],[y_act.min(),y_act.max()],lw=4,'r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    return fig 

