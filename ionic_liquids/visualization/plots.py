import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
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
    plt.scatter(y_act,y_pred)
    plt.plot([y_act.min(),y_act.max()],[y_act.min(),y_act.max()],lw=4,'r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    return fig 

