import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def model_types():
    models=['MLP Regressor','MLP Classifier','LASSO','SVR']
    return models

def scatter_plot(X,Y):
    """
    Draws a scatter plot of data

    Inputs
    ------
    X : Input vector of predictor(s) and parameters
    Y : Prediction of interest

    Returns
    -------
    out : Matplotlib axes object containing plot

    """
    pass


def predict_model():
    print("prediction")
    
def parity_plot(y_act,y_pred):
    """
    Creates a parity plot

    Input
    -----
    y_act : numpy array of 'true' (actual) values
    y_pred : numpy array of predicted values from the model 

    Output
    ------
    fig : matplotlib figure

    """

    fig = plt.figure(figsize=(4,4))
    plt.scatter(y_act.values.astype(np.float),y_pred)
    plt.plot([y_act.min(),y_act.max()],[y_act.min(),y_act.max()],lw=4,c='r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    return fig

