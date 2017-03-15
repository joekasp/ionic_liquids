import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

FIG_SIZE = (4, 4)


def test_parity_plot():
    """
    Test the parity plot

    Input
    -----
    y_pred : predicted values from the model
    y_act : 'true' (actual) values

    Output
    ------
    fig : matplotlib figure

    Check:
    1. The x,y vector has the same datatype
    2. The x,y vector has the same dimension 
    """
    y_pred=np.arange(0,1)
    y_act=np.arange(0,1)
    assert isinstance(y_pred,type(y_act)), "The two column in the parity plot should have same datatype"
    assert len(y_pred)==len(y_act), "The two column in the parity plot should have same length"
    fig = plt.figure(figsize=FIG_SIZE)
    plt.scatter(y_act, y_pred)
    plt.plot([y_act.min(), y_act.max()], [y_act.min(), y_act.max()],
             lw=4, color='r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    return fig


def test_train_test_error():
    """
    Test the plot of training vs. test error

    Input
    -----
    e_train : numpy array of training errors
    e_test : numpy array of test errors
    model_params : independent parameters of model (eg. alpha in LASSO)

    Returns
    -------
    fig : matplotlib figure
    
    Check:
    1. The e_train, e_test and model_params has the same dimension

    """
    e_train = np.arange(0,1)
    e_test = np.arange(0,1)
    model_params = np.arange(0,1)
    assert len(e_train)==len(model_params), "The training error and model parameters should have the same dimension"
    assert len(e_test)==len(model_params), "The test error and model parameters should have the same dimension"
    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(model_params, e_train, label='Training Set')
    plt.plot(model_params, e_train, label='Test Set')
    plt.xlabel('Model Parameter')
    plt.ylabel('MSE of model')
    plt.legend()

    return fig


def test_scatter_plot():
    """
    Test plot of predicted electric conductivity as a
    function of the mole fractions.

    Input
    -----
    x_vals : numpy vector x-axis (mole fractions)
    y_vals : numpy vector y-axis (predicted conductivities)
    x_variable : string for labeling the x-axis

    Returns
    ------
    fig : matplotlib figure

    Check:
    1. The x_variable is a string
    2. The x,y vector has the same dimension

    """
    x_variable = 'm'
    x_vals = np.arange(0,1)
    y_vals = np.arange(0,1)
    assert isinstance(x_variable,str), "x_variable should be a string variable"
    assert len(x_vals)==len(y_vals), "The x and y vector should have the same dimension"

    if (x_variable == 'm'):
        x_variable = 'Mole Fraction A'
    elif (x_variable == 'p'):
        x_variable = 'Pressure (kPa)'
    elif (x_variable == 't'):
        x_variable = 'Temperature (K)'
    fig = plt.figure(figsize=FIG_SIZE)
    plt.scatter(x_vals, y_vals)
    plt.xlabel(x_variable)
    plt.ylabel('Electrical Conductivity')

    return fig
