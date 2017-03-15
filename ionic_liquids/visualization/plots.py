import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

FIG_SIZE = (4, 4)


def parity_plot(y_pred, y_act):
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

    fig = plt.figure(figsize=FIG_SIZE)
    plt.scatter(y_act, y_pred)
    plt.plot([y_act.min(), y_act.max()], [y_act.min(), y_act.max()],
             lw=4, color='r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    return fig


def train_test_error(e_train, e_test, model_params):
    """
    Creates a plot of training vs. test error

    Input
    -----
    e_train : numpy array of training errors
    e_test : numpy array of test errors
    model_params : independent parameters of model (eg. alpha in LASSO)

    Returns
    -------
    fig : matplotlib figure

    """

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(model_params, e_train, label='Training Set')
    plt.plot(model_params, e_train, label='Test Set')
    plt.xlabel('Model Parameter')
    plt.ylabel('MSE of model')
    plt.legend()

    return fig


def scatter_plot(x_vals, y_vals, x_variable):
    """
    Creates a plot of predicted electric conductivity as a
    function of the mole fractions.

    Input
    -----
    x_vals : numpy vector x-axis (mole fractions)
    y_vals : numpy vector y-axis (predicted conductivities)
    x_variable : string for labeling the x-axis

    Returns
    ------
    fig : matplotlib figure

    """
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
