import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA

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

def error_values(X_train,X_test,Y_train,Y_test):
    """
    Creates the two predicted values

    Input
    -----
    X_train : numpy array, the 10% of the training set data values
    X_test : numpy array, the molecular descriptors for the testing set data values 

    Y_train: numpy array, the 10% of the training set of electronic conductivity values
    Y_test: numpy array, 'true' (actual) electronic conductivity values
    
    Output
    ------
    yh : numpy array, the prediction output for training data set
    yh2 : numpy array, the prediction output for the testing data set 
    """
    #setting up parameters and variables for plotting 
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    d = X_train.shape[1]
    hdnode = 100
    w1 = np.random.normal(0,0.001,d*hdnode).reshape((d,hdnode))
    d1 = np.zeros((d,hdnode))
    w2 = np.random.normal(0,0.001,hdnode).reshape((hdnode,1))
    d2 = np.zeros(hdnode)
    h  = np.zeros(hdnode)
    mb = 100 #minibatch size
    m = int(n_train/mb)
    batch = np.arange(m) 
    lr = 0.00020
    EP = 20000 #needed for initializing 
    ep = 0
    yh = np.zeros((n_train,1))
    yh2 = np.zeros((n_test,1))
    L_train= np.zeros(EP+1)
    L_test = np.zeros(EP+1)
    Y_train = Y_train.reshape(len(Y_train),1)
    #activation function for the hidden layer is tanh
    
    def g(A):
        return (np.tanh(A))

    def gd(A):
        return (1-np.square(np.tanh(A)))
        
    #setting up how long the epoch will run
    EP = 200
    ep = 0
    while ep < EP:
        ep += 1
        yh = g(X_train.dot(w1)).dot(w2)
        yh2 = g(X_test.dot(w1)).dot(w2)
        L_train[ep] = LA.norm(yh-Y_train.reshape(len(Y_train),1))/n_train
        L_test[ep]  = LA.norm(yh2-Y_test.reshape(len(Y_test),1))/n_test
        
        np.random.shuffle(batch)
        for i in range(m):
            st = batch[i]*mb
            ed = (batch[i]+1)*mb
            h  = g(X_train[st:ed].dot(w1))
            y = h.dot(w2)
            d2 = h.T.dot(Y_train[st:ed]-y)
            d1 = X_train[st:ed].T.dot(np.multiply((Y_train[st:ed]-y).dot(w2.T),gd(X_train[st:ed].dot(w1))))
            w2 += lr*d2
            w1 += lr*d1
    return yh, yh2


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
