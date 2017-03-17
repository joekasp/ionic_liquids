import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA

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

def test_error_values():
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
    X_train = np.array([[0,6,234.321,5,0,1,1,0,1,18.000,1,0,0,0,0,298.15,101,0.004],
        [1,6,234.321,5,0,1,1,0,1,18.000,1,0,0,0,0,300.15,101,0.005],
        [2,6,234.321,5,0,1,1,0,1,18.000,1,0,0,0,0,302.15,101,0.006],
        [3,6,234.321,5,0,1,1,0,1,18.000,1,0,0,0,0,304.15,101,0.007],
        [4,6,234.321,5,0,1,1,0,1,18.000,1,0,0,0,0,306.15,101,0.005]])
    X_test = np.array([[5,6,234.321,5,0,1,1,0,1,18.000,1,0,0,0,0,300,101,0.0045],
        [6,6,234.321,5,0,1,1,0,1,18.000,1,0,0,0,0,301,101,0.0051],
        [7,6,234.321,5,0,1,1,0,1,18.000,1,0,0,0,0,301.5,101,0.0057]])
    Y_train = np.array([0.02,0.03,0.03,0.04,0.05])
    Y_test = np.array([0.03,0.04,0.05])
    assert isinstance(X_train,np.ndarray),"The X_train should have the datatype of numpy array"
    assert isinstance(X_test,np.ndarray),"The X_test should have the datatype of numpy array"
    assert isinstance(Y_train,type(X_train)), "Y_train should have the same datatype as X_train"
    assert isinstance(Y_test,type(X_test)), "Y_test should have the same datatype as X_test"
    #setting up parameters and variables for plotting 
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    assert n_train==Y_train.shape[0],"The training data and target value should have the same dimension"
    d = X_train.shape[1]
    assert d==X_test.shape[1],"The training and testing data should have the same amount of feature"
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
    assert yh[0]==0, "yh has the initialization has problem "
    assert yh2[0]==0, "yh2 has the initialization has problem "
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
