import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA

class regressor():
    def __init__(self, w1, w2):
        self.weight1 = w1
        self.weight2 = w2
        
    def predict(self, X):
        return np.tanh(X.dot(self.weight1)).dot(self.weight2)
    
    
def MLP(X,Y):
    X = StandardScaler().fit_transform(X)
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.10,random_state=1010)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    d = X_train.shape[1]
    #print(X_train.shape, X_test.shape)
    #print(Y_train.shape, Y_test.shape)
    #print(e_train.shape, e_test.shape)

    #initializing weight for first layer(w1) and second
    #Parameters
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
    EP =20000
    y = np.zeros((mb,1))
    yh = np.zeros((n_train,1))
    yh2 = np.zeros((n_test,1))

    L_train= np.zeros(EP+1)
    L_test = np.zeros(EP+1)

    L01_train = np.zeros((EP+1))
    L01_test = np.zeros((EP+1))

    #tanh
    def g(A):
        return (np.tanh(A))

    def gd(A):
        return (1-np.square(np.tanh(A)))
    ep = 0

    while ep < EP:
        ep += 1

        yh = g(X_train.dot(w1)).dot(w2)
        yh2 = g(X_test.dot(w1)).dot(w2)

        L_train[ep] = LA.norm(yh-Y_train)/n_train
        L_test[ep]  = LA.norm(yh2-Y_test)/n_test

        #print(ep,L_train[ep],L_test[ep])

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

    regr = regressor(w1,w2)
    return regr

