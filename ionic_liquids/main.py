import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import utils
from methods import methods
from visualization import plots


FILENAME = 'datasets/inputdata.xlsx'
MODEL = 'mlp_regressor'
DIRNAME = 'my_test'

# Get X matrix and response vector y
df, y_error = utils.read_data(FILENAME)
X, y = utils.molecular_descriptors(df)

Y = np.empty((y.shape[0],2))
Y[:,0] = y.ravel()
Y[:,1] = y_error.ravel()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.10)

y_train = Y_train[:,0]
y_test = Y_test[:,0]
e_train = Y_train[:,1]
e_test = Y_test[:,1]

# Normalize testing data using training data
X_train, X_mean, X_std = utils.normalization(X_train)
X_test, trash, trash = utils.normalization(X_test, X_mean, X_std)

# Do machine_learning call
if (MODEL.lower() == 'mlp_regressor'):
    obj = methods.do_MLP_regressor(X_train, y_train.ravel())
elif (MODEL.lower() == 'lasso'):
    obj = methods.do_lasso(X_train, y_train.ravel())
elif (MODEL.lower() == 'svr'):
    obj = methods.do_svr(X_train, y_train.ravel())
else:
    raise ValueError("Model not supported")

#Plots
simple_parity_plot = plots.parity_plot(y_train, obj.predict(X_train))

#Creates a prediction and experimental values plot
predict_train = np.zeros(X_train.shape[0])
predict_test = np.zeros(X_test.shape[0])
predict_train, predict_test = plots.error_values(np.copy(X_train),np.copy(X_test),np.copy(y_train),np.copy(y_test))

#Creates a plot for the training data set
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.scatter(y_train,predict_train,s=0.5,color='blue')
plt.title('Prediction on training data')
plt.plot(np.linspace(0,12,1000),np.linspace(0,12,1000),color='black')
plt.xlim((0,12))
plt.ylim((0,12))
plt.xlabel("Experiment($S*m^2/mol$)")
plt.ylabel("Prediction($S*m^2/mol$)")

#Creates a plot for the testing data set  
plt.subplot(1,2,2)
plt.scatter(y_test,predict_test,s=2,color='blue')
plt.title('Prediction on test data')
plt.xlim((0,12))
plt.ylim((0,12))
plt.xlabel("Experiment($S*m^2/mol$)")
plt.ylabel("Prediction($S*m^2/mol$)")
plt.plot(np.linspace(0,12,1000),np.linspace(0,12,1000),color='black')
plt.show()

#Creates a plot for the experimental and predicted data to the experimental error provided by the ILThermo database
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

#Set up the training data set plot
result = pd.DataFrame(columns=['Experiment','Prediction','error'])
result.Experiment = y_train
result.Prediction = predict_train
result.error = e_train
result = result.sort_values(['Experiment','Prediction'],ascending=[1,1])

#Creates a subplot for the training data set
size=0.2
ax1.set_xlim((0,2300))
ax1.set_ylim((-1,13))
ax1.scatter(np.arange(X_train.shape[0]),result.Experiment,color="blue",s=size,label='Experiment')
ax1.scatter(np.arange(X_train.shape[0]),result.Prediction,color="red",s=size,label='Prediction')
ax1.scatter(np.arange(X_train.shape[0]),result.Experiment+result.error,color="green",s=size,label='Experiment Error')
ax1.scatter(np.arange(X_train.shape[0]),result.Experiment-result.error,color="green",s=size)
ax1.set_title('Prediction on Training data')
ax1.legend(loc='upper left')

#Setting up the test data set plot 
result = pd.DataFrame(columns=['Experiment','Prediction','error'])
result.Experiment = y_test
result.Prediction = predict_test
result.error = e_test
result = result.sort_values(['Experiment','Prediction'],ascending=[1,1])

#Creates a subplot for the testing data set
size=2
ax2.set_xlim((0,260))
ax2.set_ylim((-1,13))
ax2.scatter(np.arange(X_test.shape[0]),result.Experiment,color="blue",s=size,label='Experiment')
ax2.scatter(np.arange(X_test.shape[0]),result.Prediction,color="red",s=size,label='Prediction')
ax2.scatter(np.arange(X_test.shape[0]),result.Experiment+result.error,color="green",s=size,label='Experiment Error')
ax2.scatter(np.arange(X_test.shape[0]),result.Experiment-result.error,color="green",s=size)
ax2.set_title('Prediction on test data')
ax2.legend(loc='upper left')

#Personal preference additions to the plot
ax.set_xlabel('Data points')
ax.set_ylabel('Conductivity($S*m^2/mol$)')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
fig.tight_layout()
plt.show()
