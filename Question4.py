# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:58:02 2019

@author: chujie zhang
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#define the data set
features_first=np.zeros((506,12))
features=np.zeros((506,13))
prices=np.zeros((506,1))
'''
   read data from .csv file
'''
def loadDataSet():
    dataSet = pd.read_csv('Boston-filtered.csv')
    #print(dataSet)
    prices[:,0] = dataSet['MEDV']
    features_first=dataSet.drop('MEDV', axis = 1)
    return features_first,prices
    #print("features",features[' ZN '][6])
    #print("prices",prices.shape)
'''
    inputs are augmented with an additional 1 entry, (xi,1)
'''
def embedding_func(features_first):
    y=np.zeros((len(features_first),13))
    for i in range (0,len(features_first)):    
        for j, key in enumerate(['CRIM', ' ZN ', 'INDUS ','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT',]):
            
            y[i][j]=features_first[key][i]
            y[i][12]=1
    #F=np.array(y).reshape(506,13)        
    return y

'''
    split data randomly
'''
def splitDataSet(features,prices):
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=1/3)
    return X_train, X_test, y_train, y_test

'''
  take a vector x and a K+1 -demensinal vector w as arguments
  and returns the value of the K-th order polynomial.
'''
def poly(x, weight):
    y = weight.T@x
    return y
'''
   Define a function that fits a K-th order polynomial and returns a weight vector w
'''
def estimate_w(y, x):
    w=np.linalg.solve(x.T@x, x.T@y)
    return w


def questionA(x_train,y_train,x_test,y_test):
    vector_training=np.ones((len(y_train),1))
    w_train = estimate_w(y_train, vector_training)
    sse_train=np.sum((y_train-(w_train.T@vector_training.T).T)**2)  
    mse_train=sse_train/len(x_train)
    #print(" the MSE on the training set for question a:",mse_train)
   
    vector_test=np.ones((len(y_test),1))
    sse_test=np.sum((y_test-(w_train.T@vector_test.T).T)**2)  
    mse_test=sse_test/len(x_test)
    #print("the MSE on the test set for question a:",mse_test)
    return mse_train,mse_test

def questionC(features,prices):
     x_train, x_test, y_train, y_test=splitDataSet(features,prices)
     x=np.zeros((len(x_train),2)) 
     xTest=np.zeros((len(x_test),2)) 
     w=np.zeros((12,2,1))
     mse_train=np.zeros(12)
     mse_test=np.zeros(12)
     for key in range(12):
         for i in range (0,len(x_train)):  
             x[i][0]=x_train[i][key]
             x[i][1]=1
         w[key] = estimate_w(y_train, x)
         mse_train[key]=(np.sum((y_train-(w[key].T@x.T).T)**2))/len(x_train)  
          
         
     for key in range(12):
         for i in range (0,len(x_test)):  
             xTest[i][0]=x_test[i][key]
             xTest[i][1]=1
         mse_test[key]=(np.sum((y_test-(w[key].T@xTest.T).T)**2))/len(x_test) 
     #print("for the features:",key,w)
     #print(mse_train)
     return mse_train,mse_test,w

def questionD(x_train,y_train,x_test,y_test):     
    w_train = estimate_w(y_train, x_train)
    sse_train=np.sum((y_train-(w_train.T@x_train.T).T)**2)  
    mse_train=sse_train/len(x_train)
    #print("the MSE on the training set for question d:",mse_train)
    
    sse_test=np.sum((y_test-(w_train.T@x_test.T).T)**2)  
    mse_test=sse_test/len(x_test)
    #print("the MSE on the test set for question d:",mse_test)

    return mse_train,mse_test
'''
   read the data from .csv file
'''
features_first,prices = loadDataSet()
features=embedding_func(features_first)
x_train, x_test, y_train, y_test=splitDataSet(features,prices)

'''
   20 runs for question a 
'''
sum_train_A = 0
sum_test_A = 0
standDevationsTestA=np.zeros(20)
standDevationsTrainingA=np.zeros(20)
for i in range(20):
    #(each run based on a diﬀerent (2/3,1/3) random split
    x_train, x_test, y_train, y_test=splitDataSet(features,prices)
    mse_train,mse_test=questionA(x_train,y_train,x_test,y_test)
    sum_train_A += mse_train
    sum_test_A += mse_test
    standDevationsTestA[i]=mse_test
    standDevationsTrainingA[i]=mse_train
print("The average mse on training error for question a:",sum_train_A/20)
print("The average mse on test error for question a:",sum_test_A/20)
print("The standard dervations on training error for question a:",np.std(standDevationsTrainingA))
print("The standard dervations on test error for question a:",np.std(standDevationsTestA))

'''
   20 runs for question c
'''
sum_mse_train_C=np.zeros(12)
sum_mse_test_C=np.zeros(12)
sum_w_C=np.zeros((12,2,1))
standDevationsTestC=np.zeros((12,20))
standDevationsTrainingC=np.zeros((12,20))
for i in range(20):
    mse_train,mse_test,w=questionC(features,prices)  
    sum_mse_train_C += mse_train
    sum_mse_test_C += mse_test
    sum_w_C += w
    standDevationsTrainingC[:,i]=mse_train
    standDevationsTestC[:,i]=mse_test
for j in range(12):
    print("For feature",j+1, "the weight is",sum_w_C[j]/20, "Average mse on training set",sum_mse_train_C[j]/20,"Average mse on test set",sum_mse_test_C[j]/20  )
    print( "For feature",j+1, " the standard devation on training set is",np.std(standDevationsTrainingC[j,:]))
    print( "For feature",j+1, " the standard devation on test set is",np.std(standDevationsTestC[j,:]))

'''
   20 runs for question d
'''
sum_train_D = 0
sum_test_D = 0
standDevationsTestD=np.zeros(20)
standDevationsTrainingD=np.zeros(20)
for i in range(20):
    #(each run based on a diﬀerent (2/3,1/3) random split
    x_train, x_test, y_train, y_test=splitDataSet(features,prices)
    mse_train,mse_test=questionD(x_train,y_train,x_test,y_test)
    sum_train_D += mse_train
    sum_test_D += mse_test
    standDevationsTestD[i]=mse_test
    standDevationsTrainingD[i]=mse_train
print("The average mse on training error for question d:",sum_train_D/20)
print("The average mse on test error for question d:",sum_test_D/20)
print("The standard dervations on training error for question a:",np.std(standDevationsTrainingD))
print("The standard dervations on test error for question a:",np.std(standDevationsTestD))



    