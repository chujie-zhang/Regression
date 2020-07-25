# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 00:41:43 2019

@author: chujie zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D

''' 
   create the gamma and sigma
'''
def initialGammaAndSigma():
    gamma=[]
    sigma=[]
    for i in range(0,15):
        gamma += [pow(2,-40+i)]
    for j in {7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13}:
        sigma+=[2**j]
    return gamma,sigma
'''
   read data from .csv file
'''
def loadData():
    dataSet = pd.read_csv("Boston-filtered.csv") 
    prices = np.array((dataSet.iloc[:, 12]).tolist()).reshape(-1, 1)
    features = dataSet.iloc[:, 0:12].values
    return features,prices

'''
   get one kernel in the matrix
'''
def getOneKernel(xi,xj,sigma):
    k=np.exp((-(np.linalg.norm(xi-xj))**2)/(2*sigma**2))
    return k

'''
   create the kernel matrix
'''
def getKernelMatrix(x_data,sigma):
    kernelMatrix = np.zeros((len(x_data),len(x_data)))
    for i in range(len(x_data)):
        for j in range(len(x_data)):
            kernelMatrix[i][j] = getOneKernel(x_data[i,:],x_data[j,:],sigma)     
    return kernelMatrix

'''
   compute the alpha
'''
def getAlpha(kernelMatrix,y_training,gamma):
    temp_1 = kernelMatrix+ ((gamma*len(y_training)) * np.identity(len(y_training)))
    alpha = np.linalg.inv(temp_1) @ y_training
    return alpha
'''
    The evaluation of the regression function 
'''
def getYtest(alpha,x_training,x_test,sigma):
    y_test = np.zeros((len(x_test),1))
    for i in range(len(x_test)):
        sum = 0
        for j in range(len(x_training)):
            sum = sum + alpha[j,:] * getOneKernel(x_training[j,:],x_test[i,:],sigma)
        y_test[i][0] = sum
    return y_test

def question_A_B(traning_x,test_x,traning_y,test_y):
    kf = KFold(n_splits=5)
    minError =200
    sigma_X =[]
    gamma_Y =[]
    mse_Z =[]
    for i in range(0,15):
        for j in range(0,13):
            crossValidationError =0
            for train_index, test_index in kf.split(traning_x):
                x_train, x_test = traning_x[train_index], traning_x[test_index]
                y_train, y_test = traning_y[train_index], traning_y[test_index]
           
                kernelMatrix = getKernelMatrix(x_train, sigma[j])  
                alpha = getAlpha(kernelMatrix, y_train, gamma[i])
                predict_y = getYtest(alpha, x_train, x_test, sigma[j])
                mse = np.sum((predict_y-y_test)**2)/len(predict_y)
                
                crossValidationError = crossValidationError + mse
            
            crossValidationError =crossValidationError/5
            mse_Z = mse_Z + [crossValidationError]
            gamma_Y = gamma_Y + [gamma[i]]
            sigma_X = sigma_X + [sigma[j]]
            
          
            if(minError > crossValidationError):
                minError = crossValidationError
                bestGamma=gamma[i]
                bestSigma=sigma[j]
                print("The current error", crossValidationError)
                print("Current best sigma:",sigma[j],"Current best gamma:",gamma[i])
    ax = plt.subplot(111, projection='3d')
    ax.scatter(sigma_X, gamma_Y, mse_Z, c='y')
    ax.set_zlabel('mse')
    ax.set_ylabel('gamma')
    ax.set_xlabel('sigma')
    plt.show()
    return bestGamma,bestSigma

def questionC(bestGamma,bestSigma,features_training, features_test, prices_traning, prices_test):
    
    kernelMatrix = getKernelMatrix(features_training, bestSigma)
    alpha = getAlpha(kernelMatrix, prices_traning, bestGamma)
    predict_y_test = getYtest(alpha, features_training, features_test, bestSigma)
    predict_y_train = getYtest(alpha, features_training, features_training, bestSigma)
    
    mse_training = np.sum((predict_y_train-prices_traning)**2)/len(predict_y_train)
    mse_test = np.sum((predict_y_test-prices_test)**2)/len(predict_y_test)
   
    print("mse_test",mse_test)
    print("mse_training",mse_training)
    return mse_test,mse_training

def questionD(features,prices):
    mse_training=np.zeros(20)
    mse_test=np.zeros(20)
    for i in range(20):
        features_training, features_test, prices_traning, prices_test = train_test_split(features, prices,test_size=1/3)
        bestGamma,bestSigma=question_A_B(features_training, features_test, prices_traning, prices_test)
        kernelMatrix = getKernelMatrix(features_training, bestSigma)
        alpha = getAlpha(kernelMatrix, prices_traning, bestGamma)
        predict_y_test = getYtest(alpha, features_training, features_test, bestSigma)
        predict_y_train = getYtest(alpha, features_training, features_training, bestSigma)
    
        mse_training[i] = np.sum((predict_y_train-prices_traning)**2)/len(predict_y_train)
        mse_test[i] = np.sum((predict_y_test-prices_test)**2)/len(predict_y_test)
    standardDeviationTraining=np.std(mse_training)
    standardDeviationTest=np.std(mse_test)
    print("standard deviation on the training set is",standardDeviationTraining)
    print("standard deviation on the test set is",standardDeviationTest)
    return standardDeviationTraining,standardDeviationTest
'''
   create gamma and sigma
   read data from .csv file
   split data set randomly
'''
gamma,sigma=initialGammaAndSigma()
features,prices=loadData()
features_training, features_test, prices_traning, prices_test = train_test_split(features, prices,test_size=1/3)
'''
   For question a and b
'''
bestGamma,bestSigma = question_A_B(features_training, features_test, prices_traning, prices_test)
'''
   For question c
'''
mse_test,mse_training = questionC(bestGamma,bestSigma,features_training, features_test, prices_traning, prices_test)
'''
   For question d
'''
standardDevaiationTraining,standardDevaiationTest = questionD(features,prices)