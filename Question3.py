# -*- coding: utf-8 -*-
"""
@author: chujie zhang
"""

import numpy as np
import matplotlib.pyplot as plt

'''
   define the data set
'''
x_data=np.zeros((30,1))
y_data=np.zeros((30,1))
S_data=np.zeros((30,2))
x_test=np.zeros((1000,1))
y_test=np.zeros((1000,1))
# for question D
mse_D_trainingData=np.zeros(18)
mse_D_testData=np.zeros(18)
traingingData_temp=np.zeros((100,18))
testData_temp=np.zeros((100,18))
'''
   define the function g=sin(2*pi*x)**2+noise
'''
def createGx(x):
    #generate a random variable distributed normally 
    noise = np.random.normal(0,0.07)
    G=(np.sin(2*np.pi*x)**2)+noise
    #print(noise)
    return G

'''
   create data set for question b
'''
def createDataSet(number):
    for i in range(number):
        x_data[i][0]=np.random.random_integers(0,100)*0.01
        y_data[i][0]=createGx(x_data[i][0])
        S_data[i][0]=x_data[i][0]
        S_data[i][1]=y_data[i][0]
    return x_data,y_data,S_data

'''
   create data set for question c
'''
def createTestDataSet():
    for i_test in range(1000):
        #x_test[i_test][0]=np.random.randint(0,101)*0.01
        x_test[i_test][0]=np.random.random_integers(0,100)*0.01
        y_test[i_test][0]=createGx(x_test[i_test][0])

    return x_test,y_test

'''
 take a vector x and a K+1 -demensinal vector w as arguments
 and returns the value of the K-th order polynomial.
'''
def poly(x, weight):
    y=0
    for i in range(len(weight)):
         y += weight[i] * np.sin((i+1) * np.pi * x)
    return y

'''
   x is expected to be a numpy array of dimensions (N,)
   with features map
'''
def embedding_func(x, K):
    y=[]
    for i in range (0,len(x)):
        for j in range(0,K+1):
            y.append(np.sin((j+1) * np.pi * x[i]))
    F=np.array(y).reshape(len(x),K+1)        
    return F

'''
   fits a K-th order polynomial and returns a weight vector w
'''
def estimate_w(y, x, K):
    x=embedding_func(x, K)
    w=np.linalg.solve(x.T@x, x.T@y)
    return w
#-------------------------------------------------------------------------------

def questionB(x_data,y_data):
    #initial the sum of squared errors SSE
    sse = np.zeros(18)
    for poly_degree in range(1,19):
        w = estimate_w(y_data, x_data, poly_degree-1)  
        sse[poly_degree-1]=np.sum((y_data-poly(x_data,w))**2)
    # give the mean square error MSE
    mse=sse/30
    return mse

def questionC(x_test,y_test,x_data,y_data):
    #initial the sum of squared errors SSE
    sse_test = np.zeros(18)
    for poly_degree_test in range(1,19):
        w = estimate_w(y_data, x_data, poly_degree_test-1)  
        sse_test[poly_degree_test-1]=np.sum((y_test-poly(x_test,w))**2)
    # give the mean square error MSE
    mse_test=sse_test/1000
    return mse_test

def questionD():
    for i in range(100):
        x_data,y_data,S_data=createDataSet(30)
        x_test,y_test=createTestDataSet()
        traingingData_temp[i]=questionB(x_data,y_data)
        testData_temp[i]=questionC(x_test,y_test,x_data,y_data)
        for j in range(18):
            mse_D_trainingData[j]+=traingingData_temp[i][j]
            mse_D_testData[j]+=testData_temp[i][j]
    averageMse_trainingData=mse_D_trainingData*0.01
    averageMse_testData=mse_D_testData*0.01
    return averageMse_trainingData,averageMse_testData
#-------------------------------------------------------------------------------
'''
   initial the data set and plot for question A
'''
x_data,y_data,S_data=createDataSet(30)
x_test,y_test=createTestDataSet()

'''
   to plot training error and test error for question B and question C
'''
fig, ax = plt.subplots(figsize=(14, 10))
mse=questionB(x_data,y_data)
mse_test=questionC(x_test,y_test,x_data,y_data)
ax.plot(np.log(mse), label="training error")
ax.plot(np.log(mse_test), label="test error")
ax.set_xlim(0,18)
ax.legend()
plt.show()

'''
   to plot training error and test error for question D
'''
averageMse_trainingData,averageMse_testData=questionD()
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(np.log(averageMse_trainingData), label="Average 100runs of training error")
ax.plot(np.log(averageMse_testData), label="Average 100runs of test error")
ax.set_xlim(0,18)
ax.legend()
plt.show()