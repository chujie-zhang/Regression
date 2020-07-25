# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:44:39 2019

@author: chujie zhang
"""

import numpy as np
import matplotlib.pyplot as plt

'''
   Initial dataset for using later
'''
def initialDataSet():
    x_data=np.asarray([[1],[2],[3],[4]])
    y_data=np.asarray([[3],[2],[0],[5]])
    return x_data,y_data

'''
 take a vector x and a K+1 -demensinal vector w as arguments
 and returns the value of the K-th order polynomial.
'''
def poly(x, weight):
    y=0
    for i in range(len(weight)):
        y += weight[i]*x**i
    return y

'''
   x is expected to be a numpy array of dimensions (N,)
   with features map
'''
def embedding_func(x, K):
    y=[]
    for i in range (0,len(x)):
        for j in range(0,K+1):
            y.append(x[i]**j)
    F=np.array(y).reshape(len(x),K+1)        
    return F

'''
   fits a K-th order polynomial and returns a weight vector w
'''
def estimate_w(y, x, K):
    x=embedding_func(x, K)
    w=np.linalg.solve(x.T@x, x.T@y)
    return w

#-----------------------------------------------------------------------------

x_data,y_data = initialDataSet()
x = np.linspace(0.0, 4.0, num=100, endpoint=True)
fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(x_data, y_data, c='g', marker='.', label="data set")
'''
   For question 1.a, to produce a plot similar to figure 1, 
   superimposing the four diﬀerent curves corresponding to each ﬁt over the four data points. 
'''
for K in range(1,5):
    w = estimate_w(y_data, x_data, K-1)
    y = poly(x, w)

    ax.plot(x,y,label="%d-th order"%K)
    '''
       For question 1.b, print w for me to do the question 
    '''
    print("for each K:",K,"the weight vector is:",w)
ax.legend()
plt.show() 
'''   
   For question 1.c
'''
#initial the sum of squared errors SSE
sse = np.zeros(4)
for poly_degree in range(1,5):
    mse=0;
    w = estimate_w(y_data, x_data, poly_degree-1)  
    sse[poly_degree-1]=np.sum((y_data-poly(x_data,w))**2)
    # For each ﬁtted curve k = 1,2,3,4 give the mean square error MSE
    # as we know, m=4
    mse=sse/4
    print("for curve", poly_degree,"the mse is:",mse[poly_degree-1])
#print(mse)

#plot sse and mse   
fig, ax = plt.subplots()
ax.plot(sse, label="sse error")
ax.plot(mse, label="mse error")
ax.legend()
plt.show()
