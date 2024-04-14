# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:19:52 2024
Last Updated on Sun Apr 14  2024

@author: palac
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pds



def unit_step_func(x):
    return np.where(x > 0,1,1)


class Perceptron:
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
        
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # init parameters
        self.weights = np.zeros(n_features) #radnomly intialize them instead of zeros
        self.bias = 0
        
        
        y_ = np.where(y>0,0,1)
        
        # learn weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                # Perceptron update rule
                update = self.lr * (y_[idx]-y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self,X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    
    ## Testing
    
#Import Data
train_data_path = r"C:\Users\palac\OneDrive\Documents\Masters Program\TNN Proj\train.csv"
test_data_path = r"C:\Users\palac\OneDrive\Documents\Masters Program\TNN Proj\test.csv"
    
train_data = pds.read_csv(train_data_path, names=['Passenger','Survivde','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
test_data = pds.read_csv(test_data_path, names=['Passenger','Survivde','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
    
    
    # def accuracy(y_true, y_pred):
    #     accuracy = np.sum(y_true == y_pred)/len(y_true)
    #     return accuracy
    
    # Start the preceptron
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(train_data,test_data)
predictions = p.predicted(test_data)
                            
print('Percptron Classification accuracy', accuracy(test_data, predictions))
    # Plotting the predictions
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:,0], X_train[:,1], marker = "o", c=train_data)
    
x0_1 = np.amin(train_data[:0])
x0_2 = np.amax(train_data[:0])
    
x1_1 = (-p.weights[0]*x0_1-p.bias / p.weights[1])
x1_2 = (-p.weights[0]*x0_2-p.bias / p.weights[1])
    
ax.plot(x0_1)
                                                    
