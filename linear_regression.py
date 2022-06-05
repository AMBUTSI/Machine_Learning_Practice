#!/usr/bin/env python
# coding: utf-8

#load data
import seaborn as sns
iris = sns.load_dataset('iris')
iris

#reduce variables to two variables
iris = iris[['petal_length', 'petal_width']]
iris

#Check whether the two variables are linearly correlated
x = iris['petal_length']
y = iris['petal_width']

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.xlabel("petal length")
plt.ylabel("petal width")

#divide data between train and test data
#0.4 - 40% for testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.4, random_state=23)

x_train

#make x two-dimensional
import numpy as np
x_train = np.array(x_train).reshape(-1, 1)
x_train

x_test = np.array(x_test).reshape(-1,1)
x_test

#Import the Linear Regression model from the scikit-learn library
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#fit the training data to the regression model
lr.fit(x_train, y_train)
#training parameters m and c
c = lr.intercept_
c

m = lr.coef_
m

#predict the 'y' using x
y_pred_train = m*x_train + c
y_pred_train.flatten()

y_pred_train1 = lr.predict(x_train)
y_pred_train1

#Add a line plot to check if the prediction is right
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train)
plt.plot(x_train, y_pred_train1, color='red')
plt.xlabel("petal length")
plt.ylabel("petal width")

#check if the model works on test data as well
y_pred_test1 = lr.predict(x_test)
y_pred_test1

#confirm if the prediction is right with a plot
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred_test1, color = 'red')
plt.xlabel("petal length")
plt.ylabel("petal width")

#Multiple Linear Regression
import pandas as pd
df = pd.read_csv('insurance.csv')
df
#Change string values to categorical values
df['sex'] = df['sex'].astype('category')
df['sex'] = df['sex'].cat.codes

df['smoker'] = df['smoker'].astype('category')
df['smoker'] = df['smoker'].cat.codes

df['region'] = df['region'].astype('category')
df['region'] = df['region'].cat.codes

df
#Check if we have null values
df.isnull().sum()

#Separate x from y. 6 variables will be used to predict the expenses
x = df.drop(columns = 'expenses')
x

#y will be the expenses
y = df['expenses']

#Create training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state =23)

#LinearRegression model
lr_multiple = LinearRegression()
lr_multiple.fit(x_train, y_train)

#Training Parameters
c = lr_multiple.intercept_
c

m = lr_multiple.coef_
m
#Predict price for training and test data
y_pred_train = lr_multiple.predict(x_train)
y_pred_test = lr_multiple.predict(x_test)


#R2 to indicate goodness of fit
from sklearn.metrics import r2_score
r2_score(y_test, y_pred_test)

