#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 19:14:35 2019
@author: rohan devaki
"""

# Step 1 Load Data
import pandas as pd
dataset = pd.read_csv('credit_score_and_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

# Step 2: Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Step 3: Fit Simple Linear Regression to Training Data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 4: Make Prediction
y_pred = regressor.predict(X_test)

# Step 5 - Visualize training set results
import matplotlib.pyplot as plt
# plot the actual data points of training set
plt.scatter(X_train, y_train, color = 'red')
# plot the regression line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('creidt_score vs stipend (Training set)')
plt.xlabel('stipend')
plt.ylabel('creidt_score')
plt.show()

# Step 6 - Visualize test set results
import matplotlib.pyplot as plt
# plot the actual data points of test set
plt.scatter(X_test, y_test, color = 'red')
# plot the regression line (same as above)
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('creidt_score vs stipend (Test set)')
plt.xlabel('stipend')
plt.ylabel('creidt_score')
plt.show()

# Step 7 - Make new prediction
new_credit_score = regressor.predict([[1100]])
print('The predicted credit score of a student with 1100 stipend is ',new_credit_score)