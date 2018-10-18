#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:16:28 2018

@author: aminselim
"""

# Boston Housing 

# Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

# Information about the dataset
dataset.head()
dataset.keys()
dataset.info()
dataset.describe()

# Visualize dataset
sns.set_palette('GnBu_d')
sns.set_style(style = 'whitegrid')
sns_plot = sns.pairplot(data = dataset)
sns_plot.savefig('output.png')
y_dist_plot = sns.distplot(dataset['medv'], axlabel = 'Median Value of Housing Unit')
y_dist_plot = y_dist_plot.get_figure()
y_dist_plot.savefig('medv_distribution')
dataset_heatmap = sns.heatmap(dataset.corr())
dataset_heatmap = dataset_heatmap.get_figure()
dataset_heatmap.savefig('Dataset_heatmap')
dataset_cor = dataset.corr()
sns.jointplot(x = 'rad', y = 'tax', data = dataset, kind = 'scatter')

# Define independent and dependent variables 
X_lin = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]].values # Drop ID and rad
X = dataset.iloc[:, 1:-1].values # Drop ID
X_test = dataset_test.iloc[:, 1:].values
y = dataset.iloc[:, 14].values
ID = dataset.iloc[:, 0].values

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_opt = np.append(arr = np.ones((333, 1)).astype(int), values = X_lin, axis = 1)
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()                
X_opt = X_opt[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] # Remove x1, crim
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()                
X_opt = X_opt[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]] # Remove x2, indus
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()       
X_opt = X_opt[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]] # Remove x7, tax
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()  
X_opt = X_opt[:, [0, 1, 2, 3, 4, 6, 7, 8, 9]] # Remove x5, age
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()  

# Remove insignificant and highly correlated columns
X_lin = dataset.iloc[:, [2, 4, 5, 6, 8, 11, 12, 13]].values
X_lin_test = dataset_test.iloc[:, [2, 4, 5, 6, 8, 11, 12, 13]].values

# Splitting the dataset into the Training set and CV set
from sklearn.model_selection import train_test_split
X_lin_train, X_lin_CV, y_train, y_CV = train_test_split(X_lin, y, test_size = 0.3, random_state = 0)
ID_train, ID_CV, y_train, y_CV = train_test_split(ID, y, test_size = 0.3, random_state = 0)

# Fitting Regression Model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_lin_train, y_train)

X_lin_col = dataset[['zn', 'chas', 'nox', 'rm', 'dis', 'ptratio', 'black', 'lstat']]

# The coefficients
lin_reg.intercept_
coeff_df = pd.DataFrame(lin_reg.coef_, X_lin_col.columns, columns = ['Coefficient'])
coeff_df

# Predicting the CV set results
y_pred_lin_CV = lin_reg.predict(X_lin_CV)

# Predicting the test set results
y_pred_lin = lin_reg.predict(X_lin_test)

# Compare to CV set results
plt.scatter(y_CV, lin_reg.predict(X_lin_CV), color = 'red')
plt.title('Compare regression results to CV set results')
plt.xlabel('y')
plt.ylabel('Predicted y')
plt.show()
          
# Fitting SVR model

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_SVR = X
y_SVR = y.reshape(-1, 1)
X_SVR = sc_X.fit_transform(X_SVR)
y_SVR = sc_y.fit_transform(y_SVR)

# Splitting the dataset into the Training set and CV set
X_svr_train, X_svr_CV, y_svr_train, y_svr_CV = train_test_split(X_SVR, y_SVR, test_size = 0.3, random_state = 0)

# Fitting SVR
from sklearn.svm import SVR
svr_reg = SVR(C = 5)
svr_reg.fit(X_svr_train, y_svr_train)

# Predicting a new result with SVR
y_pred_svr_CV =  sc_y.inverse_transform(svr_reg.predict(X_svr_CV))
y_pred_svr = sc_y.inverse_transform(svr_reg.predict(sc_X.transform(X_test)))

# Compare to CV set results
plt.scatter(y_CV, y_pred_svr_CV, color = 'red')
plt.title('Compare SVR results to CV set results')
plt.xlabel('y')
plt.ylabel('Predicted y')
plt.show()

# Random Forest

# Splitting the dataset into the Training set and CV set
X_rf_train, X_rf_CV, y_rf_train, y_rf_CV = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_rf_test = X_test

# Fitting Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 500, random_state = 0)
rf_reg.fit(X_rf_train, y_rf_train)

# Predictions
y_pred_rf_CV = rf_reg.predict(X_rf_CV)
y_pred_rf_test = rf_reg.predict(X_rf_test)

# Compare to CV set results
plt.scatter(y_CV, y_pred_rf_CV, color = 'red')
plt.title('Compare Random Forest Regression results to CV set results')
plt.xlabel('y')
plt.ylabel('Predicted y')
plt.show()

# Check least SSE
sse_lin = sum((y_CV - y_pred_lin_CV)**2)
sse_svr = sum((y_CV - y_pred_svr_CV)**2)
sse_rfa = sum((y_CV - y_pred_rf_CV)**2)