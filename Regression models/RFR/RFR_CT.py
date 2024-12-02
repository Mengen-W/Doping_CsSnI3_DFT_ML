#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:19:51 2024

@author: Chadawan Khamdang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

### Reading training data (CT for X_Sn) ###

data = pd.read_csv('../Dataset/Training dataset_CT.csv')

training_data = data.to_numpy()
dopant = training_data[0:, 0]
feature = training_data[0, 1:-1]
CT = training_data[0:, -1]
X = training_data[0:, 1:-1]
n = feature.size

print(training_data)
print(dopant)
print(len(dopant))
print(CT.size)

### Split train and test data ###

X_train, X_test, CT_train, CT_test = train_test_split(X, CT, test_size=0.2)
        
### Defining hyperparameter ###
'''
param_grid = {
"n_estimators": [50, 100, 150, 200],
"max_features": [3, 4, 5, 6, 7],
"max_depth": [10, 20, 30, 40, 50],
"min_samples_leaf": [1, 2, 3, 4],
"bootstrap": [True]             
}
'''
param_grid = {
"max_features": [7],
"max_depth": [50],
}


### Train the model ###
    
rfreg_opt = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_grid, cv=5)

rfreg_opt.fit(X_train, CT_train)
print('Best estimator by RandomizedSearchCV:', rfreg_opt.best_estimator_)

Pred_train = rfreg_opt.predict(X_train)
Pred_test = rfreg_opt.predict(X_test)

feature_importances = rfreg_opt.best_estimator_.feature_importances_
feature_names = ['EA',  '1st IE', '2nd IE', '3rd IE', 'D', 't (AR) 100%', 'u (IR) 100%', 'OS', 'HF (exp)', 'S', 'HV']

feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

print('Feature importances =', feature_importances_df)

plt.figure(figsize=(8, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='#371B58')
plt.xlabel('Relative feature importance', fontsize=22)
plt.title('Feature Importances', fontsize=22)
plt.gca().invert_yaxis()
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=20)
plt.yticks(fontsize=20)
plt.show()

### Parity plot and RMSE ###

mse_test_prop  = sklearn.metrics.mean_squared_error(CT_test, Pred_test)
mse_train_prop = sklearn.metrics.mean_squared_error(CT_train, Pred_train)
rmse_test_prop  = np.sqrt(mse_test_prop)
rmse_train_prop = np.sqrt(mse_train_prop)
print('rmse_test_form  = ', rmse_test_prop)
print('rmse_train_form = ', rmse_train_prop)

### Plotting results ###

fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
fig.text(0.5, -0.01, 'DFT Calculation', ha='center', fontsize=26)
fig.text(-0.01, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=26)

plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.95, wspace=0.25, hspace=0.4)

a = [-175,125]
b = [-175,125]
ax.plot(b, a, c='k', ls='-')

ax.xaxis.set_tick_params(labelsize=26)
ax.yaxis.set_tick_params(labelsize=26)

ax.scatter(CT_test[:], Pred_test[:], c='#615EFC', marker='o', s=60, label='Test')
ax.scatter(CT_train[:], Pred_train[:], c='#FA7070', marker='o', s=60, label='Train')

te = '%.2f' % rmse_test_prop
tr = '%.2f' % rmse_train_prop

ax.set_ylim([-1.0, 2.0])
ax.set_xlim([-1.0, 2.0])

ax.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

ax.set_title('Charge transition level (+1/0)', c='k', fontsize=26, pad=10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels, fontsize=18)
plt.tick_params(axis='y', width=2,length=4, labelsize=24) 
plt.tick_params(axis='x', width=2, length=4,  labelsize=24) 

ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
plt.show()

### Reading out-of-sample data and predict CT ###

read_data_out = pd.read_csv('../Dataset/Testing dataset.csv')

data_out = read_data_out.to_numpy()

dopant_out = data_out[0:, 0]
feature_out = data_out[0, 1:-1]
X_out = data_out[0:, 1:]
n_out = dopant_out.size

X_out_fl = [[0.0 for a in range(n)] for b in range(n_out)]

for i in range(0,n_out):
    for j in range(0,n):
        X_out_fl[i][j] = float(X_out[i][j])

Pred_CT = rfreg_opt.predict(X_out_fl)

### Open a text file to save the predictions ###

with open('Predictions_RFR_CT.txt', 'w') as f:
    f.write("Dopant, Predicted Charge transition level\n")
    
    for i in range(n_out):
        f.write(f"{dopant_out[i]}, {Pred_CT[i]:.4f}\n")

print("Predictions have been saved to 'Predictions_RFR_CT.txt'.")