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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, DotProduct, ExpSineSquared

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
kernels = {
    "ExpSineSquared": [ExpSineSquared(l, p) for l in np.logspace(-2, 4, 10) for p in np.logspace(0, 2, 10)],
    "RBF": [RBF(length_scale) for length_scale in np.logspace(-2, 4, 10)],
    "RationalQuadratic": [RationalQuadratic(length_scale=l, alpha=a) for l in np.logspace(-2, 4, 10) for a in np.logspace(0, 2, 10)],
    "Matern": [Matern(length_scale=l, nu=nu) for l in np.logspace(-2, 4, 10) for nu in [0, 2, 10]],
    "DotProduct": [DotProduct(sigma_0=sigma) for sigma in np.logspace(-2, 4, 10)],
}

kernel_list = [kernel for sublist in kernels.values() for kernel in sublist]

param_grid = {"alpha": [1e0, 1e-1, 1e-2],
              "kernel": kernel_list,
              "n_restarts_optimizer": [50, 100, 200, 250]
}
'''
best_kernel = Matern(length_scale=0.01, nu=2)
best_alpha = 0.1
best_n_restarts_optimizer = 200

param_grid = {
    "alpha": [best_alpha],
    "kernel": [best_kernel],
    "n_restarts_optimizer": [best_n_restarts_optimizer]
}

### Train the model ###

gpreg_opt = RandomizedSearchCV(GaussianProcessRegressor(), param_distributions=param_grid, cv=5)
gpreg_opt.fit(X_train, CT_train)

Pred_train = gpreg_opt.predict(X_train)
Pred_test  = gpreg_opt.predict(X_test)

print('Best estimator by RandomizedSearchCV:', gpreg_opt.best_estimator_)

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

Pred_CT = gpreg_opt.predict(X_out_fl)

### Open a text file to save the predictions ###

with open('Predictions_GPR_CT.txt', 'w') as f:
    f.write("Dopant, Predicted Charge transition level\n")
    
    for i in range(n_out):
        f.write(f"{dopant_out[i]}, {Pred_CT[i]:.4f}\n")

print("Predictions have been saved to 'Predictions_GPR_CT.txt'.")