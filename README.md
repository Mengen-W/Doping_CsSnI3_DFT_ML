# Doping CsSnI3 DFT ML
This is the documentation for the work that combines density functional theory (DFT) and machine learning (ML) algorithms to predict the formation energy and charge transition levels in of the substitutional defects in cesium tin iodide.

The DFT calculations were used to generate a dataset of formation energies at two charge states (q=+1 and q=0) and charge transition levels, which is stored in "Formation energy". The raw data to calculate defect formation energies and the code to generate formation energy diagrams are also included in this folder.

The Pearson correlation analysis of features and target properties are included in "Pearson correlation".

The ML code to train and predict formation energies and charge transition levels are provided in "Regression models", including Linear Regression (LR), Gaussian Process Regression (GPR), Kernel Ridge Regression (KRR), and Random Forest Regression (RFR).

## License
This code is made available under the MIT license.

## Requirements
The ML training and prediction codes are compatible with Python 3 and the following open source Python packages should be installed:

* numpy

* matplotlib

* pandas

* scikit-learn

## Contact
Mengen Wang, SUNY Binghamton (mengenwang@binghamton.edu)

Chadawan Khamdang, SUNY Binghamton 





