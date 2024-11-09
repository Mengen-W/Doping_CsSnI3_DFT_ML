# DopingCsSnI3_DFT_ML
This project combines DFT and ML to predict the formation energy of defect charge states, specifically neutral (q=0) and q=+1, as well as the charge transition level (CT) of +1/0 for substitutional defects at cation sites in CsSnI3.

The DFT calculations were used to generate a dataset of formation energies and charge transitions for training the model, which is stored in the "Formation energy" folder. The formation energies for substitutional defects on Sn and Cs sites, as well as intrinsic defects, including the formation energy diagram, are also collected in this folder.

To train the models, feature properties are required. However, to select the most predictive features and remove redundant ones, we used the Pearson correlation coefficient to eliminate strongly correlated features. The Pearson correlation results and the final list of features are stored in the "Pearson correlation" folder under different chemical potential conditions.

To find the most predictive models, five regression models were employed in this work: Linear Regression (LR), LASSO, Gaussian Process Regression (GPR), Kernel Ridge Regression (KRR), and Random Forest Regression (RFR). These models are collected in the "Regression models" folder.
