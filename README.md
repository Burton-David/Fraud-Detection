# Fraud-Detection
Compilation of useful code for fraud detection

## compare_algos.py
This program loads a dataset of fraud data, splits it into training and test sets, scales the data using the StandardScaler class from scikit-learn, and then trains and evaluates three different algorithms: logistic regression, decision tree, and random forest. It uses the accuracy_score, f1_score, and roc_auc_score functions from scikit-learn to compute various performance metrics, and prints the results to the console.

You can modify this program to use different algorithms or performance metrics, or to use a different dataset. You can also use this program as a starting point to tune the hyperparameters of the algorithms and improve their performance.

## auto_tune.py
This script loads a dataset of fraud data, splits it into training and test sets, scales the data using the StandardScaler class from scikit-learn, and then trains a random forest classifier using grid search to tune the hyperparameters. It uses the GridSearchCV class from scikit-learn to perform the grid search, and the accuracy_score function to evaluate the performance of the model.

You can modify this script to use different algorithms or performance metrics, or to use a different dataset. You can also modify the hyperparameter grid to include different values or a different range of values to search over.

## version1.py
This script defines a simple fraud detection model using a fully-connected neural network with one hidden layer. The model is trained on fake data generated using the torch.randn and torch.rand functions, and the loss is computed using the binary cross-entropy loss (BCELoss). The model is then tested on a separate set of fake data, and the accuracy is printed to the console.  

Very basic, do not use in prod.