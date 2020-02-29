
# Bank-Marketing-Data-Set

----

## Table of contents
* [About](#about)
* [Data Description](#data-description)
* [Methodology](#methodology)
* [Code Sample](#code-sample)
* [Result](#result)

## About

This exercise was the part of Machine Learning coursework at The University of Texas at Dallas.
The goal here was to understand various machine learning classification algorithms and also compare various functions/parameters for the algorithms (e.g. kernels in SVM, pruning in decision trees, etc.)

## Data Description

The dataset used here is from [UCI - Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 
The data is related with direct marketing campaigns of a Portuguese banking institution. 
The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
in order to access if the product (bank term deposit) would be (or not) subscribed. More info can be found in the [Jupyter Notebook](https://github.com/ShilpaPBhat/Bank-Marketing-Data-Set/blob/master/Bank-Marketing-Data.ipynb)

## Methodology

* Used various machine learning models including Linear and kernelized SVM, Logistic Regression, Decision Trees, Random Forest, KNN 
* To try improvising accuracy bagging, boostng was implemented

## Code Sample
Few snippet from the notebook:

````
# Linear SVM

from sklearn.svm import SVC
params = {'C': [10**i for i in range(-4, 5)] ,'max_iter':[1000,10000]}
grid = GridSearchCV(SVC(kernel = 'linear'), params, cv = 5)
grid.fit(data_train_x, data_train_y)
print('Best parameters:  {}'.format(grid.best_params_))
print('Best estimtor: {}'.format(grid.best_estimator_))
print('Best score: {}'.format(grid.best_score_))

svc_linear = SVC(kernel = 'linear', C = 0.1, degree = 3, random_state = 1)
svc_linear.fit(data_train_x, np.ravel(data_train_y))
pred_train = svc_linear.predict(data_train_x)
pred_test = svc_linear.predict(data_test_x)
print('Confusion matrix:\n', confusion_matrix(data_test_y, pred_test))
train_accuracy = metrics.accuracy_score(data_train_y, pred_train)
test_accuracy = metrics.accuracy_score(data_test_y, pred_test)
print(classification_report(data_test_y,pred_test))


````

## Result
The training and testing accuracy of various models: 

Model	| Train Accuracy | Test Accuracy
--- | --- | ---|
Linear SVC | 0.929977 |	0.891247
SVM RBF |	0.949817 |	0.885057
Decision Tree |	0.989330 |	0.849691
Logistic Regression |	0.935812 |	0.894783
Random Forest |	0.993831 | 0.883289
KNN |	1.000000 |	0.872679
SVM with Bagging |	0.924308 |	0.878868
SVM-RBF with Bagging |	0.931811 |	0.877100
Decision Tree with Bagging |	0.996999 |	0.881521
Logistic Reg with Bagging |	0.923641 |	0.879752
RF with Bagging |	0.982494 |	0.874447
KNN with Bagging	| 1.000000 |	0.882405
Gradient Boosting |	0.959153 |	0.866490

KNN overfits the data, bagging and boosting is not improving the accuracy.
Logistic Regression fits the best.
