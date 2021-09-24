# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:37:14 2021

@author: simon
"""

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv('bike_data.csv', delimiter=',', encoding='ISO-8859-1', usecols=[1,2,3,4,5,6,7,8,9,10,11])
df['Seasons'] = df['Seasons'].replace('Winter','1')
df['Seasons'] = df['Seasons'].replace('Spring','2')
df['Seasons'] = df['Seasons'].replace('Summer','3')
df['Seasons'] = df['Seasons'].replace('Autumn','4')
X = df.iloc[:,1:]
#X = pd.get_dummies(df, columns=['Hour', 'Seasons'])

y = df['Rented Bike Count']
y_cat = pd.cut(df['Rented Bike Count'], bins = [0,249.9999,800,1e6], labels = ['LOW', 'AVERAGE', 'BUSY' ], include_lowest=True)


#X = sklearn.preprocessing.normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

pipe = make_pipeline(sklearn.preprocessing.Normalizer(), LinearRegression())
scores = cross_val_score(pipe, X_train, y_train, cv=10)
print(np.mean(scores))

pipe.fit(X_train, y_train)
print(pipe.score(X_train, y_train))
print(pipe.score(X_test, y_test))


pipe_ridge =  make_pipeline(sklearn.preprocessing.Normalizer(), Ridge())
param_grid = {'ridge__alpha':np.logspace(-5, 3)}

grid = GridSearchCV(pipe_ridge, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
print(grid.best_params_)




results = pd.DataFrame(grid.cv_results_)
results.plot('params', 'mean_train_score')
results.plot('params', 'mean_test_score', ax=plt.gca())
plt.fill_between(results.param_alpha.astype(np.float),
                 results['mean_train_score'] + results['std_train_score'],
                 results['mean_train_score'] - results['std_train_score'], alpha=0.2)
plt.fill_between(results.param_alpha.astype(np.float),
                 results['mean_test_score'] + results['std_test_score'],
                 results['mean_test_score'] - results['std_test_score'], alpha=0.2)
plt.legend()
plt.xscale("log")


#Lasso
pipe_lasso =  make_pipeline(sklearn.preprocessing.Normalizer(), Lasso())
param_grid = {'lasso__alpha':np.logspace(-5, 3)}
grid_Lasso = GridSearchCV(pipe_lasso, param_grid, return_train_score = True, cv=10)
grid_Lasso.fit(X_train, y_train)

print("best mean cross-validation score: {:.3f}".format(grid_Lasso.best_score_))
print("best parameters: {}".format(grid_Lasso.best_params_))
print("test-set score: {:.3f}".format(grid_Lasso.score(X_test, y_test)))
x = grid_Lasso.best_estimator_.named_steps['lasso'].coef_
x




'''
#PERFORMANCE ISSUES
grid = GridSearchCV(Lasso(), param_grid,  return_train_score=True, cv=10)
grid.fit(X_train, y_train)

results = pd.DataFrame(grid.cv_results_)
results.plot('param_alpha', 'mean_train_score')
results.plot('param_alpha', 'mean_test_score', ax=plt.gca())
plt.fill_between(results.param_alpha.astype(np.float),
                 results['mean_train_score'] + results['std_train_score'],
                 results['mean_train_score'] - results['std_train_score'], alpha=0.2)
plt.fill_between(results.param_alpha.astype(np.float),
                 results['mean_test_score'] + results['std_test_score'],
                 results['mean_test_score'] - results['std_test_score'], alpha=0.2)
plt.legend()
plt.xscale("log")


lr = LinearRegression().fit(X_train, y_train)
plt.scatter(range(X_train.shape[1]), lr.coef_, c=np.sign(lr.coef_), cmap="bwr_r")

ridge = grid.best_estimator_
plt.scatter(range(X_train.shape[1]), ridge.coef_, c=np.sign(ridge.coef_), cmap="bwr_r")
'''




X_train, X_test, y_train, y_test = train_test_split(X, y_cat, random_state=42, train_size=0.8)

neighbors = range(1, 30, 2)

training_scores = []
test_scores = []


for n_neighbors in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    training_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
plt.figure()
plt.plot(neighbors, training_scores, label="training scores")
plt.plot(neighbors, test_scores, label="test scores")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()


