import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
             'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

sns.set(style='whitegrid', context='notebook')
#y = a0 + a1*xLSTAR + a2xRM
#split dataset into training set
#put two differnt values into X, same function call
X = df[['RM'] ].values
y = df['MEDV'].values
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

iris = load_iris()
X = iris.data
y = iris.target# -*- coding: utf-8 -*-
df = pd.DataFrame(iris.data, columns=iris.feature_names)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=123456)
rf.fit(X, y)
importances = rf.feature_importances_
print(importances)
print(iris.feature_names)
