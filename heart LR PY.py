# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:11:13 2020

@author: A MURALI
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('framingham.csv')

df = df.dropna()
df.isna().any
df.isna().count

np.isnan(df)

df.head()


df.keys()


import seaborn as sns
sns.pairplot(df, hue = 'TenYearCHD', vars = ['age', 'totChol', 'sysBP','diaBP', 'BMI', 'heartRate', 'glucose'] )
np.corrcoef(df)
sns.heatmap(np.corrcoef(df))

sns.countplot(df['TenYearCHD'])

df.keys()

plt.scatterplot(x = 'male', y = 'age', hue = 'TenYearCHD', data = df)

plt.figure(figsize = (20,10))
sns.heatmap(df.corr(), annot = True)

df['TenYearCHD'].value_counts()

X = df.drop(['TenYearCHD'], axis = 1)
X.head()
X.keys()

y = df['TenYearCHD']

X1 = pd.get_dummies(X, columns = ['education'])
X1.keys()
X = X1.drop(['education_4.0'], axis = 1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.35, random_state = 100)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred

y_predi = np.where(y_pred > 0.5, 1, 0)
print(y_predi)


from sklearn.metrics import classification_report, confusion_matrix 
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)
cm



from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
acc


