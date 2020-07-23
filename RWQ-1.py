# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:55:24 2020

@author: Abhishek
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import os
from pylab import rcParams
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def generate_model_report(y_actual, y_predicted):
    print("Accuracy = " , accuracy_score(y_actual, y_predicted))
    print("Precision = " ,precision_score(y_actual, y_predicted))
    print("Recall = " ,recall_score(y_actual, y_predicted))
    print("F1 Score = " ,f1_score(y_actual, y_predicted))
    pass

def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test,  y_pred_proba)
    auc = roc_auc_score(Y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass

os.chdir("C:/Users/Abhishek/Desktop/Data Science/IVY/Python Project/Red wine Quality")

df = pd.read_csv("winequality-red.csv")

df.info()

df['quality'].value_counts()

target = 'quality'

X = df.loc[:, df.columns!=target]
Y = df.loc[:, df.columns==target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.33, 
                                                    random_state=42)

ax = sns.countplot(x=target, data=df)
print(df[target].value_counts())

Y_train[target].value_counts()

clf = LogisticRegression().fit(X_train, Y_train)

Y_Test_Pred = clf.predict(X_test)

pd.crosstab(Y_Test_Pred, Y_test[target], rownames=['Predicted'], colnames=['Actual'])

generate_model_report(Y_test, Y_Test_Pred)

generate_auc_roc_curve(clf, X_test)

unique_classes = list(df[target].unique())
unique_classes

out_dict = {}
for classes in unique_classes:
    out_dict[classes] = df.shape[0]/((df.loc[df[target] == classes].shape[0])
                                     *len(unique_classes))
    
out_dict

print (X_train.shape, Y_train.shape)

clf = LogisticRegression(class_weight='balanced').fit(X_train, Y_train)

from sklearn.utils import class_weight
class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train[target])

Y_Test_Pred = clf.predict(X_test)
pd.crosstab(Y_Test_Pred, Y_test[target], rownames=['Predicted'], colnames=['Actual'])

generate_model_report(Y_test, Y_Test_Pred)
generate_auc_roc_curve(clf, X_test)

weights = np.linspace(0.05, 0.95, 20)
gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=5
)

grid_result = gsc.fit(X_train, Y_train)
print("Best parameters : %s" % grid_result.best_params_)

data_out = pd.DataFrame({'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
data_out.plot(x='weight')

clf = LogisticRegression(**grid_result.best_params_).fit(X_train, Y_train)
Y_Test_Pred = clf.predict(X_test)

pd.crosstab(Y_Test_Pred, Y_test[target], rownames=['Predicted'], colnames=['Actual'])

generate_model_report(Y_test, Y_Test_Pred)
generate_auc_roc_curve(clf, X_test)
