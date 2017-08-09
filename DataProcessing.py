# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 00:40:58 2017

@author: Aditya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('F://UdemyML//Machine Learning A-Z Template Folder//Part 1 - Data Preprocessing//Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values 

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

# Encoding Categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X=LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])
oneHotencoder_x=OneHotEncoder(categorical_features=[0])
X=oneHotencoder_x.fit_transform(X).toarray()
labelEncoder_Y=LabelEncoder()
y=labelEncoder_Y.fit_transform(y)

#splitting the data into train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_x.fit_transform(X)
X_test=sc_x.transform(X)