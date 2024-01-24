# -*- coding: utf-8 -*-
"""
Decision Tree ^_^
all data has to be numerical We have to convert the non numerical columns
Pandas has a map() method that takes a dictionary with information on how to convert the values.
"""

import pandas as pd 
df=pd.read_csv("DT_data.txt")

#1.Change string values into numerical values using map() method:
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nat']=df['Nat'].map(d)
d1= {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d1)
print(df)

#2.separate the feature columns from the target column(x features and y terget).
X = df[['Age', 'Experience', 'Rank', 'Nat']]
y = df['Go']
print(X)
print(y)

#3.create the actual decision tree,Start by importing the modules
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#from  DecisionTreeClassifier class make object called dtree
dtree = DecisionTreeClassifier()

#4.fit it with our details
dtree = dtree.fit(X, y)
tree.plot_tree(dtree, feature_names=['Age', 'Experience', 'Rank', 'Nat'])