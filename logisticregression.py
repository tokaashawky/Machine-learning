"""
Logistic Regression ^___^
"""
'''step1 import libraries '''
import pandas as pd

'''step2 read dataset and check if it cleaning data'''
df=pd.read_csv("iris.csv")
df["150"].fillna(df["150"].mean(), inplace = True)
df["4"].fillna(df["4"].mean(), inplace = True)
df["setosa"].fillna(df["setosa"].mean(), inplace = True)
df["versicolor"].fillna(df["versicolor"].mean(), inplace = True)

'''step3 divide data into x and y'''
x=df.iloc[: ,[0,1,2,3]].values
y=df.iloc[:,4].values

'''step8 scaling(=normalization)
   make the data all in same range with keep information exist 
'''
from sklearn.preprocessing import StandardScaler
#object from normalizer standscaler
ss=StandardScaler()
#fit_transform() learn parameter and apply the transformation to data
x=ss.fit_transform(x)

'''step4 divide data int train and test'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

'''step5 train the mode'''
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train, y_train)

'''step6 testing'''
y_pred=classifier.predict(x_test)

'''syep7 evaluation using confusion matrix'''
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred,)
#we wind new problem data isn't in same range so return and normalize it


'''step9 using confusion matrix we calculate them 
   notice we have three classes so we use one of them (macro,micro,weighted)
'''
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
precision=precision_score(y_test, y_pred,average='macro')*100
accuracy=accuracy_score(y_test, y_pred)*100
recall=recall_score(y_test, y_pred,average='macro')*100
f1=f1_score(y_test, y_pred,average='macro')*100