"""steps to build linear regression"""

'''step1 import libraries '''
import pandas as pd 
import matplotlib.pyplot as plt

'''step2 read dataset '''
df= pd.read_csv("SalaryData.csv")
df= df.dropna()#would drop row 172,260

'''step3 spilt dataset into x and y
   1. x features would be column 0,1,2,3 and note( x must be matrix)
   2. y would be column 4and note y would be a vector
'''
x=df.iloc[: ,[4]]
y=df.iloc[: ,5]

'''step4 split dataset into train and test
   -we should import train_test_spilt method (which return 4 values) from small package model_selection which be 
    in the large package sklearn
   -the method parametes would be x and y and the precent it take for test data
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=.30)

'''step5 learning linear regression '''
from sklearn.linear_model import LinearRegression
#make object from class
regressor=LinearRegression()
#fit :Learn and estimate the parameters of the transformation
regressor.fit(x_train,y_train)

'''step6 test my model
  predict(x) method take the test feature and
  according to the model return the predicted value
'''
y_pred=regressor.predict(x_test)

'''step7 mean square error 
   it used to calculate error between the actual values
   and the predicted ones.
'''
from sklearn.metrics import mean_squared_error
error=mean_squared_error(y_test, y_pred)

'''visualization
1. for train 
2. for test
'''
#scatter digram make points at value of(x,y)
plt.scatter(x_train ,y_train,color="b")
#plot would make the line of model
plt.plot(x_train ,regressor.predict(x_train),color="r")
plt.title("SalaryData")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

plt.scatter(x_test,y_test,c='k')
plt.plot(x_train ,regressor.predict(x_train),color="r")
plt.show()

#to predict a new value
sal=regressor.predict([[5]])
print(sal)













