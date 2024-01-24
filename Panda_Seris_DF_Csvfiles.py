"""pandas: analyzing, cleaning, exploring, and manipulating data."""
#Pandas is usually imported under the pd alias.
import pandas as pd
#------------------------------------------------------------------------------

'''Pandas Series is like a column in a table.'''
# Seris: one-dimensional array holding data of any type
a = [1, 7, 2]
myvar1 = pd.Series(a)
print(myvar1)
# first value of the Series: 
print(myvar1[0])
#when print line index[0] default  or index ['a']


'''you can name your own labels which replace index from 0,1,2 => x,y,z'''
myvar1 = pd.Series(a, index = ["x", "y", "z"])

'''dictionary, when creating a Series The keys of the dictionary become the labels'''
calories = {"day1": 420, "day2": 380, "day3": 390}
myvar2 = pd.Series(calories)
#dict.keys      dict.values
#variable 2 in series() also be the index of data
print(myvar2)

'''we could choose only some columns to show'''
myvar3 = pd.Series(calories, index = ["day1", "day2"])
print(myvar3)

#------------------------------------------------------------------------------

'''Pandas DataFrame 2 dimensional array, or a table with rows and columns.'''
#DataFrame: 2 dimensional data structure
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45] }
df = pd.DataFrame(data)
#return one or more specified row(s)
print(df.loc[0])
#Return row 0 and 1
print(df.loc[[0, 1]])
#Add a list of names to give each row a name
df2 = pd.DataFrame(data, index = ["day1", "day2", "day3"])
#refer to the named index:
print(df2.loc["day2"])

#------------------------------------------------------------------------------

'''files Pandas can load them into a DataFrame CSV files (comma separated files).'''
df = pd.read_csv('iris.csv')
#to print the entire DataFrame.
print(df.to_string()) 
#print(df) 

'''your system's maximum rows ->give 60
  DataFrame contains more than 60 rows,
 the print(df) statement will return only the headers and the first and last 5 rows.
 we could increase the value like -> pd.options.display.max_rows = 9999
'''
print(pd.options.display.max_rows)

'''method returns the headers and a specified number of rows, starting from the top.'''
print(df.head())# default return 5 from the top
print(df.head(20))#return from top your number of rows you give

'''method returns the headers and a specified number of rows, starting from the bottom.'''
print(df.tail())# default return 5 from bottom
print(df.tail(10))#return from bottem your number of rows you give

''' gives you more information about the data set.
The result tells us there are 107 rows and 5 columns and
how many Non-Null values there are present in each column
you treat removing rows with empty values.
'''
print(df.info()) 

# class distribution:  number of instances (rows) that belong to each class.
print(dataset.groupby('class').size())


















