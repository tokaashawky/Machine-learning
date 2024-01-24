# -*- coding: utf-8 -*-
"""
Data Cleaning^_^:
Bad data could be:
    1.Empty cells
    2.Data in wrong format
    3.Wrong data
    4.Duplicates
    5.Drop Unwanted Columns
"""
import pandas as pd
df = pd.read_csv("data-clean-set.csv")
# shape: how many instances (rows) and how many attributes (columns) 
print(df.shape)

# summarize the number of unique values in each column
print(df.nunique())

'''a-Empty Cells:
The data set contains some empty cells ("Date" in row 22, and "Calories" in row 18 and 28).
   ----------------------------------------------------------------------
    1.dropna() One way to deal with empty cells is to remove rows that contain empty cells.
        This is usually OK, since data sets can be very big,and removing a few rows will
        not have a big impact on the result.
        method returns a new DataFrame, and will not change the original
        If you want to change the original DataFrame, use the inplace = True argument.
        after method it return df without rows 18,22,28 ^_^
     
    2.fillna() method Replace Empty Values:
         ->with any specific value
         ->Replace Using Mean, Median, or Mode
'''
new_df = df.dropna()#give new copy of data &remove rows with null values
print(new_df)
#df.dropna(inplace = True)#remove rows of null values on the data itself
#print(df)

#Replace NULL values with the number 130 in the original data:
df.fillna(130, inplace = True)
#Replace Only For Specified Columns
df["Calories"].fillna(130, inplace = True)

#replace using mean value(the sum of all values divided by number of values).
x = df["Calories"].mean()
df["Calories"].fillna(x, inplace = True)

#replace using median value(the value in the middle,after you have sorted all values ascending).
x = df["Calories"].median()
df["Calories"].fillna(x, inplace = True)

##replace using mode value(the value in the middle,after you have sorted all values ascending).
x = df["Calories"].mode()[0]
df["Calories"].fillna(x, inplace = True)

'''b-Data in wrong format
The data set contains wrong format ("Date" in row 26).
   ----------------------------------------------------------------------
   1.to_datetime() method convert data for the right format
     we use argument format='mixed'->format will be inferred for each element individually
     
   2.str.strip('letters to be removed') column might contain unwanted characters.
     We can clean this column by stripping away those characters  
     
   3.if we have missing value in data attribute we could remove them 
     using dropna for data attribute->would drop raw 22 in original data.
'''
df['Date'] = pd.to_datetime(df['Date'],format='mixed')

#df['Last_Name'] = df['Last_Name'].str.strip("123._/")

df.dropna(subset=['Date'], inplace = True)

'''C-Wrong data
like if someone registered "199" instead of "1.99".
The data set contains wrong data ("Duration" in row 7).
   ----------------------------------------------------------------------
   1.replace with specific value one by one in small data but not for big data sets
   2.set some boundaries for legal values, and replace any values that are outside of the boundaries.
   3.remove the rows that contains wrong data using drop function.
'''
df.loc[7, 'Duration'] = 45

#note: The index of a DataFrame is a series of labels that identify each row.
for x in df.index:
  if df.loc[x, "Duration"] > 50:
    df.loc[x, "Duration"] = 70

for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.drop(x, inplace = True)

'''d-Duplicates
The data set contains duplicates (row 11 and 12).
   ----------------------------------------------------------------------
   -The duplicates() method returns a Boolean values for each row
    Returns True for every row that is a duplicate, otherwise False( line 12 would be true)
   -To remove duplicates, use the drop_duplicates() method to remove duplicate rows from our DataFrame.
       
'''
print(df.duplicated())
df.drop_duplicates(inplace = True)

'''e-Drop Unwanted Columns
 columns that are not relevant to your analysis. We can drop these columns using the drop() method.
'''
df = df.drop(columns="Date")





















