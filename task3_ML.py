# Import necessary libraries
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

###############################################################################
# Load the dataset from a CSV file
df = pd.read_csv('heart_disease_uci.csv')  # ==> num column to predict

###############################################################################
# Display basic information about the dataset
print(df.head())  # Display the first 5 rows of the dataset
print(df[df.duplicated()])  # Display duplicated rows, if any
print(df.nunique())  # Display the number of unique values in each column
print(df.describe())  # Display summary statistics for numerical columns
print(df.isnull().sum())  # Display the count of missing values for each column

###############################################################################
# Data Cleaning and Handling Missing Values
df = df.drop(columns=['id', 'ca', 'thal', 'slope'])  # Drop unnecessary columns
df["trestbps"].fillna(df["trestbps"].mean(), inplace=True)  # Fill missing values in 'trestbps' with mean
df["chol"].fillna(df["chol"].mean(), inplace=True)  # Fill missing values in 'chol' with mean
df["thalch"].fillna(df["thalch"].mean(), inplace=True)  # Fill missing values in 'thalch' with mean
df["oldpeak"].fillna(df["oldpeak"].mean(), inplace=True)  # Fill missing values in 'oldpeak' with mean
df["exang"].fillna(df["exang"].mode()[0], inplace=True)  # Fill missing values in 'exang' with mode (most frequent value)
df["fbs"].fillna(df["fbs"].mode()[0], inplace=True)  # Fill missing values in 'fbs' with mode
df["restecg"].fillna(df["restecg"].mode()[0], inplace=True)  # Fill missing values in 'restecg' with mode

###############################################################################
# Display information after handling missing values
print(df.isnull().sum())  # Confirm that there are no more missing values
print(df.info())  # Display concise summary of the dataframe
print(df.nunique())  # Display the updated number of unique values in each column

###############################################################################
# Outlier Removal
columns_to_check = ['chol', 'trestbps', 'thalch']
for column in columns_to_check:
    Q1 = df[column].quantile(0.25)  # Calculate the first quartile
    Q3 = df[column].quantile(0.75)  # Calculate the third quartile
    IQR = Q3 - Q1  # Calculate the Interquartile Range (IQR)
    lower_bound = Q1 - 1.5 * IQR  # Calculate the lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Calculate the upper bound for outliers
    df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]  # Keep data within the bounds

###############################################################################
# Data Visualization - Boxplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(data=df, orient='h')  # Create a horizontal boxplot for each numerical column
plt.title('Boxplot of Value')  # Set the title of the plot
plt.show()  # Display the plot

###############################################################################
# Identify columns with data type 'object'
object_col = df.select_dtypes(include=["object"]).columns
# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, columns=object_col)
# Replace boolean values with 1 for True and 0 for False
df = df.replace({True: 1, False: 0})
# Display information about the updated dataframe
print(df.info())
# Define features (x) and target (y)
x = df.drop('num', axis=1)
y = (df['num'] > 0).astype(int)  # Convert 'num' to binary (1 for positive, 0 for non-positive)
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)

###############################################################################
# Import the StandardScaler class from the sklearn.preprocessing module
from sklearn.preprocessing import StandardScaler
# Create an instance of the StandardScaler class
scalestd = StandardScaler()
# Fit the scaler to the training data and simultaneously transform it
x_train_scaled = scalestd.fit_transform(x_train)
# Transform the test data using the same scaler fitted on the training data
x_test_scaled = scalestd.transform(x_test)

###############################################################################
# Import the RandomForestClassifier class from the sklearn.ensemble module
from sklearn.ensemble import RandomForestClassifier
# Import accuracy_score and confusion_matrix from sklearn.metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Create an instance of the RandomForestClassifier class with specified parameters
RandomForestClassifieer = RandomForestClassifier(n_estimators=10, random_state=42)
# Train the Random Forest Classifier using the scaled training data and corresponding labels
RandomForestClassifieer.fit(x_train_scaled, y_train)
# Make predictions on the scaled test data
y_pred = RandomForestClassifieer.predict(x_test_scaled)
# Calculate the confusion matrix for the predictions and actual labels
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Calculate and print the accuracy of the Random Forest Classifier
print(f"The accuracy of the RandomForestClassifieer is: {accuracy_score(y_test, y_pred) * 100}")

###############################################################################
# Import the KNeighborsClassifier class from the sklearn.neighbors module
from sklearn.neighbors import KNeighborsClassifier
# Create an instance of the KNeighborsClassifier class with specified parameters
KNeighborsClassifieer = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# Train the K-NN Classifier using the scaled training data and corresponding labels
KNeighborsClassifieer.fit(x_train_scaled, y_train)
# Make predictions on the scaled test data
y_pred = KNeighborsClassifieer.predict(x_test_scaled)
# Calculate the confusion matrix for the predictions and actual labels
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Calculate and print the accuracy of the K-NN Classifier
print(f"The accuracy of the KNeighborsClassifieer is: {accuracy_score(y_test, y_pred) * 100}")

###############################################################################
# Import the DecisionTreeClassifier class from the sklearn.tree module
from sklearn.tree import DecisionTreeClassifier
# Create an instance of the DecisionTreeClassifier class with a specified maximum depth
DecisionTreeClassifieer = DecisionTreeClassifier(max_depth=3)
# Train the Decision Tree Classifier using the scaled training data and corresponding labels
DecisionTreeClassifieer.fit(x_train_scaled, y_train)
# Make predictions on the scaled test data
y_pred = DecisionTreeClassifieer.predict(x_test_scaled)
# Calculate the confusion matrix for the predictions and actual labels
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Calculate and print the accuracy of the Decision Tree Classifier
print(f"The accuracy of the DecisionTreeClassifier is: {accuracy_score(y_test, y_pred) * 100}")

###############################################################################
# Import the GaussianNB class from the sklearn.naive_bayes module
from sklearn.naive_bayes import GaussianNB
# Create an instance of the Gaussian Naive Bayes classifier
GaussianNBB = GaussianNB()
# Train the Gaussian Naive Bayes classifier using the scaled training data and corresponding labels
GaussianNBB.fit(x_train_scaled, y_train)
# Make predictions on the scaled test data
predicted = GaussianNBB.predict(x_test_scaled)
# Calculate the confusion matrix for the predictions and actual labels
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted)
print(cm)
# Calculate and print the accuracy of the Gaussian Naive Bayes Classifier
print(f"The accuracy of the GaussianNB Classifier is: {accuracy_score(y_test, predicted) * 100}")

###############################################################################
# Import the Support Vector Classifier (SVC) class from the sklearn.svm module
from sklearn.svm import SVC
# SVM with linear kernel
svm_linear = SVC(kernel='linear')
# Train the Support Vector Machine (SVM) with linear kernel using the scaled training data and corresponding labels
svm_linear.fit(x_train_scaled, y_train)

# SVM with RBF kernel
# Parameters: C is the regularization parameter, gamma is the kernel coefficient
svm_rbf = SVC(C=100, kernel="rbf", gamma=0.1)
# Train the Support Vector Machine (SVM) with RBF kernel using the scaled training data and corresponding labels
svm_rbf.fit(x_train_scaled, y_train)

# Evaluate SVM with linear kernel
y_pred_linear = svm_linear.predict(x_test_scaled)
# Evaluate SVM with RBF kernel
y_pred_rbf = svm_rbf.predict(x_test_scaled)
# Calculate the confusion matrix for the predictions and actual labels using SVM with RBF kernel
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rbf)
print(cm)
# Calculate and print the accuracy of the SVM with RBF kernel
print(f"The accuracy of the SVM with RBF kernel is: {accuracy_score(y_test, y_pred_rbf) * 100}")

###############################################################################
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Evaluate model1
y_pred1 = RandomForestClassifieer.predict(x_test_scaled)
conf_matrix1 = confusion_matrix(y_test, y_pred1)
accuracy1 = accuracy_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
f1score1 = f1_score(y_test, y_pred1)

# Evaluate model2
y_pred2 = KNeighborsClassifieer.predict(x_test_scaled)
conf_matrix2 = confusion_matrix(y_test, y_pred2)
accuracy2 = accuracy_score(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)
f1score2 = f1_score(y_test, y_pred2)

# Evaluate model3
y_pred3 = DecisionTreeClassifieer.predict(x_test_scaled)
conf_matrix3 = confusion_matrix(y_test, y_pred3)
accuracy3 = accuracy_score(y_test, y_pred3)
precision3 = precision_score(y_test, y_pred3)
recall3 = recall_score(y_test, y_pred3)
f1score3 = f1_score(y_test, y_pred3)

# Evaluate model4
y_pred4 = GaussianNBB.predict(x_test_scaled)
conf_matrix4 = confusion_matrix(y_test, y_pred4)
accuracy4 = accuracy_score(y_test, y_pred4)
precision4 = precision_score(y_test, y_pred4)
recall4 = recall_score(y_test, y_pred4)
f1score4 = f1_score(y_test, y_pred4)

# Evaluate model5
y_pred5 = svm_linear.predict(x_test_scaled)
conf_matrix5 = confusion_matrix(y_test, y_pred5)
accuracy5 = accuracy_score(y_test, y_pred5)
precision5 = precision_score(y_test, y_pred5)
recall5 = recall_score(y_test, y_pred5)
f1score5 = f1_score(y_test, y_pred5)

# Evaluate model6
y_pred6 = svm_rbf.predict(x_test_scaled)
conf_matrix6 = confusion_matrix(y_test, y_pred6)
accuracy6 = accuracy_score(y_test, y_pred6)
precision6 = precision_score(y_test, y_pred6)
recall6 = recall_score(y_test, y_pred6)
f1score6 = f1_score(y_test, y_pred6)

# Compare accuracies to find the best model
accuracies = [accuracy1*100,accuracy2*100, accuracy3*100,accuracy4*100,accuracy5*100,accuracy6*100]
print(accuracies)
best_model_index = accuracies.index(max(accuracies))
if best_model_index == 0 :
  best_model_name = "RandomForestClassifieer" 
elif best_model_index == 1 :
    best_model_name = "KNeighborsClassifieer"
elif best_model_index == 2 :
  best_model_name= "DecisionTreeClassifieer" 
elif best_model_index == 3 :     
  best_model_name="GaussianNBB" 
elif best_model_index == 4 :
  best_model_name ="svm_linear" 
else :
  best_model_name = "svm_rbf"

print(f"The best accuracy is the model: {best_model_name}")




