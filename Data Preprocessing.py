# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
df= pd.read_csv("C:\\Users\\Admin\\Documents\\Online dataset.csv")

# print the dataset
df

# shape of the dataset
df.shape

#summary statitics of the dataset
df.describe()

#types of data in dataset
df.dtypes

# checking null values by using isnull() and isna()
df.isnull()

df.isna()

# Handling Missing Values

from sklearn.impute import SimpleImputer

# 'np.nan' signifies that we are targeting missing values
# and the strategy we are choosing is replacing it with 'mean'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(df.iloc[:, 1:3])
df.iloc[:, 1:3] = imputer.transform(df.iloc[:, 1:3])  

# print the dataset
df

#Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# [0] signifies the index of the column we are 
#appliying the encoding on
df = pd.DataFrame(ct.fit_transform(df))
df

# In the last column, i.e. the purchased column, 
#the data is in binary form meaning that there are only 
#two outcomes either Yes or No. Therefore here we need 
#to perform Label Encoding.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:,-1] = le.fit_transform(df.iloc[:,-1])
# 'df.iloc[:,-1]' is used to select the column 
#that we need to be encoded
df

# Normalizing the dataset
#Feature scaling is bringing all of the features on the 
#dataset to the same scale, this is necessary while 
#training a machine learning model

#When we normalize the dataset it brings the value of 
#all the features between 0 and 1 so that all the columns 
#are in the same range
#Now to normalize the dataset we use MinMaxScaler class from the same ScikitLearn library.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df))
df

#Note: Feature scaling is not always necessary and 
#only required in some machine learning models.


# Splitting the dataset

#Before we begin training our model there is 
#one final step to go, which is splitting of the 
#testing and training dataset. 
#In machine learning, a larger part of the dataset 
#is used to train the model, and a small part is used 
#to test the trained model for finding out the accuracy 
#and the efficiency of the model.

#Now before we begin splitting the dataset we need to 
#separate the dependent and independent variables

#The last (purchased) column is the dependent variable 
#and the rest are independent variables, 
#so we’ll store the dependent variable in ‘y’ and 
#the independent variables in ‘X’.

#Another important part we need to remember is that 
#while training the model accepts data as arrays 
#so it is necessary that we convert the data to arrays. 
#We do that while separating the dependent and 
#independent variables by adding .values 
#while storing data in ‘X’ and ‘y’.


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# .values function coverts the data into arrays
print("Independent Variable\n")
print(X)
print("\nDependent Variable\n")
print(y)


#Now let’s split the dataset between Testing data and 
#Training data. To do this we’ll be using the 
#train_test_split class from the same ScikitLearn library.
#Deciding the ratio between testing data and 
#training data is up to us and depends on what 
#we are trying to achieve with our model. 
#In our case, we are going to go with an 
#80-20% split between the train-test data. 
#So 80% training and 20% testing data.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#'test_size=0.2' means 20% test data and 80% train data


# Here the test_size = 0.2 signifies that 
#we have selected 20% of data as testing data, 
#you can change that according to your choice.
#After this, the X_train and X_test variables 
#will have their respective data.

#Now our data is finally ready for training!!
