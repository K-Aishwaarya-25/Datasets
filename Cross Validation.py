
import pandas as pd
df=pd.read_csv('C:\\Users\Admin\Downloads\heart.csv')
df.head()

###  Independent And dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X.head()


y

y.value_counts()
# 1 represent heart attack
# 0 represent no heart attack

#HoldOut Validation Approach- Train And Test Split

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)

#K Fold Cross Validation
from sklearn.model_selection import KFold
model=DecisionTreeClassifier()
kfold_validation=KFold(10)
import numpy as np
from sklearn.model_selection import cross_val_score
results=cross_val_score(model,X,y,cv=kfold_validation)
print(results)
print(np.mean(results)*100)
print(np.std(results)*100)


#Leave One Out Cross Validation(LOOCV)
# LOOCV is the case of Cross-Validation where just a 
#single observation is held out for validation.
#The model is evaluated for every held out observation. 
#The final result is then calculated by taking the mean of 
#all the individual evaluations.

# It can be computationally expensive to use LOOCV
# The other problem with LOOCV is that it can be subject to 
#high variance or overfitting as we are feeding the model 
#almost all the training data to learn and just a single observation to evaluate.

# These problems can be addressed by using another validation 
#technique known as k-Fold Cross-Validation.

from sklearn.model_selection import LeaveOneOut
model=DecisionTreeClassifier()
leave_validation=LeaveOneOut()
results=cross_val_score(model,X,y,cv=leave_validation)
results
print(np.mean(results)*100)
print(np.std(results)*100)
