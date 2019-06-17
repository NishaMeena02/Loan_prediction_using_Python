#libraries
import numpy as np
import pandas as pd

#import datasets
dataset1=pd.read_csv('trainset_load_pred.csv')
X_train=dataset1.iloc[:, 1:12]
y_train=dataset1.iloc[:,12]    

dataset2=pd.read_csv('testset_loan_pred.csv')
X_test=dataset2.iloc[:, 1:12]

#exploratory data analysis fro continuous variables
X_train['ApplicantIncome'].hist(bins=50)
X_train.boxplot(column ='ApplicantIncome') # lots of outliers are there
X_train.boxplot(column ='ApplicantIncome', by='Education')
#we will see that there is no substantial difference btw the mean income of graduates and non graduates.
#but there are higher no. of graduates,with very high income,which are appereaing to be outliers.
X_train['LoanAmount'].hist(bins=50)
X_train.boxplot(column ='LoanAmount')
#lots of outliers are there
X_train.boxplot(column ='LoanAmount', by='Gender')
#loanamount has missing as well as extreme values 
#applicant income has few extreme values

#feature transformation of continuous variables(taking care of outliers)
X_train['LoanAmount']=np.log(X_train['LoanAmount'])
X_test['LoanAmount']=np.log(X_test['LoanAmount'])

X_train['TotalIncome']=X_train['ApplicantIncome']+X_train['CoapplicantIncome']
X_test['TotalIncome']=X_test['ApplicantIncome']+X_test['CoapplicantIncome']

X_train['TotalIncome']=np.log(X_train['TotalIncome'])
X_test['TotalIncome']=np.log(X_test['TotalIncome'])

X_train.drop(["ApplicantIncome","CoapplicantIncome"],axis=1,inplace=True)
X_test.drop(["ApplicantIncome","CoapplicantIncome"],axis=1,inplace=True)

#missing value treatment in training set
X_train.apply(lambda x: sum(x.isnull()),axis=0) #to check missing values in each column

#filling missing values in self employed
X_train['Self_Employed'].value_counts()
X_train['Self_Employed'].fillna('No',inplace=True) #impute missing value with No (median)

# imputing missing value of loan amount based on Self_Employed and Education using pivot approach 
table = X_train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
print(table)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]

# Replace missing values
X_train['LoanAmount'].fillna(X_train[X_train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

X_train['Married'].value_counts()
X_train['Married'].fillna('Yes',inplace=True) #impute missing value most frequent

X_train['Dependents'].value_counts()
X_train['Dependents'].fillna('0',inplace=True) #impute missing value with most frequent


X_train['Gender'].value_counts()
X_train['Gender'].fillna('Male',inplace=True) #impute missing value with most frequent

X_train['Credit_History'].value_counts()
X_train['Credit_History'].fillna(1.0,inplace=True) #impute missing value with most frequent

X_train['Loan_Amount_Term'].value_counts()
X_train['Loan_Amount_Term'].fillna(360.0,inplace=True) #impute missing value with most frequent

X_train.apply(lambda x: sum(x.isnull()),axis=0) #check if now there is any missing values

#missing value treatment in test set
X_test.apply(lambda x: sum(x.isnull()),axis=0) #to check missing values in each column

#filling missing values in self employed
X_test['Self_Employed'].value_counts()
X_test['Self_Employed'].fillna('No',inplace=True) #impute missing value with No (median)

# imputing missing value of loan amount based on Self_Employed and Education using pivot approach 
table = X_test.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
print(table)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]

# Replace missing values
X_test['LoanAmount'].fillna(X_test[X_test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

X_test['Married'].value_counts()
X_test['Married'].fillna('Yes',inplace=True) #impute missing value most frequent

X_test['Dependents'].value_counts()
X_test['Dependents'].fillna('0',inplace=True) #impute missing value with most frequent


X_test['Gender'].value_counts()
X_test['Gender'].fillna('Male',inplace=True) #impute missing value with most frequent

X_test['Credit_History'].value_counts()
X_test['Credit_History'].fillna(1.0,inplace=True) #impute missing value with most frequent

X_test['Loan_Amount_Term'].value_counts()
X_test['Loan_Amount_Term'].fillna(360.0,inplace=True) #impute missing value with most frequent

X_test.apply(lambda x: sum(x.isnull()),axis=0) #check if now there is any missing values

# First we need to encode levels in the categorical variables to numeric using LabelEncoder function
#encoding categorical data
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    X_train[i] = le.fit_transform(X_train[i])
X_train.dtypes 
for j in var_mod:
    X_test[j] = le.fit_transform(X_test[j])
X_test.dtypes 

#fitiing random forest on training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=300,n_jobs=-1,oob_score=True,max_depth=7,
                                  criterion ='entropy',random_state=50,max_features=1 ,min_samples_split=100)
classifier.fit(X_train,y_train)

#applying Kfold cross validation on training set
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=150)
accuracies.mean()

#predict test set results
y_test=classifier.predict(X_test)
y_test=pd.Series(y_test)

#fitiing random forest on training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=300 ,n_jobs=-1,oob_score=True,max_depth=7,
                                  criterion ='entropy',random_state=50,max_features="auto",min_samples_leaf=50)
classifier.fit(X_test,y_test)

#applying Kfold cross validation on test set,
#here i divide the training set into k folds and then calculate accuracies and then take there average
from sklearn.model_selection import cross_val_score
accuracies_test=cross_val_score(estimator=classifier,X=X_test,y=y_test,cv=20)
accuracies_test.mean()

#to convert series into csv 
sample=pd.Series.to_csv(y_test,index=None)
