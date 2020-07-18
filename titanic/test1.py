# Author: Louis Chen
import numpy as np
import pandas as pd
import re as re

train = pd.read_csv('D:/Kaggle/titanic/train.csv', header=0, dtype={'Age': np.float64})
test = pd.read_csv('D:/Kaggle/titanic/test.csv', header=0, dtype={'Age': np.float64})
full_data = [train, test]
# print(train.info())
# print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
'''
# drop the unnecessary columns
df_train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
df_test.drop( ['Name','Ticket','Cabin'],axis=1,inplace=True)

# encode the categorical(no details)
sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)
df_train = pd.concat([df_train,sex,embark],axis=1)
df_train.drop(['Sex','Embarked'],axis=1,inplace=True)

sex = pd.get_dummies(df_test['Sex'],drop_first=True)
embark = pd.get_dummies(df_test['Embarked'],drop_first=True)
df_test = pd.concat([df_test,sex,embark],axis=1)
df_test.drop(['Sex','Embarked'],axis=1,inplace=True)
# fill the nan
df_train.fillna(df_train.mean(),inplace=True)
df_test.fillna(df_test.mean(),inplace=True)

# normalization operator in sklearn
Scaler1 = StandardScaler()
Scaler2 = StandardScaler()

# feature
train_columns = df_train.columns
test_columns  = df_test.columns

# normalization
df_train = pd.DataFrame(Scaler1.fit_transform(df_train))
df_test  = pd.DataFrame(Scaler2.fit_transform(df_test))

df_train.columns = train_columns
df_test.columns  = test_columns
X_test = df_test.iloc[:,1:].values

# location, get feature list and target
features = df_train.iloc[:,2:].columns.tolist()
target   = df_train.loc[:, 'Survived'].name
# get data without feature name and 'passengerID','Survived', and get labels
X_train = df_train.iloc[:,2:].values
y_train = df_train.loc[:, 'Survived'].values'''
