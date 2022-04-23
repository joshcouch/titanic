"""
Functions to import csv data, clean and then output as pytorch tensors:
X_train, X_test, y_train, y_test
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

# Features used
features = ['Survived','Cherbourg','Queenstown']

# Import and prepare data
train_data_df = pd.read_csv(train_csv_path)

def sex_to_binary(df):
    df['Binary Sex'] = df['Sex'] == 'male'
    df['Binary Sex'] = df['Binary Sex'].astype(int)
    return df

def categorise_age(df):
    df['Child'] = df['Age'] < 15
    df['Child'] = df['Child'].astype(int)
    df['Teen'] = df['Age'] < 18
    df['Teen'] = df['Teen'].astype(int) - df['Child']
    df['Young Adult'] = df['Age'] <= 30
    df['Young Adult'] = df['Young Adult'].astype(int) - df['Teen'] - df['Child']
    return df

def square_Pclass(df):
    df['Pclass Squared'] = df['Pclass']
    return df

def categorise_embarkation_port(df):
    df['Cherbourg'] = df['Embarked'] == 'C'
    df['Cherbourg'] = df['Cherbourg'].astype(int)
    df['Queenstown'] = df['Embarked'] == 'Q'
    df['Queenstown'] = df['Queenstown'].astype(int)
    df['Southampton'] = df['Embarked'] == 'S'
    df['Southampton'] = df['Southampton'].astype(int)
    return df

def clean_df(df):
    df = sex_to_binary(df)
    df = categorise_age(df)
    df = square_Pclass(df)
    df = categorise_embarkation_port(df)
    return df



train_data_df = clean_df(train_data_df)

train_data_np = train_data_df[features].to_numpy()

X, y = train_data_np[:,1:], train_data_np[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# remove NaN
indices_of_not_nan = ~np.isnan(X_train).any(axis=1)
X_train = X_train[indices_of_not_nan]
y_train = y_train[indices_of_not_nan]


X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

train_data_torch = X_train, y_train
test_data_torch = X_test, y_test