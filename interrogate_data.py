import torch
import prep_data
import pandas as pd
import numpy as np

train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"

features = ['Survived','PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

train_data_df = pd.read_csv(train_csv_path)
train_data_np = train_data_df[features].to_numpy()

for i in range(train_data_np.shape[1]):
    n_nan = 0
    column = train_data_np[:,i]
    if type(column[1]) is not str:
        for x in column:
            if np.isnan(x):
                n_nan += 1
    else:
        for x in column:
            if type(x) is float:
                n_nan += 1
    print(f"{features[i]}: n_nan = {n_nan}")
