import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Features used
features = ["Survived","child", "male", "10^class"]

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

def binary_sex(df):
    gender_dummies = pd.get_dummies(df["Sex"])
    df = pd.concat([df, gender_dummies], axis=1)
    return df

def bucket_age(df):
    df["Age"].fillna(df["Age"].mean())
    df["child"] = (df["Age"] <= 15).astype(int)
    df["15 - 30"] = ((df["Age"] > 15) & (df["Age"] <= 30)).astype(int)
    df["30 - 50"] = ((df["Age"] > 30) & (df["Age"] <= 50)).astype(int)
    df["50+"] = (df["Age"] > 50).astype(int)
    return df

def emphasise_class(df):
    df["10^class"] = 10 ** df["Pclass"]
    return df

def bucket_port(df):
    port_dummies = pd.get_dummies(df["Embarked"])
    df = pd.concat([df, port_dummies], axis=1)
    return df

def strong_categorisation(df):
    """
    Get dummies for categories like child, young man, young woman
    """
    df["young man"] = ((df["male"] == 1) & (df["50+"] == 0)).astype(int)
    df["young woman"] = ((df["male"] == 0) & (df["50+"] == 0)).astype(int)
    df["old man"] = ((df["male"] == 1) & (df["50+"] == 1)).astype(int)
    df["old woman"] = ((df["male"] == 0) & (df["50+"] == 1)).astype(int)
    df["no friends or family"] = ((df["SibSp"] == 0) & (df["Parch"] == 0)).astype(int)
    return df

def prep_data(df):
    df = binary_sex(df)
    df = bucket_age(df)
    df = bucket_port(df)
    df = strong_categorisation(df)
    df = emphasise_class(df)
    return df

def scale_data(np_array):
    sc = StandardScaler()
    np_array_scaled = sc.fit_transform(np_array)
    return np_array_scaled

def convert_to_torch(final_df, test_size, random_state):
    train_data_np = final_df.to_numpy()

    X, y = train_data_np[:,1:], train_data_np[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    X_train = scale_data(X_train)
    X_test = scale_data(X_test)
    
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)

    train_data_torch = X_train, y_train
    test_data_torch = X_test, y_test
    return train_data_torch, test_data_torch
    
def main(test_size, random_state):
    train_df = pd.read_csv(train_csv_path)
    train_df = prep_data(train_df)
    #print(train_df.corr()["Survived"])
    #print(train_df.head())
    final_df = train_df[features].copy()
    
    train_data_torch, test_data_torch = convert_to_torch(final_df, test_size, random_state)
    
    return train_data_torch, test_data_torch

if __name__ == '__main__':
    main(0.2, None)
