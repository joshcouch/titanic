"""
Train model developed in second_model.py with whole train.csv, then create predictions on test.csv
"""
import numpy as np
import pandas as pd
import prep_data_2
from sklearn.ensemble import RandomForestClassifier

from third_model import X_train

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

train_data_df = prep_data_2.main(train_csv_path=train_csv_path)
test_data_df = prep_data_2.main(train_csv_path=final_test_csv_path)

y_feature = "Survived"

X_train = train_data_df.iloc[:,2:]
y_train = train_data_df[y_feature]

hyper_params = {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 20}

clf = RandomForestClassifier(**hyper_params)

clf.fit(X_train, y_train)

X_test = test_data_df.iloc[:,1:]

y_pred = clf.predict(X_test)

y_pred_df = pd.DataFrame()
y_pred_df["PassengerId"] = test_data_df["PassengerId"]
y_pred_df["Survived"] = y_pred

y_pred_df.to_csv("Datasets/submission 3.csv", index=False)

