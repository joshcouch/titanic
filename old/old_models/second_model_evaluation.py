"""
Train model developed in second_model.py with whole train.csv, then create predictions on test.csv
"""
import numpy as np
import pandas as pd
import data_summary
import prep_data_2
from sklearn.ensemble import RandomForestClassifier

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

train_data_df = prep_data_2.main(train_csv_path=train_csv_path)
test_data_df = prep_data_2.main(train_csv_path=final_test_csv_path)

X_features = ['Fare', 'male', '10^class']
y_feature = "Survived"

X_train = train_data_df[X_features]
y_train = train_data_df[y_feature]

X_test = test_data_df[X_features]
X_test.fillna(X_test["Fare"].mean(), inplace=True)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

y_pred_df = pd.DataFrame()
y_pred_df["PassengerId"] = test_data_df["PassengerId"]
y_pred_df["Survived"] = y_pred

y_pred_df.to_csv("Datasets/submission 1.csv", index=False)

