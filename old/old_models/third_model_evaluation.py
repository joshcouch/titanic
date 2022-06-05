"""
Train model developed in second_model.py with whole train.csv, then create predictions on test.csv
"""
import numpy as np
import pandas as pd
import prep_data_2
from sklearn.ensemble import RandomForestClassifier

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

train_data_df = prep_data_2.main(train_csv_path=train_csv_path)
test_data_df = prep_data_2.main(train_csv_path=final_test_csv_path)

y_feature = "Survived"
X_features = ['Age', 'Fare', 'female', 'no friends or family', '10^class']

X_train = train_data_df[X_features]
y_train = train_data_df[y_feature]

hyper_params = {'n_estimators': 80, 'max_features': 'log2', 'max_depth': 8}

clf = RandomForestClassifier(**hyper_params)

clf.fit(X_train, y_train)

X_test = test_data_df[X_features]

y_pred = clf.predict(X_test)

y_pred_df = pd.DataFrame()
y_pred_df["PassengerId"] = test_data_df["PassengerId"]
y_pred_df["Survived"] = y_pred

y_pred_df.to_csv("Datasets/submission 6.csv", index=False)

