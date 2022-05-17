"""
New model using SKLearn random forest regressor
"""
import numpy as np
import pandas as pd
import data_summary
import prep_data_2
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
summary_stats_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/summary stats.txt"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

data_summary.main(train_csv_path, summary_stats_path)

data_df = prep_data_2.main(train_csv_path=train_csv_path)

X_features = ['Fare', 'male', '10^class']
y_feature = "Survived"

X = data_df[X_features]
y = data_df[y_feature]

k = 5
n_repeats = 1

rkfold = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=42)
acc_running = []

# Define Randomized Search CV grid
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 400, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

for train_index, test_index in rkfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf = RandomForestClassifier(n_estimators=100, random_state=1234)
    clf.fit(X_train, y_train)

    y_pred=clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    acc_running.append(acc)

print(f"mean accuracy with {k} splits and {n_repeats} repeats = {np.mean(acc_running):.4f}, " + \
      f"(std = {np.std(acc_running):.4f})")

feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)

# final_correct = (y_test == y_pred).reset_index(drop=True)
# final_incorrect = (y_test != y_pred).reset_index(drop=True)
# y_pred_df = pd.DataFrame(y_pred, columns=["y_pred"])
# X_test.reset_index(drop=True, inplace=True)
# y_test.reset_index(drop=True, inplace=True)

# correct_df = pd.concat([X_test[final_correct],y_test[final_correct],y_pred_df[final_correct]], axis=1)
# incorrect_df = pd.concat([X_test[final_incorrect],y_test[final_incorrect],y_pred_df[final_incorrect]], axis=1)
# correct_df.to_csv("Datasets/correct.csv")
# incorrect_df.to_csv("Datasets/incorrect.csv")


