"""
New model using SKLearn random forest regressor
"""
import numpy as np
import pandas as pd
import data_summary
import prep_data_2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
summary_stats_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/summary stats.txt"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

data_summary.main(train_csv_path, summary_stats_path)

data_df = prep_data_2.main(train_csv_path=train_csv_path)

X_features = ['Fare', 'male', '10^class']
y_feature = "Survived"

X = data_df.iloc[:,2:].select_dtypes(exclude=object)
X = X[X.columns[~X.isnull().any()]]
y = data_df[y_feature]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

# Define Randomized Search CV grid
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 80, num = 20)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [4, 8, 16]
min_samples_leaf = [1, 2, 4]

param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

clf = RandomForestClassifier()

clf_random = GridSearchCV(estimator = clf, param_grid = param_grid, 
                            verbose=1, n_jobs = -1)

clf_random.fit(X_train, y_train)
print(clf_random.best_params_)
print(f"Best train accuracy: {clf_random.best_score_:.4f}")

best_estimator = clf_random.best_estimator_
y_pred = best_estimator.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)

print(f"Test accuracy: {acc:.4f}")
