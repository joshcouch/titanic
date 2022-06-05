"""
New model using SKLearn random forest regressor
"""
import numpy as np
import pandas as pd
import data_summary
import prep_data_2
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
summary_stats_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/summary stats.txt"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

data_df = prep_data_2.main(train_csv_path=train_csv_path)

y_feature = "Survived"
X_features = ['Age', 'Fare', 'female', 'no friends or family', '10^class']

X = data_df[X_features]
y = data_df[y_feature]

# Define model
model = RandomForestClassifier()

# Define seach space
search_space = {}
search_space["max_features"] = ["sqrt", "log2"]
search_space["n_estimators"] = np.linspace(10, 400, 50, dtype=int)
search_space["max_depth"] = np.linspace(2, 30, 10, dtype=int)
search_space["min_samples_leaf"] = np.linspace(2, 30, 10, dtype=int)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1234)

search = RandomizedSearchCV(model, search_space, cv=rskf, scoring='accuracy', n_iter=500, verbose=1, n_jobs=-1)

result = search.fit(X, y)

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)