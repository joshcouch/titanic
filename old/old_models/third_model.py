"""
New model using SKLearn random forest regressor
"""
import numpy as np
import pandas as pd
import data_summary
import prep_data_2
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.inspection import permutation_importance

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
summary_stats_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/summary stats.txt"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

data_summary.main(train_csv_path, summary_stats_path)

data_df = prep_data_2.main(train_csv_path=train_csv_path)

y_feature = "Survived"

X_features = ['Age', 'Fare', 'female', 'no friends or family', '10^class']

X = data_df[X_features]
y = data_df[y_feature]

hyper_params = {'bootstrap': True, 'max_depth': 70, 
                'max_features': 'sqrt', 'min_samples_leaf': 2, 
                'min_samples_split': 5, 'n_estimators': 36}

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1234)

running_acc = []

for train_index, test_index in rkf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf = RandomForestClassifier(**hyper_params)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    running_acc.append(acc)

    #print(f"Test accuracy: {acc:.4f}")

print(f"Mean test accuracy: {np.mean(running_acc)*100:.2f}%, std = {np.std(running_acc):.4f}")

result = permutation_importance(
    clf, X_test, y_test, n_repeats=10, random_state=1234, n_jobs=2
)

importances = result.importances_mean

forest_importances = pd.Series(importances, index=X.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()


importances = clf.feature_importances_
forest_importances = pd.Series(importances, index=X_features)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()