Model Versions:

0.0.0
Basic logistic regression applied only on numeric fields. All rows with any NaN filtered out of X_train and y_train. No feature scaling.
num_epochs = 100
learning_rate = 0.01
columns = ["Survived","Pclass","Age","SibSp","Parch","Fare"]
Accuracy = 0.50 - 0.52
random state (for splitting of train and test data) = 1234

0.0.1
Feature scaling added
Accuracy unchanged

0.0.2
num_epochs = 1000
loss converges to around 0.58
accuracy = 0.5475

0.0.3
learning_rate = 0.1
accuracy = 0.5363

0.0.4
learning_rate = 0.1
columns = ["Survived","Pclass","Age","Fare"]
accuracy = 0.5475

0.0.5
num_epochs = 10,000
loss converges to around 0.5748
accuracy = 0.5531

0.0.6
Try individual features.
Age --> accuracy = 0.4693
Pclass --> accuracy = 0.6816
Fare --> accuracy = 0.6369

0.0.7
features = ['Survived','Fare','Pclass']
accuracy = 0.6536

Switched to cleaning data in its own script prep_data.py. Using basic feature transformation now.

0.1.0
Sex converted to binary
Binary Sex only (scaled) --> accuracy = 0.8212
Binary Sex only (not scaled) --> accuracy = 0.8212 (unchanged as expected)

0.1.1
features = ['Survived','Binary Sex','Pclass']
accuracy = 0.8212

0.1.2
features = ['Survived','Binary Sex','Age']
accuracy = 0.6536

0.1.3
created features 'Child', 'Teen', 'Young Adult'
Child only --> accuracy = 0.6145
Teen only --> accuracy = 0.6201
Young Adult only --> accuracy = 0.6089
All --> accuracy = 0.6145
Child age = 15 --> accuracy = 0.6257

0.1.4
features = ['Survived','Child','Parch','SibSp'] --> accuracy = 0.6313

0.1.5
squared Pclass, no effect on accuracy
neither does 1/Pclass^2

0.1.6
features = ['Survived','Pclass','Child'] --> accuracy = 0.6983
features = ['Survived','Pclass','Child','Binary Sex'] --> accuracy = 0.8268

0.1.7
categorised port of embarkation
features = ['Survived','Cherbourg','Queenstown']
accuracy = 0.6201

0.1.8
features = ['Survived','Cherbourg','Queenstown','Child','Binary Sex','Parch','SibSp','Pclass']
accuracy = 0.6201

Need to interrogate data more to find better features

0.1.9
removed nan ages
accuracy = 0.5664

0.2.0
Changing to pandas-based data manipulation in prep_data_2.py
gender only --> 0.8212 as before

0.2.1
categorising into "child","young man", "young woman", "old man", "old woman" did not improve the accuracy. This is relatively expected as they are just combinations of bucket age and gender
accuracy = 0.8212

0.2.2
new category: "no friends or family"
No impact on accuracy. This is surprising as I would've thought this is new information for the model.
Going to use tensor board now to investigate learning curves to see what might be a good next step.

TEST:
change the number of samples in the train dataset (test set kept constant). Look for signs of high variance or high bias.
two random states used for train test data split. Random state for split of data into test and train changed to "12" from "1234" as test accuracy was higher than train accuracy
FINDINGS:
plots of test and train accuracy vs num samples in training shows:
1.1 (rs = 12) the train accuracy starts at 0.9+, drops to 0.75 (100-200 samples) and then approximately converges to 0.8
1.2 (rs = 1234) the train accuracy starts at 0.9+, converges to around 0.77.
2.1 (rs = 12) test accuracy starts below 0.5, and converges on 0.7486 after 25 samples.
2.2 (rs = 1234) test accuracy starts below 0.5, converges to 0.82 after around 100 samples
CONCLUSION:
test error and train error are approximately equal, but quite high. This indicates high bias in the model. Therefore:
1. increasing n_samples won't help
2. more features are needed


0.2.3
continuing on with rs = 12
accuracy = 0.7486
bucketised age to 0-15, 15-30, 30-50, 50+ but little correlation shown except with 0-15

0.2.4
bucket embarkment port. correlations: C=0.168240, Q=0.003650, S=-0.155660
C and S features added
accuracy = 0.7486 again

0.2.5
features = ["Survived","child", "50+", "male","Pclass"]
accuracy = 0.7430
first feature in a while to affect accuracy. but made it worse.
features = ["Survived","child", "50+", "male", "2^class"]
accuracy still 0.7430

0.2.6
to avoid being too affected by the random state of the train_test_split, I will take the average of 10 runs (have since learned that this is equivalent to k-Fold Cross Validation)
features = ["Survived","male"]
mean accuracy over 10 random states = 0.7749
features = ["Survived","child", "male"]
mean accuracy over 10 random states = 0.7749
mean accuracy over 100 random states = 0.7876
features = ["Survived","child", "male", "Pclass"]
mean accuracy over 10 random states = 0.7827
mean accuracy over 100 random states = 0.7920
features = ["Survived","child", "male", "2^class"]
mean accuracy over 10 random states = 0.7860
mean accuracy over 100 random states = 0.7952 --> 2^class increases accuracy

features = ["Survived","child", "male", "10^class"]
mean accuracy over 10 random states = 0.7888 --> 10^class is better

optim x^class
2^class = 0.7860
3^class = 0.7877
4^class = 0.7888
5^class = 0.7888
10^class = 0.7888

Tried bucketing classes upper and lower:
bucketised (3 buckets) mean accuracy over 10 random states = 0.7849
bucketised (lower, upper+middle) mean accuracy over 10 random states = 0.7877

1.0.0
Moving to SKLearn RandomForestClassifier() as model in second_model.py
Temporarily changed prep_data_2.py to output a df instead of the torch array

features = ["Survived","child", "male", "10^class"]
mean accuracy over 10 random states = 0.7832 (marginally worse)

1.0.1

train_features = ['Fare', 'male', 'child', '15 - 30', '30 - 50', '50+', 
                  'C', 'Q', 'S', 'no friends or family', '10^class']
mean accuracy over 10 random states = 0.7939 (best yet)

Feature importances:
Fare                    0.397313
male                    0.300901
10^class                0.112182
child                   0.039469
no friends or family    0.034521
15 - 30                 0.030588
30 - 50                 0.027624
S                       0.020783
C                       0.013044
Q                       0.012138
50+                     0.011437

I will now remove the lowest importance features and see until I maximise accuracy

Left with:
Fare        0.530653 (interestingly this is higher than "male" despite having a lower correlation with df.corr())
male        0.362416
10^class    0.106931
mean accuracy over 10 random states = 0.8145 (best yet)

1.1.0
Implemented k-fold cross-validation, with 10 splits Results were noisy.
mean accuracy with 10 splits = 0.8227, standard deviation = 0.0398
mean accuracy with 10 splits = 0.8193, standard deviation = 0.0390
mean accuracy with 10 splits = 0.8237, standard deviation = 0.0288

1.1.1
Implemented repeated k-fold cross-validation with 10 splits and 10 repeats
mean accuracy with 10 splits and 10 repeats = 0.8153, (std = 0.0398)
mean accuracy with 10 splits and 10 repeats = 0.8142, (std = 0.0394)
mean accuracy with 10 splits and 10 repeats = 0.8163, (std = 0.0395)
Much less noisy now

1.1.2
Found the source of noise was the randomforestclassifier, set random_state=1234
mean accuracy with 10 splits and 10 repeats = 0.8163, (std = 0.0396)

###
Submitting to kaggle
second_model_evaluation.py made to do this
nan in "Fare" feature was replaced using mean values
score = 0.77511 (3,965th)
###

1.2.0
Let the hyperparameter tuning begin (using RandomizedSearchCV)
CONTINUE WITH https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
Pre random-search: Test Accuracy: 0.8156 (n_estimators=100, random_state=1234)

n_estimators = [int(x) for x in np.linspace(start = 20, stop = 400, num = 10)]
max_features = ['sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, 
                                n_iter = 200, verbose=1, random_state=1234, 
                                n_jobs = -1)

Fitting 5 folds for each of 200 candidates, totalling 1000 fits
{'n_estimators': 315, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
Best train accuracy: 0.8259
Test accuracy: 0.8436

Now a GridSearchCV based on the previous


GRID SEARCH #1
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 400, num = 20)]
max_features = ['sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [5, 10, 20]
min_samples_leaf = [2, 4]
bootstrap = [True]

Fitting 5 folds for each of 1440 candidates, totalling 7200 fits
{'bootstrap': True, 'max_depth': 70, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 36}
Best train accuracy: 0.8273
Test accuracy: 0.8436

GRID SEARCH #2
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 80, num = 20)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [4, 8, 16]
min_samples_leaf = [1, 2, 4]

Fitting 5 folds for each of 2160 candidates, totalling 10800 fits
hyper_params_2 = {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 20}
Best train accuracy: 0.8328
Test accuracy: 0.8324

Higher train accuracy but lower test accuracy. I'll k-Fold cross validated the two and pick the best (third_model.py)
hyper_params_1: Mean test accuracy: 82.70%, std = 0.03780837606119187
hyper_params_2: Mean test accuracy: 81.96%, std = 0.04316061945017869
Conclusion: hyper_params_2 are likely overfit

all nan were replaced using mean values

###
Submitting hyper_params_1 to kaggle
score = 0.77033 (7,254th)
###

###
Submitting hyper_params_2 to kaggle
score = 0.77511 (3,965th) - same as baseline model
###

Next step is to start looking at examples of failures: (abandoned for now)
In the incorrect predictions, 19 women were incorrectly predicted to survive and 9 to die
Survival rate for women correctly predicted = 82%, incorrectly predicted = 32%


NEXT STEPS
tweak RandomForestClassifier parameters, 
add regularisation, 
try pca
look at failure examples

