Model Versions:

1.0.1
Basic logistic regression applied only on numeric fields. All rows with any NaN filtered out of X_train and y_train. No feature scaling.
num_epochs = 100
learning_rate = 0.01
columns = ["Survived","Pclass","Age","SibSp","Parch","Fare"]
Accuracy = 0.50 - 0.52
random state (for splitting of train and test data) = 1234

1.0.2
Feature scaling added
Accuracy unchanged

1.0.3
num_epochs = 1000
loss converges to around 0.58
accuracy = 0.5475

1.0.4
learning_rate = 0.1
accuracy = 0.5363

1.0.5
learning_rate = 0.1
columns = ["Survived","Pclass","Age","Fare"]
accuracy = 0.5475

1.0.6
num_epochs = 10,000
loss converges to around 0.5748
accuracy = 0.5531

1.0.7
Try individual features.
Age --> accuracy = 0.4693
Pclass --> accuracy = 0.6816
Fare --> accuracy = 0.6369

1.0.10
features = ['Survived','Fare','Pclass']
accuracy = 0.6536

Switched to cleaning data in its own script prep_data.py. Using basic feature transformation now.

1.1.1
Sex converted to binary
Binary Sex only (scaled) --> accuracy = 0.8212
Binary Sex only (not scaled) --> accuracy = 0.8212 (unchanged as expected)

1.1.2
features = ['Survived','Binary Sex','Pclass']
accuracy = 0.8212

1.1.3
features = ['Survived','Binary Sex','Age']
accuracy = 0.6536

1.1.4
created features 'Child', 'Teen', 'Young Adult'
Child only --> accuracy = 0.6145
Teen only --> accuracy = 0.6201
Young Adult only --> accuracy = 0.6089
All --> accuracy = 0.6145
Child age = 15 --> accuracy = 0.6257

1.1.5
features = ['Survived','Child','Parch','SibSp'] --> accuracy = 0.6313

1.1.6
squared Pclass, no effect on accuracy
neither does 1/Pclass^2

1.1.7
features = ['Survived','Pclass','Child'] --> accuracy = 0.6983
features = ['Survived','Pclass','Child','Binary Sex'] --> accuracy = 0.8268

1.1.8
categorised port of embarkation
features = ['Survived','Cherbourg','Queenstown']
accuracy = 0.6201

1.1.9
features = ['Survived','Cherbourg','Queenstown','Child','Binary Sex','Parch','SibSp','Pclass']
accuracy = 0.6201

Need to interrogate data more to find better features

1.1.10
removed nan ages
accuracy = 0.5664

1.2.0
Changing to pandas-based data manipulation in prep_data_2.py
gender only --> 0.8212 as before

1.2.1
categorising into "child","young man", "young woman", "old man", "old woman" did not improve the accuracy. This is relatively expected as they are just combinations of bucket age and gender
accuracy = 0.8212

1.2.2
new category: "no friends or family"
No impact on accuracy. This is surprising as I would've thought this is new information for the model.
Going to use tensor board now to investigate learning curves to see what might be a good next step.

TEST:
change the number of samples in the train dataset (test set kept constant). Look for signs of high variance or high bias.
two random states used for train test data split. (random state for split of data into test and train changed to "12" from "1234" as test accuracy was higher than train accuracy)
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


1.2.3
continuing on with rs = 12
accuracy = 0.7486
bucketised age to 0-15, 15-30, 30-50, 50+ but little correlation shown except with 0-15

1.2.4
bucket embarkment port. correlations: C=0.168240, Q=0.003650, S=-0.155660
C and S features added
accuracy = 0.7486 again

1.2.5
features = ["Survived","child", "50+", "male","Pclass"]
accuracy = 0.7430
first feature in a while to affect accuracy. but made it worse.
features = ["Survived","child", "50+", "male", "2^class"]
accuracy still 0.7430

1.2.6
to avoid being too affected by the random state of the train_test_split, I will take the average of 10 runs
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
mean accuracy over 10 random states = 0.7888

TRY bucketing classes upper and lower

