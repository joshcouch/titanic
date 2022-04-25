Model Versions:

1.0.1
Basic logistic regression applied only on numeric fields. All rows with any NaN filtered out of X_train and y_train. No feature scaling.
num_epochs = 100
learning_rate = 0.01
columns = ["Survived","Pclass","Age","SibSp","Parch","Fare"]
Accuracy = 0.50 - 0.52

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
Binary Sex only (not scaled) --> accuracy = 0.8212 (unchanged)

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
