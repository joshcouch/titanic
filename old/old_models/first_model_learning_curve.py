import torch
import torch.nn as nn
import data_summary
import prep_data_2
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
summary_stats_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/summary stats.txt"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

def train_and_test(test_size=0.2, n_samples = None):
    (X_train, y_train), (X_test, y_test) = prep_data_2.main(test_size=test_size)
    
    if not n_samples:
        n_samples, n_features = X_train.shape
    else:
        n_features = X_train.shape[1]
        X_train = X_train[:n_samples]
        y_train = y_train[:n_samples]

    # model
    class LogisticRegression(nn.Module):

        def __init__(self, n_input_features) -> None:
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(n_input_features, 1)

        def forward(self, x):
            y_predicted = torch.sigmoid(self.linear(x))
            return y_predicted

    model = LogisticRegression(n_features)

    # loss and optimizer
    learning_rate = 0.01
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    num_epochs = 10000
    for epoch in range(num_epochs):
        y_predicted = model(X_train)
        loss = criterion(y_predicted, y_train)
        loss.backward()
        #writer.add_scalar("Loss/train", loss, epoch)

        optimizer.step()
        optimizer.zero_grad()


        if (epoch+1) % 2000 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    # evaluation against train and test set
    with torch.no_grad():
        y_predicted_train = model(X_train)
        y_predicted_train_cls = y_predicted_train.round()
        train_acc = y_predicted_train_cls.eq(y_train)
        train_acc = train_acc.sum()/float(y_train.shape[0])
        print(f'[TRAIN] number of samples = {n_samples}, accuracy = {train_acc:.4f}')
        writer.add_scalar("training accuracy vs num samples", train_acc, n_samples)
        
        y_predicted_test = model(X_test)
        y_predicted_test_cls = y_predicted_test.round()
        test_acc = y_predicted_test_cls.eq(y_test)
        test_acc = test_acc.sum()/float(y_test.shape[0])
        print(f'[TEST] number of samples = {n_samples}, accuracy = {test_acc:.4f}')
        writer.add_scalar("test accuracy vs num samples", test_acc, n_samples)
    
    writer.flush()
    
for num in range(1,27):
    train_and_test(n_samples=num**2)

writer.close()