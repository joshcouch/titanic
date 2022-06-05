import numpy as np
import torch
import torch.nn as nn
import data_summary
import prep_data_2
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

# Define Paths
train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"
summary_stats_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/summary stats.txt"
final_test_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/test.csv"

data_summary.main(train_csv_path, summary_stats_path)

def train_and_test(test_size=0.2, random_state=12):
    (X_train, y_train), (X_test, y_test) = prep_data_2.main(test_size=0.2, random_state=random_state)

    n_samples, n_features = X_train.shape

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


        # if (epoch+1) % 1000 == 0:
        #     print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    #writer.flush()
    #writer.close()

    # evaluation
    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(y_test)
        acc = acc.sum()/float(y_test.shape[0])
        print(f'accuracy = {acc:.4f} (random state = {random_state})')
    
    return acc

acc_running = []
for i in range(10):
    acc_running.append(train_and_test(test_size=0.2, random_state=i))

print(f'mean accuracy over 10 random states = {np.mean(acc_running):.4f}')