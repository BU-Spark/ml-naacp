
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x


def train_network(X_train, y_train, X_test, y_test):
    batch_size = 64
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    for batch, (X, y) in enumerate(train_dataloader):
        print(f"Batch: {batch+1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        break
        """
        Batch: 1
        X shape: torch.Size([64, 2])
        y shape: torch.Size([64])
        """


    input_dim = 150
    hidden_dim = 150
    output_dim = 1


       
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    print(model)


    """
    NeuralNetwork(
    (layer_1): Linear(in_features=2, out_features=10, bias=True)
    (layer_2): Linear(in_features=10, out_features=1, bias=True)
    )
    """

    learning_rate = 0.1

    loss_fn = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 100
    loss_values = []


    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

    print("Training Complete")

    step = np.linspace(0, 100, 10500)

    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


    """
    We're not training so we don't need to calculate the gradients for our outputs
    """
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            predicted = np.where(outputs < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            y_pred.append(predicted)
            y_test.append(y)
            total += y.size(0)
            correct += (predicted == y.numpy()).sum().item()

    print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')





    y_pred = list(itertools.chain(*y_pred))
    y_test = list(itertools.chain(*y_test))


    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)

    plt.subplots(figsize=(8, 5))

    sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")

    plt.show()






