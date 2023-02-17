
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float64))
        self.y = torch.from_numpy(y.astype(np.float64)).type(torch.LongTensor)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], np.array(self.y[index])
   
    def __len__(self):
        return self.len


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 70)
        self.linear3 = nn.Linear(70, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.normal = torch.nn.BatchNorm1d(hidden_dim)
        self.linear_final = nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax()



    def forward(self, x):
        x = x.to(torch.float32)
        
        x = self.linear1(x)
        x = self.relu(x)


        x = self.linear2(x)
        x = self.relu(x)
        

        x = self.linear3(x)
        x = self.relu(x)


        x = self.linear4(x)
        x = self.relu(x)

        #x = x.to(torch.float32)
        
        x = self.linear5(x)
        x = self.relu(x)

        x = self.linear_final(x)
        x = x.to(torch.float32)

        return x


def train_network(X_train, y_train, X_test, y_test):
    print("> topicModel: topicNetwork: training network")
    batch_size = 2
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    for batch, (X, y) in enumerate(train_dataloader):
        print(f"Batch: {batch+1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        break



    input_dim = 150
    hidden_dim = 150
    output_dim = 70


       
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    #print(model)



    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #e-1

    num_epochs = 200
    loss_values = []


    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            pred = model(X).type(torch.float32)
            loss = cross_el(pred, y)

            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

    print("Training Complete")

    step = np.linspace(0, 100, len(loss_values))

    fig, ax = plt.subplots(figsize=(10,8))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


    y_pred = []
    y_test = []
    total = 0
    correct = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            print(y)
            print("--")
            y_pred.append(predicted)
            y_test.append(y)
            total += 1

    torch.save(model, './trainedmodels/dvlabeler'+str(total))

class network_handler:
    def __init__(self, path):
        self.model = torch.load(path)
    def classify(self, X):
        outputs = self.model(torch.from_numpy(X))
        _, predicted = torch.max(outputs.data, 1)
        return(predicted)



