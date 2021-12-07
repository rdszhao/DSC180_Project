import torch
import torch.nn as nn

input_size = 15
output_size = 2
hidden_size = 10
num_epochs = 2
learning_rate = 0.001


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output

    def train(self, X, y):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for i in range(20000):
            # optimizer.zero_grad()
            # outputs = self(X.float())
            # loss = criterion(outputs, y.float())
            # loss.backward()
            # optimizer.step()
            for x_i, y_i in zip(X, y):
                outputs = self(x_i.float())
                loss = criterion(outputs, y_i.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X):
        maps = []
        for x in X:
            testX = torch.from_numpy(x)
            prediction = self(testX.float()).tolist()
            maps.append(prediction)
        return maps
