from utils import *
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models


def model_selector(model_name):

    # default values
    batch_size = 64
    train_stop_criteria = 'valid loss'  # 'F1 score'
    train_stop_patience = 5

    if model_name == 'LeNet5':
        learn_rate = 0.001
        model = LeNet5()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    if model_name == 'LeNet5_web':
        batch_size = 100
        model = LeNet5_web()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

    elif model_name == 'LeNet5+Dropout':
        batch_size = 256
        learn_rate = 0.001/4
        dropout_prob = 0.2
        model = LeNet5_drop(dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'LeNet5+Weight_decay':
        batch_size = 128
        learn_rate = 0.002
        weight_dec = 0.001
        model = LeNet5()
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, weight_decay=weight_dec)
        criterion = nn.NLLLoss()

    elif model_name == 'LeNet5+batch_norm':
        batch_size = 128
        learn_rate = 0.002
        model = LeNet5_batch_norm()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    else:
        print('Error - wrong model specified')
        model = optimizer = criterion = 0

    os.makedirs('models/' + model_name, exist_ok=True)

    return model, optimizer, criterion, batch_size, train_stop_criteria, train_stop_patience

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        y = self.LogSoftmax(y)
        return y

class LeNet5_drop(nn.Module):
    def __init__(self, dropout_prob):
        super(LeNet5_drop, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout2d(dropout_prob)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.drop1(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.drop2(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.drop3(y)
        y = self.fc3(y)
        y = self.relu5(y)
        y = self.LogSoftmax(y)
        return y

class LeNet5_batch_norm(nn.Module):
    def __init__(self):
        super(LeNet5_batch_norm, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.norm2 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.norm3 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.norm1(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.norm2(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.norm3(y)
        y = self.fc3(y)
        y = self.relu5(y)
        y = self.LogSoftmax(y)
        return y


class LeNet5_web(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1 * 1 * 120, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.act2(self.conv2(x))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.act3(self.conv3(x))
        # input 120x1x1, output 84
        x = self.act4(self.fc1(self.flat(x)))
        # input 84, output 10
        x = self.fc2(x)
        return x