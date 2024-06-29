from utils import *
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models


def model_selector(model_name):

    # default values
    batch_size = 64
    train_stop_criteria = 'valid loss'
    train_stop_patience = 10
    lr_scheduler = []

    if model_name == 'LeNet5':
        batch_size = 100
        learn_rate = 0.005*2
        model = LeNet5()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.66)
        criterion = nn.CrossEntropyLoss()

    elif model_name == 'LeNet5+Dropout':
        batch_size = 100
        learn_rate = 0.005*2
        dropout_prob = 0.2
        model = LeNet5_drop(dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.66)
        criterion = nn.CrossEntropyLoss()

    elif model_name == 'LeNet5+Weight_decay':
        batch_size = 100
        learn_rate = 0.005*2
        weight_dec = 2e-4
        model = LeNet5()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_dec)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.66)
        criterion = nn.CrossEntropyLoss()

    elif model_name == 'LeNet5+batch_norm':
        batch_size = 100
        learn_rate = 0.005*2
        model = LeNet5_batch_norm()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.66)
        criterion = nn.CrossEntropyLoss()

    else:
        print('Error - wrong model specified')
        model = optimizer = criterion = 0

    os.makedirs('models/' + model_name, exist_ok=True)

    return model, optimizer, lr_scheduler, criterion, batch_size, train_stop_criteria, train_stop_patience

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Sigmoid()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Sigmoid()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1 * 1 * 120, 84)
        self.act4 = nn.Sigmoid()
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

class LeNet5_drop(nn.Module):
    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Sigmoid()
        self.drop1 = nn.Dropout2d(dropout_prob)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Sigmoid()
        self.drop2 = nn.Dropout2d(dropout_prob)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1 * 1 * 120, 84)
        self.act4 = nn.Sigmoid()
        self.drop3 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.drop1(self.act2(self.conv2(x)))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.drop2(self.act3(self.conv3(x)))
        # input 120x1x1, output 84
        x = self.drop3(self.act4(self.fc1(self.flat(x))))
        # input 84, output 10
        x = self.fc2(x)
        return x
class LeNet5_batch_norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.act2 = nn.Sigmoid()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(120)
        self.act3 = nn.Sigmoid()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1 * 1 * 120, 84)
        self.bn3 = nn.BatchNorm1d(84)
        self.act4 = nn.Sigmoid()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.act2(self.bn1(self.conv2(x)))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.act3(self.bn2(self.conv3(x)))
        # input 120x1x1, output 84
        x =self.act4( self.bn3(self.fc1(self.flat(x))))
        # input 84, output 10
        x = self.fc2(x)
        return x