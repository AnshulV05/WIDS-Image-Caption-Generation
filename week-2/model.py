import torch as th

class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = th.nn.Linear(784, 196, bias=True)
        self.nl1 = th.nn.ReLU()
        self.fc2 = th.nn.Linear(196, 50, bias=True)
        self.bn = th.nn.BatchNorm1d(num_features=50)
        self.fc3 = th.nn.Linear(50,10 , bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nl1(x)
        x = self.fc2(x)
        x = self.nl1(x)
        x = self.bn(x)
        x = self.fc3(x)
        return th.nn.functional.softmax(x, dim=-1)

