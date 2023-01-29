import torch.nn as nn
import torch 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5, padding=2)   # 32*32*3 -> 32*32*6
        self.maxpool1 = nn.MaxPool2d(2,stride=2) # 32*32*6 -> 16*16*6
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5, padding=2)  # 16*16*6 -> 16*16*16
        self.maxpool2 = nn.MaxPool2d(2,stride=2) # 16*16*16 -> 8*8*16
        self.fc1 = nn.Linear(16*8*8,120, bias=True)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120,84, bias=True)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84,10, bias=True)
        self.nl = nn.ReLU()

    def forward(self, x):
        x = self.nl(self.conv1(x))
        x = self.maxpool1(x)
        x = self.nl(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.nl(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.nl(x)
        
        return nn.functional.softmax(x, dim=-1)