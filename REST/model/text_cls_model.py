import torch
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier,self).__init__()
        self.fc1 = nn.Linear(768*20,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,2)
    def forward_feature(self,x):
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # x = torch.nn.functional.normalize(x,2)
        return x
    def forward(self,x):
        x = self.forward_feature(x)
        return self.fc3(x)