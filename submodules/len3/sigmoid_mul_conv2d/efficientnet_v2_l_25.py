import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.sigmoid25 = Sigmoid()
        self.conv2d162 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x523, x519):
        x524=self.sigmoid25(x523)
        x525=operator.mul(x524, x519)
        x526=self.conv2d162(x525)
        return x526

m = M().eval()
x523 = torch.randn(torch.Size([1, 1344, 1, 1]))
x519 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x523, x519)
end = time.time()
print(end-start)
