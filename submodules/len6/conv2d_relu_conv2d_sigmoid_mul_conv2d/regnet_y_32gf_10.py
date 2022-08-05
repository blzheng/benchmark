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
        self.conv2d56 = Conv2d(1392, 348, kernel_size=(1, 1), stride=(1, 1))
        self.relu43 = ReLU()
        self.conv2d57 = Conv2d(348, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d58 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x176, x175):
        x177=self.conv2d56(x176)
        x178=self.relu43(x177)
        x179=self.conv2d57(x178)
        x180=self.sigmoid10(x179)
        x181=operator.mul(x180, x175)
        x182=self.conv2d58(x181)
        return x182

m = M().eval()
x176 = torch.randn(torch.Size([1, 1392, 1, 1]))
x175 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x176, x175)
end = time.time()
print(end-start)
