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
        self.conv2d67 = Conv2d(440, 110, kernel_size=(1, 1), stride=(1, 1))
        self.relu51 = ReLU()
        self.conv2d68 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d69 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x210, x209):
        x211=self.conv2d67(x210)
        x212=self.relu51(x211)
        x213=self.conv2d68(x212)
        x214=self.sigmoid12(x213)
        x215=operator.mul(x214, x209)
        x216=self.conv2d69(x215)
        return x216

m = M().eval()
x210 = torch.randn(torch.Size([1, 440, 1, 1]))
x209 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x210, x209)
end = time.time()
print(end-start)
