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
        self.conv2d62 = Conv2d(440, 110, kernel_size=(1, 1), stride=(1, 1))
        self.relu47 = ReLU()
        self.conv2d63 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d64 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x194, x193):
        x195=self.conv2d62(x194)
        x196=self.relu47(x195)
        x197=self.conv2d63(x196)
        x198=self.sigmoid11(x197)
        x199=operator.mul(x198, x193)
        x200=self.conv2d64(x199)
        return x200

m = M().eval()
x194 = torch.randn(torch.Size([1, 440, 1, 1]))
x193 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x194, x193)
end = time.time()
print(end-start)
