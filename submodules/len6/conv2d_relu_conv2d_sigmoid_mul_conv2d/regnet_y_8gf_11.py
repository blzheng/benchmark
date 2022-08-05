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
        self.conv2d61 = Conv2d(896, 224, kernel_size=(1, 1), stride=(1, 1))
        self.relu47 = ReLU()
        self.conv2d62 = Conv2d(224, 896, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d63 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x192, x191):
        x193=self.conv2d61(x192)
        x194=self.relu47(x193)
        x195=self.conv2d62(x194)
        x196=self.sigmoid11(x195)
        x197=operator.mul(x196, x191)
        x198=self.conv2d63(x197)
        return x198

m = M().eval()
x192 = torch.randn(torch.Size([1, 896, 1, 1]))
x191 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x192, x191)
end = time.time()
print(end-start)
