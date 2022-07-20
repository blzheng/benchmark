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
        self.conv2d116 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()
        self.conv2d117 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x361, x358):
        x362=self.conv2d116(x361)
        x363=self.sigmoid23(x362)
        x364=operator.mul(x363, x358)
        x365=self.conv2d117(x364)
        return x365

m = M().eval()
x361 = torch.randn(torch.Size([1, 44, 1, 1]))
x358 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x361, x358)
end = time.time()
print(end-start)
