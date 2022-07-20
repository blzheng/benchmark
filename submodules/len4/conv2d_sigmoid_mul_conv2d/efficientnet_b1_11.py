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
        self.conv2d57 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d58 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x174, x171):
        x175=self.conv2d57(x174)
        x176=self.sigmoid11(x175)
        x177=operator.mul(x176, x171)
        x178=self.conv2d58(x177)
        return x178

m = M().eval()
x174 = torch.randn(torch.Size([1, 20, 1, 1]))
x171 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x174, x171)
end = time.time()
print(end-start)
