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
        self.conv2d56 = Conv2d(288, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x174, x169):
        x175=operator.mul(x174, x169)
        x176=self.conv2d56(x175)
        return x176

m = M().eval()
x174 = torch.randn(torch.Size([1, 288, 1, 1]))
x169 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x174, x169)
end = time.time()
print(end-start)
