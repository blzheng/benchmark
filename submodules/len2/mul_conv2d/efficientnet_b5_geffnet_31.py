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
        self.conv2d157 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x462, x467):
        x468=operator.mul(x462, x467)
        x469=self.conv2d157(x468)
        return x469

m = M().eval()
x462 = torch.randn(torch.Size([1, 1824, 7, 7]))
x467 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x462, x467)
end = time.time()
print(end-start)
