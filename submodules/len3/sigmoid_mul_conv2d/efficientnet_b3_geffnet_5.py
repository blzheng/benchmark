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
        self.conv2d28 = Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x82, x78):
        x83=x82.sigmoid()
        x84=operator.mul(x78, x83)
        x85=self.conv2d28(x84)
        return x85

m = M().eval()
x82 = torch.randn(torch.Size([1, 192, 1, 1]))
x78 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x82, x78)
end = time.time()
print(end-start)
