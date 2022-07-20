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
        self.conv2d216 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x640, x645):
        x646=operator.mul(x640, x645)
        x647=self.conv2d216(x646)
        return x647

m = M().eval()
x640 = torch.randn(torch.Size([1, 2304, 7, 7]))
x645 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x640, x645)
end = time.time()
print(end-start)
