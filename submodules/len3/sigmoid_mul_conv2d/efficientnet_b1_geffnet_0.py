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
        self.conv2d4 = Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x10, x6):
        x11=x10.sigmoid()
        x12=operator.mul(x6, x11)
        x13=self.conv2d4(x12)
        return x13

m = M().eval()
x10 = torch.randn(torch.Size([1, 32, 1, 1]))
x6 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x10, x6)
end = time.time()
print(end-start)
