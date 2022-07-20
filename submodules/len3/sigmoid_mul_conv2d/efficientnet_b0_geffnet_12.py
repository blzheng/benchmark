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
        self.conv2d64 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x185, x181):
        x186=x185.sigmoid()
        x187=operator.mul(x181, x186)
        x188=self.conv2d64(x187)
        return x188

m = M().eval()
x185 = torch.randn(torch.Size([1, 1152, 1, 1]))
x181 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x185, x181)
end = time.time()
print(end-start)
