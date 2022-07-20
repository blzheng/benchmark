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
        self.conv2d44 = Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x123, x128):
        x129=operator.mul(x123, x128)
        x130=self.conv2d44(x129)
        return x130

m = M().eval()
x123 = torch.randn(torch.Size([1, 480, 14, 14]))
x128 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x123, x128)
end = time.time()
print(end-start)
