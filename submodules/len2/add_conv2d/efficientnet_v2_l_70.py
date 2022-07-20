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
        self.conv2d328 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x1052, x1037):
        x1053=operator.add(x1052, x1037)
        x1054=self.conv2d328(x1053)
        return x1054

m = M().eval()
x1052 = torch.randn(torch.Size([1, 640, 7, 7]))
x1037 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1052, x1037)
end = time.time()
print(end-start)
