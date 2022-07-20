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
        self.conv2d197 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x618, x603):
        x619=operator.add(x618, x603)
        x620=self.conv2d197(x619)
        return x620

m = M().eval()
x618 = torch.randn(torch.Size([1, 384, 7, 7]))
x603 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x618, x603)
end = time.time()
print(end-start)
