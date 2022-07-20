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
        self.conv2d192 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x566, x571):
        x572=operator.mul(x566, x571)
        x573=self.conv2d192(x572)
        return x573

m = M().eval()
x566 = torch.randn(torch.Size([1, 3072, 7, 7]))
x571 = torch.randn(torch.Size([1, 3072, 1, 1]))
start = time.time()
output = m(x566, x571)
end = time.time()
print(end-start)
