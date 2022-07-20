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
        self.conv2d222 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x714, x709):
        x715=operator.mul(x714, x709)
        x716=self.conv2d222(x715)
        return x716

m = M().eval()
x714 = torch.randn(torch.Size([1, 2304, 1, 1]))
x709 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x714, x709)
end = time.time()
print(end-start)
