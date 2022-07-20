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
        self.conv2d243 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x782, x767):
        x783=operator.add(x782, x767)
        x784=self.conv2d243(x783)
        return x784

m = M().eval()
x782 = torch.randn(torch.Size([1, 384, 7, 7]))
x767 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x782, x767)
end = time.time()
print(end-start)
