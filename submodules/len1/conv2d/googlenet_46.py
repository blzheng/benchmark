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
        self.conv2d46 = Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x166):
        x170=self.conv2d46(x166)
        return x170

m = M().eval()
x166 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x166)
end = time.time()
print(end-start)
