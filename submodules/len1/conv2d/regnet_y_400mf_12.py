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
        self.conv2d12 = Conv2d(104, 104, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x35):
        x36=self.conv2d12(x35)
        return x36

m = M().eval()
x35 = torch.randn(torch.Size([1, 104, 28, 28]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)