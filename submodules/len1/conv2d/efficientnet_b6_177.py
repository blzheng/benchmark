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
        self.conv2d177 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x554):
        x555=self.conv2d177(x554)
        return x555

m = M().eval()
x554 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x554)
end = time.time()
print(end-start)
