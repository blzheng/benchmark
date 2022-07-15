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
        self.conv2d193 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x606):
        x607=self.conv2d193(x606)
        return x607

m = M().eval()
x606 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x606)
end = time.time()
print(end-start)
