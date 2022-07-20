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
        self.conv2d61 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x197, x188):
        x198=operator.add(x197, x188)
        x199=self.conv2d61(x198)
        return x199

m = M().eval()
x197 = torch.randn(torch.Size([1, 184, 7, 7]))
x188 = torch.randn(torch.Size([1, 184, 7, 7]))
start = time.time()
output = m(x197, x188)
end = time.time()
print(end-start)
