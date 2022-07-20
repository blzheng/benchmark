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
        self.conv2d59 = Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x167, x172):
        x173=operator.mul(x167, x172)
        x174=self.conv2d59(x173)
        return x174

m = M().eval()
x167 = torch.randn(torch.Size([1, 672, 7, 7]))
x172 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x167, x172)
end = time.time()
print(end-start)
