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
        self.conv2d113 = Conv2d(960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x334, x330):
        x335=x334.sigmoid()
        x336=operator.mul(x330, x335)
        x337=self.conv2d113(x336)
        return x337

m = M().eval()
x334 = torch.randn(torch.Size([1, 960, 1, 1]))
x330 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x334, x330)
end = time.time()
print(end-start)
