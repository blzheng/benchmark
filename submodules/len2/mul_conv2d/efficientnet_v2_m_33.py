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
        self.conv2d193 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x617, x612):
        x618=operator.mul(x617, x612)
        x619=self.conv2d193(x618)
        return x619

m = M().eval()
x617 = torch.randn(torch.Size([1, 1824, 1, 1]))
x612 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x617, x612)
end = time.time()
print(end-start)
