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
        self.conv2d111 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x330):
        x331=x330.mean((2, 3),keepdim=True)
        x332=self.conv2d111(x331)
        return x332

m = M().eval()
x330 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x330)
end = time.time()
print(end-start)
