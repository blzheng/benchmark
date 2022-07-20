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
        self.conv2d44 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid4 = Hardsigmoid()

    def forward(self, x128):
        x129=self.conv2d44(x128)
        x130=self.hardsigmoid4(x129)
        return x130

m = M().eval()
x128 = torch.randn(torch.Size([1, 168, 1, 1]))
start = time.time()
output = m(x128)
end = time.time()
print(end-start)
