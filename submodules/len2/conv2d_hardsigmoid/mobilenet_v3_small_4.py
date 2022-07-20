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
        self.conv2d29 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid4 = Hardsigmoid()

    def forward(self, x84):
        x85=self.conv2d29(x84)
        x86=self.hardsigmoid4(x85)
        return x86

m = M().eval()
x84 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x84)
end = time.time()
print(end-start)