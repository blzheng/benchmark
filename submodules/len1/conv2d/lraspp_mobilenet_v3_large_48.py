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
        self.conv2d48 = Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x141):
        x142=self.conv2d48(x141)
        return x142

m = M().eval()
x141 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x141)
end = time.time()
print(end-start)
