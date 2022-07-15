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
        self.conv2d151 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x473):
        x474=self.conv2d151(x473)
        return x474

m = M().eval()
x473 = torch.randn(torch.Size([1, 50, 1, 1]))
start = time.time()
output = m(x473)
end = time.time()
print(end-start)
