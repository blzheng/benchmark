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
        self.conv2d34 = Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)

    def forward(self, x101):
        x102=self.conv2d34(x101)
        return x102

m = M().eval()
x101 = torch.randn(torch.Size([1, 184, 14, 14]))
start = time.time()
output = m(x101)
end = time.time()
print(end-start)
