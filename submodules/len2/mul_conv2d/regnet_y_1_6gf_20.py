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
        self.conv2d108 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x340, x335):
        x341=operator.mul(x340, x335)
        x342=self.conv2d108(x341)
        return x342

m = M().eval()
x340 = torch.randn(torch.Size([1, 336, 1, 1]))
x335 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x340, x335)
end = time.time()
print(end-start)
