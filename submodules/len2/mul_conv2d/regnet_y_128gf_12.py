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
        self.conv2d68 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x212, x207):
        x213=operator.mul(x212, x207)
        x214=self.conv2d68(x213)
        return x214

m = M().eval()
x212 = torch.randn(torch.Size([1, 2904, 1, 1]))
x207 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x212, x207)
end = time.time()
print(end-start)
