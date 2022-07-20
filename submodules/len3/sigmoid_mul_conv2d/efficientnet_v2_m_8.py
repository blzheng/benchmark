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
        self.sigmoid8 = Sigmoid()
        self.conv2d68 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x218, x214):
        x219=self.sigmoid8(x218)
        x220=operator.mul(x219, x214)
        x221=self.conv2d68(x220)
        return x221

m = M().eval()
x218 = torch.randn(torch.Size([1, 1056, 1, 1]))
x214 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x218, x214)
end = time.time()
print(end-start)
