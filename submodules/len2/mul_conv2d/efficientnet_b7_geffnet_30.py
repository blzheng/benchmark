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
        self.conv2d151 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x446, x451):
        x452=operator.mul(x446, x451)
        x453=self.conv2d151(x452)
        return x453

m = M().eval()
x446 = torch.randn(torch.Size([1, 1344, 14, 14]))
x451 = torch.randn(torch.Size([1, 1344, 1, 1]))
start = time.time()
output = m(x446, x451)
end = time.time()
print(end-start)
