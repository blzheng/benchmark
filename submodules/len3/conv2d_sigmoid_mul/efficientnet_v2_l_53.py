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
        self.conv2d301 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid53 = Sigmoid()

    def forward(self, x968, x965):
        x969=self.conv2d301(x968)
        x970=self.sigmoid53(x969)
        x971=operator.mul(x970, x965)
        return x971

m = M().eval()
x968 = torch.randn(torch.Size([1, 96, 1, 1]))
x965 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x968, x965)
end = time.time()
print(end-start)
