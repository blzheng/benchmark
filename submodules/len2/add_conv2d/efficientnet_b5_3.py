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
        self.conv2d28 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x85, x70):
        x86=operator.add(x85, x70)
        x87=self.conv2d28(x86)
        return x87

m = M().eval()
x85 = torch.randn(torch.Size([1, 40, 56, 56]))
x70 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x85, x70)
end = time.time()
print(end-start)
