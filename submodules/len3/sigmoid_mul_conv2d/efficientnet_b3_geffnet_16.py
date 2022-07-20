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
        self.conv2d83 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x244, x240):
        x245=x244.sigmoid()
        x246=operator.mul(x240, x245)
        x247=self.conv2d83(x246)
        return x247

m = M().eval()
x244 = torch.randn(torch.Size([1, 816, 1, 1]))
x240 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x244, x240)
end = time.time()
print(end-start)