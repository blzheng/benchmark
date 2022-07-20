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
        self.conv2d82 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d83 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x265, x262):
        x266=self.conv2d82(x265)
        x267=self.sigmoid11(x266)
        x268=operator.mul(x267, x262)
        x269=self.conv2d83(x268)
        return x269

m = M().eval()
x265 = torch.randn(torch.Size([1, 44, 1, 1]))
x262 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x265, x262)
end = time.time()
print(end-start)
