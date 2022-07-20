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
        self.conv2d87 = Conv2d(52, 1248, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d88 = Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x266, x263):
        x267=self.conv2d87(x266)
        x268=self.sigmoid17(x267)
        x269=operator.mul(x268, x263)
        x270=self.conv2d88(x269)
        return x270

m = M().eval()
x266 = torch.randn(torch.Size([1, 52, 1, 1]))
x263 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x266, x263)
end = time.time()
print(end-start)
