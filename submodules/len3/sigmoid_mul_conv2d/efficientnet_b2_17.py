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
        self.sigmoid17 = Sigmoid()
        self.conv2d88 = Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x267, x263):
        x268=self.sigmoid17(x267)
        x269=operator.mul(x268, x263)
        x270=self.conv2d88(x269)
        return x270

m = M().eval()
x267 = torch.randn(torch.Size([1, 1248, 1, 1]))
x263 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x267, x263)
end = time.time()
print(end-start)
