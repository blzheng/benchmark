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
        self.conv2d87 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d88 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x268, x265):
        x269=self.conv2d87(x268)
        x270=self.sigmoid17(x269)
        x271=operator.mul(x270, x265)
        x272=self.conv2d88(x271)
        return x272

m = M().eval()
x268 = torch.randn(torch.Size([1, 40, 1, 1]))
x265 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x268, x265)
end = time.time()
print(end-start)
