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
        self.conv2d281 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid49 = Sigmoid()
        self.conv2d282 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x904, x901):
        x905=self.conv2d281(x904)
        x906=self.sigmoid49(x905)
        x907=operator.mul(x906, x901)
        x908=self.conv2d282(x907)
        return x908

m = M().eval()
x904 = torch.randn(torch.Size([1, 96, 1, 1]))
x901 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x904, x901)
end = time.time()
print(end-start)
