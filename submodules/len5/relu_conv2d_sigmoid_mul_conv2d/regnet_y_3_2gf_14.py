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
        self.relu59 = ReLU()
        self.conv2d77 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d78 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x241, x239):
        x242=self.relu59(x241)
        x243=self.conv2d77(x242)
        x244=self.sigmoid14(x243)
        x245=operator.mul(x244, x239)
        x246=self.conv2d78(x245)
        return x246

m = M().eval()
x241 = torch.randn(torch.Size([1, 144, 1, 1]))
x239 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x241, x239)
end = time.time()
print(end-start)
