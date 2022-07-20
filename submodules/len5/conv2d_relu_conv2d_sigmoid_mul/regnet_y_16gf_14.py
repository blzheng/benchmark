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
        self.conv2d76 = Conv2d(1232, 308, kernel_size=(1, 1), stride=(1, 1))
        self.relu59 = ReLU()
        self.conv2d77 = Conv2d(308, 1232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()

    def forward(self, x240, x239):
        x241=self.conv2d76(x240)
        x242=self.relu59(x241)
        x243=self.conv2d77(x242)
        x244=self.sigmoid14(x243)
        x245=operator.mul(x244, x239)
        return x245

m = M().eval()
x240 = torch.randn(torch.Size([1, 1232, 1, 1]))
x239 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x240, x239)
end = time.time()
print(end-start)
