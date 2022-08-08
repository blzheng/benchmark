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
        self.conv2d76 = Conv2d(2904, 726, kernel_size=(1, 1), stride=(1, 1))
        self.relu59 = ReLU()
        self.conv2d77 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d78 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x240, x239, x233):
        x241=self.conv2d76(x240)
        x242=self.relu59(x241)
        x243=self.conv2d77(x242)
        x244=self.sigmoid14(x243)
        x245=operator.mul(x244, x239)
        x246=self.conv2d78(x245)
        x247=self.batchnorm2d48(x246)
        x248=operator.add(x233, x247)
        return x248

m = M().eval()
x240 = torch.randn(torch.Size([1, 2904, 1, 1]))
x239 = torch.randn(torch.Size([1, 2904, 14, 14]))
x233 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x240, x239, x233)
end = time.time()
print(end-start)
