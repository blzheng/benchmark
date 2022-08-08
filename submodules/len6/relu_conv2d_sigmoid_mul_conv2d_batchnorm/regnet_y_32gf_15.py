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
        self.relu63 = ReLU()
        self.conv2d82 = Conv2d(348, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d83 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x257, x255):
        x258=self.relu63(x257)
        x259=self.conv2d82(x258)
        x260=self.sigmoid15(x259)
        x261=operator.mul(x260, x255)
        x262=self.conv2d83(x261)
        x263=self.batchnorm2d51(x262)
        return x263

m = M().eval()
x257 = torch.randn(torch.Size([1, 348, 1, 1]))
x255 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x257, x255)
end = time.time()
print(end-start)
