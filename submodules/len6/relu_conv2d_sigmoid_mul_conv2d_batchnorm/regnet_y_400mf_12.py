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
        self.relu51 = ReLU()
        self.conv2d68 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d69 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x211, x209):
        x212=self.relu51(x211)
        x213=self.conv2d68(x212)
        x214=self.sigmoid12(x213)
        x215=operator.mul(x214, x209)
        x216=self.conv2d69(x215)
        x217=self.batchnorm2d43(x216)
        return x217

m = M().eval()
x211 = torch.randn(torch.Size([1, 110, 1, 1]))
x209 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x211, x209)
end = time.time()
print(end-start)
