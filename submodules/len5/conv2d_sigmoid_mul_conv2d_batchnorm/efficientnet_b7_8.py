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
        self.conv2d40 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d41 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x124, x121):
        x125=self.conv2d40(x124)
        x126=self.sigmoid8(x125)
        x127=operator.mul(x126, x121)
        x128=self.conv2d41(x127)
        x129=self.batchnorm2d23(x128)
        return x129

m = M().eval()
x124 = torch.randn(torch.Size([1, 12, 1, 1]))
x121 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x124, x121)
end = time.time()
print(end-start)
