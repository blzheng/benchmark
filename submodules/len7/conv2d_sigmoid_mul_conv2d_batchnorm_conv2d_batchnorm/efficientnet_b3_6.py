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
        self.conv2d122 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid24 = Sigmoid()
        self.conv2d123 = Conv2d(1392, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d124 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x378, x375):
        x379=self.conv2d122(x378)
        x380=self.sigmoid24(x379)
        x381=operator.mul(x380, x375)
        x382=self.conv2d123(x381)
        x383=self.batchnorm2d73(x382)
        x384=self.conv2d124(x383)
        x385=self.batchnorm2d74(x384)
        return x385

m = M().eval()
x378 = torch.randn(torch.Size([1, 58, 1, 1]))
x375 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x378, x375)
end = time.time()
print(end-start)
