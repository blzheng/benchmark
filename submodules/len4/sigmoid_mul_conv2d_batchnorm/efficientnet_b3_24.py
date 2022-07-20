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
        self.sigmoid24 = Sigmoid()
        self.conv2d123 = Conv2d(1392, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x379, x375):
        x380=self.sigmoid24(x379)
        x381=operator.mul(x380, x375)
        x382=self.conv2d123(x381)
        x383=self.batchnorm2d73(x382)
        return x383

m = M().eval()
x379 = torch.randn(torch.Size([1, 1392, 1, 1]))
x375 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x379, x375)
end = time.time()
print(end-start)