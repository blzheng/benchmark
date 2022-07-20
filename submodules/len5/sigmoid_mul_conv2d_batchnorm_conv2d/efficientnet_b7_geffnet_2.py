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
        self.conv2d91 = Conv2d(480, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d92 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x272, x268):
        x273=x272.sigmoid()
        x274=operator.mul(x268, x273)
        x275=self.conv2d91(x274)
        x276=self.batchnorm2d53(x275)
        x277=self.conv2d92(x276)
        return x277

m = M().eval()
x272 = torch.randn(torch.Size([1, 480, 1, 1]))
x268 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x272, x268)
end = time.time()
print(end-start)
