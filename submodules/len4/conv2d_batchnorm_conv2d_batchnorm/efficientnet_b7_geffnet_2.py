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
        self.batchnorm2d54 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x274):
        x275=self.conv2d91(x274)
        x276=self.batchnorm2d53(x275)
        x277=self.conv2d92(x276)
        x278=self.batchnorm2d54(x277)
        return x278

m = M().eval()
x274 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
