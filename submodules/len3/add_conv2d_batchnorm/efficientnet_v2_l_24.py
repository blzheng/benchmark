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
        self.conv2d83 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x274, x259):
        x275=operator.add(x274, x259)
        x276=self.conv2d83(x275)
        x277=self.batchnorm2d63(x276)
        return x277

m = M().eval()
x274 = torch.randn(torch.Size([1, 192, 14, 14]))
x259 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x274, x259)
end = time.time()
print(end-start)
