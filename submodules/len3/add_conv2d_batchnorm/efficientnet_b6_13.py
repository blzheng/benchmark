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
        self.conv2d88 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(864, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x273, x258):
        x274=operator.add(x273, x258)
        x275=self.conv2d88(x274)
        x276=self.batchnorm2d52(x275)
        return x276

m = M().eval()
x273 = torch.randn(torch.Size([1, 144, 14, 14]))
x258 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x273, x258)
end = time.time()
print(end-start)
