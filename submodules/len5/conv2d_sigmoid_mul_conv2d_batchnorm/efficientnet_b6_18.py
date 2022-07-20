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
        self.conv2d91 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d92 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x283, x280):
        x284=self.conv2d91(x283)
        x285=self.sigmoid18(x284)
        x286=operator.mul(x285, x280)
        x287=self.conv2d92(x286)
        x288=self.batchnorm2d54(x287)
        return x288

m = M().eval()
x283 = torch.randn(torch.Size([1, 36, 1, 1]))
x280 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x283, x280)
end = time.time()
print(end-start)
