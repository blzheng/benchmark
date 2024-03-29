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
        self.conv2d105 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()
        self.conv2d106 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d62 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x328, x325):
        x329=self.conv2d105(x328)
        x330=self.sigmoid21(x329)
        x331=operator.mul(x330, x325)
        x332=self.conv2d106(x331)
        x333=self.batchnorm2d62(x332)
        return x333

m = M().eval()
x328 = torch.randn(torch.Size([1, 40, 1, 1]))
x325 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x328, x325)
end = time.time()
print(end-start)
