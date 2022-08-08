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
        self.conv2d97 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d98 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x306, x303, x297):
        x307=self.conv2d97(x306)
        x308=self.sigmoid18(x307)
        x309=operator.mul(x308, x303)
        x310=self.conv2d98(x309)
        x311=self.batchnorm2d60(x310)
        x312=operator.add(x297, x311)
        return x312

m = M().eval()
x306 = torch.randn(torch.Size([1, 726, 1, 1]))
x303 = torch.randn(torch.Size([1, 2904, 14, 14]))
x297 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x306, x303, x297)
end = time.time()
print(end-start)
