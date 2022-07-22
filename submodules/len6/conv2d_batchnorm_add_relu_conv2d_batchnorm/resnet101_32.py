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
        self.conv2d93 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu88 = ReLU(inplace=True)
        self.conv2d94 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d94 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x306, x300):
        x307=self.conv2d93(x306)
        x308=self.batchnorm2d93(x307)
        x309=operator.add(x308, x300)
        x310=self.relu88(x309)
        x311=self.conv2d94(x310)
        x312=self.batchnorm2d94(x311)
        return x312

m = M().eval()
x306 = torch.randn(torch.Size([1, 256, 14, 14]))
x300 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x306, x300)
end = time.time()
print(end-start)