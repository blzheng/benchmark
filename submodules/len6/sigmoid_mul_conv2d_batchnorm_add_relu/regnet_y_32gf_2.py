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
        self.sigmoid2 = Sigmoid()
        self.conv2d17 = Conv2d(696, 696, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(696, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)

    def forward(self, x49, x45, x39):
        x50=self.sigmoid2(x49)
        x51=operator.mul(x50, x45)
        x52=self.conv2d17(x51)
        x53=self.batchnorm2d11(x52)
        x54=operator.add(x39, x53)
        x55=self.relu12(x54)
        return x55

m = M().eval()
x49 = torch.randn(torch.Size([1, 696, 1, 1]))
x45 = torch.randn(torch.Size([1, 696, 28, 28]))
x39 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x49, x45, x39)
end = time.time()
print(end-start)
