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
        self.conv2d0 = Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batchnorm2d0 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x2, x8, x12):
        x3=operator.mul(x2, 0.458)
        x4=operator.add(x3, -0.030000000000000027)
        x13=torch.cat((x4, x8, x12), 1)
        x14=self.conv2d0(x13)
        x15=self.batchnorm2d0(x14)
        return x15

m = M().eval()
x2 = torch.randn(torch.Size([1, 1, 224, 224]))
x8 = torch.randn(torch.Size([1, 1, 224, 224]))
x12 = torch.randn(torch.Size([1, 1, 224, 224]))
start = time.time()
output = m(x2, x8, x12)
end = time.time()
print(end-start)
