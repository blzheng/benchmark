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
        self.maxpool2d9 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d44 = Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x128, x134, x140, x144):
        x145=torch.cat([x128, x134, x140, x144], 1)
        x161=self.maxpool2d9(x145)
        x162=self.conv2d44(x161)
        x163=self.batchnorm2d44(x162)
        return x163

m = M().eval()
x128 = torch.randn(torch.Size([1, 112, 14, 14]))
x134 = torch.randn(torch.Size([1, 288, 14, 14]))
x140 = torch.randn(torch.Size([1, 64, 14, 14]))
x144 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x128, x134, x140, x144)
end = time.time()
print(end-start)
