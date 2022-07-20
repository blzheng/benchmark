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
        self.batchnorm2d27 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x86, x79):
        x87=self.batchnorm2d27(x86)
        x88=operator.add(x79, x87)
        x89=self.relu24(x88)
        x90=self.conv2d28(x89)
        x91=self.batchnorm2d28(x90)
        return x91

m = M().eval()
x86 = torch.randn(torch.Size([1, 720, 14, 14]))
x79 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x86, x79)
end = time.time()
print(end-start)
