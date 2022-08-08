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
        self.conv2d26 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x86, x90):
        x87=self.conv2d26(x86)
        x88=self.batchnorm2d26(x87)
        x91=operator.add(x88, x90)
        x92=self.relu22(x91)
        x93=self.conv2d28(x92)
        x94=self.batchnorm2d28(x93)
        return x94

m = M().eval()
x86 = torch.randn(torch.Size([1, 256, 28, 28]))
x90 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x86, x90)
end = time.time()
print(end-start)
