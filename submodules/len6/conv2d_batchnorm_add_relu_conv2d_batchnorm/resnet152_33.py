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
        self.conv2d96 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d96 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu91 = ReLU(inplace=True)
        self.conv2d97 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d97 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x316, x310):
        x317=self.conv2d96(x316)
        x318=self.batchnorm2d96(x317)
        x319=operator.add(x318, x310)
        x320=self.relu91(x319)
        x321=self.conv2d97(x320)
        x322=self.batchnorm2d97(x321)
        return x322

m = M().eval()
x316 = torch.randn(torch.Size([1, 256, 14, 14]))
x310 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x316, x310)
end = time.time()
print(end-start)
