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
        self.relu57 = ReLU(inplace=True)
        self.conv2d61 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)
        self.conv2d62 = Conv2d(896, 896, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=7, bias=False)
        self.batchnorm2d62 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu59 = ReLU(inplace=True)

    def forward(self, x198):
        x199=self.relu57(x198)
        x200=self.conv2d61(x199)
        x201=self.batchnorm2d61(x200)
        x202=self.relu58(x201)
        x203=self.conv2d62(x202)
        x204=self.batchnorm2d62(x203)
        x205=self.relu59(x204)
        return x205

m = M().eval()
x198 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x198)
end = time.time()
print(end-start)
