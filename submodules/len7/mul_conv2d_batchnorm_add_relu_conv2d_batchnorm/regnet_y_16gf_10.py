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
        self.conv2d58 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x180, x175, x169):
        x181=operator.mul(x180, x175)
        x182=self.conv2d58(x181)
        x183=self.batchnorm2d36(x182)
        x184=operator.add(x169, x183)
        x185=self.relu44(x184)
        x186=self.conv2d59(x185)
        x187=self.batchnorm2d37(x186)
        return x187

m = M().eval()
x180 = torch.randn(torch.Size([1, 1232, 1, 1]))
x175 = torch.randn(torch.Size([1, 1232, 14, 14]))
x169 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x180, x175, x169)
end = time.time()
print(end-start)
