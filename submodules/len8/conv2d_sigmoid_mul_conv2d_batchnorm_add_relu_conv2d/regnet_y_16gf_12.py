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
        self.conv2d67 = Conv2d(308, 1232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d68 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x210, x207, x201):
        x211=self.conv2d67(x210)
        x212=self.sigmoid12(x211)
        x213=operator.mul(x212, x207)
        x214=self.conv2d68(x213)
        x215=self.batchnorm2d42(x214)
        x216=operator.add(x201, x215)
        x217=self.relu52(x216)
        x218=self.conv2d69(x217)
        return x218

m = M().eval()
x210 = torch.randn(torch.Size([1, 308, 1, 1]))
x207 = torch.randn(torch.Size([1, 1232, 14, 14]))
x201 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x210, x207, x201)
end = time.time()
print(end-start)
