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
        self.relu23 = ReLU()
        self.conv2d32 = Conv2d(52, 208, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d33 = Conv2d(208, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x97, x95, x89):
        x98=self.relu23(x97)
        x99=self.conv2d32(x98)
        x100=self.sigmoid5(x99)
        x101=operator.mul(x100, x95)
        x102=self.conv2d33(x101)
        x103=self.batchnorm2d21(x102)
        x104=operator.add(x89, x103)
        return x104

m = M().eval()
x97 = torch.randn(torch.Size([1, 52, 1, 1]))
x95 = torch.randn(torch.Size([1, 208, 14, 14]))
x89 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x97, x95, x89)
end = time.time()
print(end-start)
