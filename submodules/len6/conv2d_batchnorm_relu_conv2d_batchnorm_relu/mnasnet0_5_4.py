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
        self.conv2d12 = Conv2d(16, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(48, 48, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=48, bias=False)
        self.batchnorm2d13 = BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)

    def forward(self, x34):
        x35=self.conv2d12(x34)
        x36=self.batchnorm2d12(x35)
        x37=self.relu8(x36)
        x38=self.conv2d13(x37)
        x39=self.batchnorm2d13(x38)
        x40=self.relu9(x39)
        return x40

m = M().eval()
x34 = torch.randn(torch.Size([1, 16, 56, 56]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)
