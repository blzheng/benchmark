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
        self.conv2d10 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d10 = BatchNorm2d(96, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(32, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x28):
        x29=self.conv2d10(x28)
        x30=self.batchnorm2d10(x29)
        x31=self.relu7(x30)
        x32=self.conv2d11(x31)
        x33=self.batchnorm2d11(x32)
        return x33

m = M().eval()
x28 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x28)
end = time.time()
print(end-start)
