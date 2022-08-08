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
        self.conv2d4 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)

    def forward(self, x6, x14):
        x15=self.conv2d4(x6)
        x16=self.batchnorm2d4(x15)
        x17=operator.add(x14, x16)
        x18=self.relu1(x17)
        return x18

m = M().eval()
x6 = torch.randn(torch.Size([1, 64, 56, 56]))
x14 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x6, x14)
end = time.time()
print(end-start)
