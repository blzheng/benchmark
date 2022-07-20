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
        self.batchnorm2d1 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d2 = Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x18):
        x19=self.batchnorm2d1(x18)
        x20=torch.nn.functional.relu(x19,inplace=True)
        x21=self.conv2d2(x20)
        x22=self.batchnorm2d2(x21)
        return x22

m = M().eval()
x18 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
